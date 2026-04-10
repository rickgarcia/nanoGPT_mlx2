import os
import math
import time
import json

import numpy as np
from typing import List

import mlx
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

import wandb

from model import GPTConfig, GPT
from optimizer import AdamW


# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
d_type = 'float32'

# adamw optimizer
learning_rate = 6.0e-4 # max learning rate
min_lr = 6.0e-5
num_iters = 600000 # total number of training iterations
warmup_pct = 0.1
warmup_iters = 2000
lr_decay_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
meta_vocab_size = None

# dataset
dataset = 'openwebtext'
batch_size = 1
gradient_accumulation_steps = 512
context_size = 1024

# eval
save_interval = 1
eval_interval = 10
log_interval = 10
eval_only = False
early_stopping_patience = 5 # stop after N evals with no val loss improvement (0 to disable)
early_stopping_max_gap = 0.0 # stop when train/val loss gap exceeds this ratio (0 to disable)
init_from = 'resume' # 'scratch' or 'resume'
out_dir = 'gpt2_openwebtext_pretrain'

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# Load vocab and dataset:
# poor man's data loader
data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

# model save path
save_model_path = os.path.join(out_dir, out_dir + '.npz')
save_model_config_path = os.path.join(out_dir, out_dir + '.json')

os.makedirs(out_dir, exist_ok=True)

# initialize wandb logging
wandb.init(project=out_dir, config=config)


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = np.random.randint(len(data) - context_size, size=(batch_size,))
    x = mx.stack([(mx.array(data[i:i+context_size])) for i in ix]).astype(mx.int64)
    y = mx.stack([(mx.array(data[i+1:i+1+context_size])) for i in ix]).astype(mx.int64)
    return x, y


def print_loss(optimizer, iteration_count, average_loss, tic):
    toc = time.perf_counter()
    print(
        f"iter {iteration_count}: train loss {average_loss:.3f}, "
        f"it/sec {1.0 / (toc - tic):.3f}, "
        f"lr {optimizer.learning_rate.item():.9f}"
    )
    return toc


def update_learning_rate(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (
        lr_decay_iters - warmup_iters
    )
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    new_lr = min_lr + coeff * (learning_rate - min_lr)
    return new_lr
    

def log_wandb(log_dict, itr):
    wandb.log(log_dict, step=itr)


def main():
    # model init
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=context_size,
                    bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line

    # initialize model:
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    print(model)

    weights = tree_map(lambda p: p.astype(getattr(mx, d_type)), model.parameters())
    model.update(weights)

    # resume from checkpoint if available, preferring best checkpoint over regular
    resume_iter = 0
    best_model_path = os.path.join(out_dir, out_dir + '_best.npz')
    best_model_config_path = save_model_config_path.replace('.json', '_best.json')
    if init_from == 'resume':
        if os.path.exists(best_model_path) and os.path.exists(best_model_config_path):
            print(f"Resuming from best checkpoint: {best_model_path}")
            model.load_weights(best_model_path)
            with open(best_model_config_path, "r") as f:
                checkpoint_meta = json.load(f)
            resume_iter = checkpoint_meta.get('iter_num', 0)
            print(f"Resuming from iteration {resume_iter}")
        elif os.path.exists(save_model_path) and os.path.exists(save_model_config_path):
            print(f"Resuming from checkpoint: {save_model_path}")
            model.load_weights(save_model_path)
            with open(save_model_config_path, "r") as f:
                checkpoint_meta = json.load(f)
            resume_iter = checkpoint_meta.get('iter_num', 0)
            print(f"Resuming from iteration {resume_iter}")
        else:
            print("No checkpoint found, initializing from scratch")
    else:
        print("Initializing model from scratch")

    mx.eval(model.parameters())
    nparams = sum(
        x.size for k, x in tree_flatten(model.parameters()) if "embedding" not in k
    )
    print(f"Training a transformer with {nparams / 1024**2:.3f} M parameters")


    def loss_fn(model, x, y, reduce=True):
        logits = model(x)
        losses = nn.losses.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), y.reshape(-1)
        )
        return mx.mean(losses) if reduce else mx.mean(losses, axis=(-1, -2))


    # setup optimizer
    optimizer = AdamW(learning_rate=learning_rate, 
                            betas=[beta1, beta2], 
                            weight_decay=weight_decay)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)


    def step(inputs, targets, gradient_accumulation_steps):
        # gradient accumulation
        accumulated_grads = tree_map(
                    lambda x: mx.zeros_like(x), model.parameters()
                )
        accumulated_loss = 0.0
        for _ in range(gradient_accumulation_steps):
            loss, grads = loss_and_grad_fn(model, X, Y)

            accumulated_grads = tree_map(
                lambda acc, new: acc + new * (1.0 / gradient_accumulation_steps),
                accumulated_grads,
                grads,
            )

            tree_map(
                lambda grad: mx.eval(grad),
                accumulated_grads,
            )

            accumulated_loss += loss.item()

        # scale the loss to account for gradient accumulation
        loss = mx.array(accumulated_loss / gradient_accumulation_steps) 

        optimizer.update(model, accumulated_grads)

        accumulated_grads = tree_map(
            lambda x: mx.zeros_like(x), model.parameters()
        )
        return loss

    def eval_step(x, y):
        return loss_fn(model, x, y)

    def estimate_loss():
        """Evaluate train and val loss over eval_iters batches."""
        model.eval()  # disable dropout during evaluation
        out = {}
        for split in ['train', 'val']:
            losses = mx.zeros(eval_iters)
            for i in range(eval_iters):
                x, y = get_batch(split)
                losses[i] = eval_step(x, y)
            mx.eval(losses)
            out[split] = losses.mean().item()
        model.train()  # re-enable dropout for training
        return out

    # fetch the first batch of samples.
    X, Y = get_batch('train')

    state = [model.state, optimizer.state]

    tic = time.perf_counter()
    local_iter_num = 0 # number of iterations in the lifetime of this process
    iter_num = resume_iter
    
    best_val_loss = float('inf')
    evals_without_improvement = 0
    best_model_path = os.path.join(out_dir, out_dir + '_best.npz')
    last_saved_train_loss = float('inf')
    last_saved_iter = -1

    while True:
        if iter_num == 0 and eval_only:
            break

        # evaluate train/val loss periodically
        if iter_num % eval_interval == 0:
            losses = estimate_loss()
            print(f"eval iter {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            log_wandb({'eval/train_loss': losses['train'], 'eval/val_loss': losses['val']}, iter_num)

            # early stopping: track best val loss and save best model
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                evals_without_improvement = 0
                # save best model
                flat_params = tree_flatten(model.parameters())
                mx.savez(best_model_path, **dict(flat_params))
                checkpoint_meta = model.config.__dict__.copy()
                checkpoint_meta['iter_num'] = iter_num
                with open(save_model_config_path.replace('.json', '_best.json'), "w") as f:
                    json.dump(checkpoint_meta, f)
                print(f"  ** new best val loss: {best_val_loss:.4f}, saved to {best_model_path}")
            else:
                evals_without_improvement += 1
                print(f"  val loss did not improve ({evals_without_improvement}/{early_stopping_patience})")

            if early_stopping_patience > 0 and evals_without_improvement >= early_stopping_patience:
                print(f"Early stopping: val loss has not improved for {early_stopping_patience} evals. "
                      f"Best val loss: {best_val_loss:.4f}")
                break

            # early stopping based on train/val gap (overfitting detection)
            if early_stopping_max_gap > 0 and losses['train'] > 0:
                gap_ratio = losses['val'] / losses['train']
                print(f"  train/val gap ratio: {gap_ratio:.2f}x")
                if gap_ratio > early_stopping_max_gap:
                    print(f"Early stopping: train/val gap ratio {gap_ratio:.2f}x exceeds "
                          f"threshold {early_stopping_max_gap:.2f}x. Best val loss: {best_val_loss:.4f}")
                    break

        # lr schedule
        new_lr = update_learning_rate(iter_num)
        optimizer.set_learning_rate(new_lr)

        # mx.simplify(loss, model.parameters())
        loss = step(X, Y, gradient_accumulation_steps)
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')

        tic = print_loss(optimizer, iter_num, loss.item(), tic)

        mx.eval(state)

        if iter_num % log_interval == 0:
            log_wandb({'train/loss': loss.item(), 'train/lr': new_lr}, iter_num)
        
        # save if loss dropped >10% from last save, or every save_interval iters (but not iter 0)
        loss_improved = loss.item() < last_saved_train_loss * 0.9
        interval_reached = iter_num > 0 and (iter_num - last_saved_iter) >= save_interval
        if local_iter_num > 0 and (loss_improved or interval_reached):
            flat_params = tree_flatten(model.parameters())
            mx.savez(save_model_path, **dict(flat_params))
            checkpoint_meta = model.config.__dict__.copy()
            checkpoint_meta['iter_num'] = iter_num
            with open(save_model_config_path, "w") as f:
                json.dump(checkpoint_meta, f)
            last_saved_train_loss = loss.item()
            last_saved_iter = iter_num
            print(f"  checkpoint saved at iter {iter_num} (loss={loss.item():.4f})")

        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > num_iters:
            break

if __name__ == "__main__":
    main()
