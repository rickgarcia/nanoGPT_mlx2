# train a miniature character-level shakespeare model
# matches karpathy's nanoGPT config for direct comparison
out_dir = 'out-shakespeare-char'
dataset = 'shakespeare_char'
meta_vocab_size = 65

# these match karpathy's settings exactly
gradient_accumulation_steps = 1
batch_size = 64
context_size = 256

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.0

learning_rate = 1e-3
num_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100

# eval stuff
save_interval = 1000
eval_interval = 250
eval_iters = 200
log_interval = 10
early_stopping_patience = 0  # disable early stopping for full run
