import os
import pickle
import tiktoken
import time
import json

import mlx.core as mx
from mlx.utils import tree_unflatten, tree_flatten

from model import GPT, GPTConfig


init_from = 'resume'   # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out'        # ignored if init_from is not 'resume'
start = "\n"           # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10       # number of samples to draw
max_new_tokens = 256   # number of tokens generated in each sample
temperature = 0.8      # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200            # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337

# overrides from command line or config file
exec(open('configurator.py').read()) 

model_weights_path = os.path.join(out_dir, out_dir + '_best.npz')
model_config_path = os.path.join(out_dir, out_dir + '_best.json')

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    with open(model_config_path, "r") as f:
        config_args = json.load(f)

    config_args.pop('iter_num', None)
    config = GPTConfig(**config_args)
    model = GPT(config)

    weights = mx.load(model_weights_path)
    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())
    model.eval()  # disable dropout for inference

    nparams = sum(x.size for k, x in tree_flatten(model.parameters()))
    print(f"Loaded GPT-2 with {nparams / 1e6:.3f} M parameters")

elif init_from.startswith('gpt2'):
    # TODO
    raise NotImplementedError("This feature/functionality is not yet implemented.")

# detect character-level vs BPE encoding
# search data directories for a meta.pkl with matching vocab_size
meta_path = None
for d in os.listdir('data'):
    candidate = os.path.join('data', d, 'meta.pkl')
    if os.path.exists(candidate):
        with open(candidate, 'rb') as f:
            meta = pickle.load(f)
        if meta['vocab_size'] == config.vocab_size:
            meta_path = candidate
            break

if meta_path:
    stoi, itos = meta['stoi'], meta['itos']
    vocab_size = meta['vocab_size']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l if i < vocab_size])
    print(f"Using character-level encoding (vocab_size={vocab_size})")
else:
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode([t for t in l if t < vocab_size])
    print(f"Using tiktoken GPT-2 BPE encoding (vocab_size={vocab_size})")

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (mx.array([start_ids], dtype=mx.uint32))

# run generation
start = time.time()
for k in range(num_samples):
    y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
    tokens = [t for t in y[0].tolist() if t < vocab_size]
    print(decode(tokens))
end = time.time()
