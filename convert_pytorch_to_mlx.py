"""Convert a PyTorch nanoGPT checkpoint to MLX format (.npz + .json)."""
import argparse
import json
import re

import numpy as np
import torch
import mlx.core as mx


def convert(pt_path, out_dir):
    ckpt = torch.load(pt_path, map_location='cpu', weights_only=False)

    model_args = ckpt['model_args']
    print(f"Model config: {model_args}")
    print(f"Iter: {ckpt.get('iter_num', 'N/A')}")
    print(f"Best val loss: {ckpt.get('best_val_loss', 'N/A')}")

    key_map = {
        '_orig_mod.transformer.wte.weight': 'wte.weight',
        '_orig_mod.transformer.wpe.weight': 'wpe.weight',
        '_orig_mod.transformer.ln_f.weight': 'ln_f.weight',
        '_orig_mod.lm_head.weight': 'out_proj.weight',
    }

    # Also handle non-compiled checkpoints (no _orig_mod prefix)
    key_map_no_compile = {
        'transformer.wte.weight': 'wte.weight',
        'transformer.wpe.weight': 'wpe.weight',
        'transformer.ln_f.weight': 'ln_f.weight',
        'lm_head.weight': 'out_proj.weight',
    }

    def map_key(pt_key):
        # Check direct mappings first
        if pt_key in key_map:
            return key_map[pt_key]
        if pt_key in key_map_no_compile:
            return key_map_no_compile[pt_key]

        # transformer.h.N.* -> transformer.N.*
        # Handle both _orig_mod.transformer.h.N.* and transformer.h.N.*
        m = re.match(r'(?:_orig_mod\.)?transformer\.h\.(\d+)\.(.*)', pt_key)
        if m:
            layer_idx = m.group(1)
            rest = m.group(2)
            return f'transformer.{layer_idx}.{rest}'

        # Handle bias keys the same way
        if 'ln_f.bias' in pt_key:
            return 'ln_f.bias'

        return None

    mlx_weights = {}
    unmapped = []

    for pt_key, pt_val in ckpt['model'].items():
        mlx_key = map_key(pt_key)
        if mlx_key is None:
            unmapped.append(pt_key)
            continue
        mlx_weights[mlx_key] = mx.array(pt_val.numpy())

    if unmapped:
        print(f"\nWarning: {len(unmapped)} unmapped keys:")
        for k in unmapped:
            print(f"  {k}")

    print(f"\nConverted {len(mlx_weights)} weight tensors")

    # Save weights
    import os
    os.makedirs(out_dir, exist_ok=True)
    npz_path = os.path.join(out_dir, out_dir + '.npz')
    json_path = os.path.join(out_dir, out_dir + '.json')

    mx.savez(npz_path, **mlx_weights)
    print(f"Saved weights to {npz_path}")

    # Save config (same format as train.py saves)
    config_out = {
        'n_layer': model_args['n_layer'],
        'n_head': model_args['n_head'],
        'n_embd': model_args['n_embd'],
        'block_size': model_args['block_size'],
        'bias': model_args['bias'],
        'vocab_size': model_args['vocab_size'],
        'dropout': model_args['dropout'],
        'iter_num': ckpt.get('iter_num', 0),
    }
    with open(json_path, 'w') as f:
        json.dump(config_out, f)
    print(f"Saved config to {json_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pt_path', help='Path to PyTorch .pt checkpoint')
    parser.add_argument('--out_dir', default='converted_model', help='Output directory name')
    args = parser.parse_args()
    convert(args.pt_path, args.out_dir)
