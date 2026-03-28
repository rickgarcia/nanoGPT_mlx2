# `mlx_nanoGPT`

An attempted port of Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) in Apple's machine learning framework, [MLX](https://github.com/ml-explore/mlx).

Train OpenAI's GPT-2 models or custom GPT-style models from scratch

Almost works.


Dependencies:
- [mlx](https://ml-explore.github.io/mlx/build/html/index.html)
- [numpy](https://numpy.org/install/)
-  `datasets` for huggingface datasets (if you want to download + preprocess OpenWebText)
-  `tiktoken` for OpenAI's fast BPE code
-  `tensorboardX` for optional logging
-  `tqdm` for progress bars

## Simple Start
Prepare the shakespeare dataset similar to nanoGPT. Creates a `train.bin` and `val.bin` in that data directory.
```bash
python data/shakespeare/prepare.py
```

Training on MLX GPU:
```bash
python train.py configs/train_gpt2_shakespeare.py
```

So once the training finishes we can sample from the best model by pointing the sampling script at this directory:
```bash
python sample.py --out_dir=gpt2_small_shakespeare
```

## openwebtext
To train a GPT-2 model on OpenWebText similar to nanoGPT, first prepare the dataset:
```bash
python data/openwebtext/prepare.py
```

Then, train a 124M GPT-2 model on your MAC GPU:
```bash
python train.py configs/train_gpt2_owt.py
```

## todos
- [ ] disable weight decay on non-decay params in optimizer
- [ ] add bfloat16 training support
- [ ] integrate Eleuther Eval
- [ ] add checkpoint conversion for loading pre-trained HF models
- [x] add saveing and loading pre-trained MLX models
- [ ] enable finetuning models from pre-trained checkpoints
- [x] enable inference with pre-trained models

## issues so far
- `mx.where` with `float('-inf')` produces NaN — use `-1e9` as a workaround for large negative values
- `mx.random.categorical` can get stuck in repetition loops due to MLX lazy evaluation / random state not advancing — currently using numpy for sampling as a workaround
- `@mx.compile` can cache initial model weights, preventing parameter updates from being seen — avoid using on eval functions during training
- Overfitting causes repetitive/degenerate output (e.g. "be be be", "!!!") — early stopping with patience and train/val gap ratio detection added to mitigate; output quality degrades when val/train loss gap exceeds ~2x
- `tiktoken` cache lives in `/tmp` and is cleared on reboot — set `TIKTOKEN_CACHE_DIR` to a persistent location to avoid re-downloading
- Padded vocab tokens (50257-50303) have no tiktoken mapping and cause pyo3 panic on decode — filter with `[t for t in tokens if t < enc.n_vocab]`

## acknowledgements
[Andrej Karpthy](https://github.com/karpathy) for nanoGPT
