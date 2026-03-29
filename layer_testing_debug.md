# Debugging the Inverted Causal Attention Mask

Summary of the debugging approach used to isolate an inverted argument in `mx.where` that broke both training and inference in the initial MLX port of nanoGPT.

## Background

After initial training runs, nanoGPT_mlx was producing degenerate output during inference — repetitive tokens ("be be be", "!!!", "EOEOEO") regardless of training duration, dataset, or hyperparameter tuning. Training loss would decrease, but generated text never became coherent. Early stopping, dropout tuning, and various sampling fixes (numpy fallback, top-k rewrite) were applied but none resolved the core issue.

The fix was found after introducing a known-good reference: a PyTorch checkpoint trained with Karpathy's base nanoGPT implementation on the same architecture and data.

## Methodology

### Step 1: Establish a Known-Good Reference

A Shakespeare character-level model was trained using Karpathy's (relatively) unmodified nanoGPT (PyTorch) to iteration 1750, achieving val loss 1.47. At this loss level, the model produces recognizable (if imperfect) Shakespeare-like text. This checkpoint serves as ground truth — if the MLX model can't reproduce its outputs given identical weights, the bug is in the MLX forward pass.

Config: `n_layer=6, n_head=6, n_embd=384, block_size=256, vocab_size=65, bias=False, dropout=0.0`

### Step 2: Weight Conversion and Verification

A conversion script (`convert_pytorch_to_mlx.py`) was written to map PyTorch state dict keys to the MLX model's parameter tree:

| PyTorch key | MLX key |
|---|---|
| `_orig_mod.transformer.wte.weight` | `wte.weight` |
| `_orig_mod.transformer.h.N.*` | `transformer.N.*` |
| `_orig_mod.transformer.ln_f.weight` | `ln_f.weight` |
| `_orig_mod.lm_head.weight` | `out_proj.weight` |

The `_orig_mod.` prefix comes from `torch.compile()`. No transposition was needed — both PyTorch and MLX `nn.Linear` store weights as `[out_features, in_features]`.

Weight identity was verified by loading both formats and comparing element-wise:

```python
for pt_key, mlx_key in key_pairs:
    pt_w = pt_weights[pt_key].numpy()
    mlx_w = np.array(mlx_weights[mlx_key])
    diff = np.abs(pt_w - mlx_w).max()
    # Result: max_diff = 0.00000000 for all weight tensors
```

This confirmed the conversion introduced zero error.

### Step 3: End-to-End Generation Test

The converted model was loaded into the MLX `GPT` class and text was generated using `model.generate()`:

```
CAMA:
t canounorenoreanononer wher tun ncce te puncete
anye, I se t unt n m. t st t t mbouruse he bo f theds...
```

Garbage. With identical weights and val loss 1.47, this immediately points to a bug in the MLX model's forward pass or generation loop.

### Step 4: Isolating Forward Pass vs. Generation Loop

To determine whether the bug was in the forward pass itself or in the autoregressive generation loop, a single forward pass was tested:

```python
test = "\nTo be or not"
ids = [stoi[c] for c in test]
x = mx.array([ids], dtype=mx.uint32)
logits = model(x)

# Top predictions for next token:
# 'h': 8.2527   (for "nothing")
# 'o': 7.6224   (for "not to")
```

This looked correct — reasonable predictions with sensible logit magnitudes. This initially suggested the generation loop was at fault, not the forward pass.

### Step 5: Manual Generation Loop vs. model.generate()

To test this hypothesis, the generation loop was reimplemented outside of `model.generate()`, stepping through token-by-token:

```python
for step in range(10):
    idx_cond = idx if idx.shape[1] <= 256 else idx[:, -256:]
    logits = model(idx_cond)
    logits = logits[:, -1, :] / temperature
    mx.eval(logits)
    # ... numpy sampling ...
    idx = mx.concatenate([idx, mx.expand_dims(idx_next, axis=0)], axis=1)
```

Result after 10 steps: `"\nthing t ou"` — coherent text. But `model.generate()` with the same logic produced garbage.

This was a red herring. The manual loop happened to work for short sequences because the causal mask has less impact on short contexts (there are fewer future positions to attend to, and the model can partially compensate). The divergence becomes catastrophic as sequence length grows.

### Step 6: Tracking Degradation Over Sequence Length

To confirm the length-dependent degradation, generation was traced over 200 steps with entropy and output quality logged at checkpoints:

```
Step  10: entropy=1.217, last10='she casipa'
Step  20: entropy=1.696, last10='since s he'
Step  50: entropy=1.488, last10=' bunoouncu'
Step 100: entropy=1.899, last10='nororoo ba'
Step 150: entropy=1.380, last10='ne d\nce er'
Step 199: entropy=0.945, last10='nonoflonon'
```

The output degrades progressively. By step 50, the model is already producing nonsense syllables. By step 199, entropy has collapsed and the model is stuck in repetitive loops. This pattern is consistent with corrupted attention — as the context fills with garbage generated from broken attention, each subsequent prediction gets worse, creating a compounding error cascade.

### Step 7: Cross-Framework Logit Comparison

The definitive test. The same input sequence was fed through both the PyTorch and MLX models, and the output logits were compared element-wise:

```python
# PyTorch forward pass (run in Karpathy's repo to avoid import conflicts)
pt_logits = pt_model(torch.tensor([ids], dtype=torch.long))
np.save('/tmp/pt_logits.npy', pt_logits[0, -1, :].numpy())

# MLX forward pass
mlx_logits = mlx_model(mx.array([ids], dtype=mx.uint32))
mlx_last = np.array(mlx_logits[0, -1, :])

# Comparison
pt_last = np.load('/tmp/pt_logits.npy')
diff = np.abs(pt_last - mlx_last)
```

Input: `"\nROMEO:\nO, she doth teach"`

**Before fix:**
```
Logits max diff: 8.837387
Logits mean diff: 2.619270
Correlation: 0.65358459

PyTorch top 5: [(' ', 8.77), ('e', 6.22), (',', 4.10), ('i', 3.71), ("'", 3.16)]
MLX top 5:     [('e', 8.38), ('a', 7.36), ('o', 6.75), ('i', 6.46), ('u', 4.65)]
```

A max logit difference of 8.8 and correlation of only 0.65 with identical weights is catastrophic. The models aren't just slightly different — they're computing fundamentally different attention patterns. PyTorch confidently predicts a space after "teach"; MLX predicts a vowel. The MLX predictions look like a model that can see local character patterns but has no coherent understanding of word boundaries or syntax — exactly what you'd expect from broken causal masking.

### Step 8: Identifying the Bug

With the forward pass confirmed as the source, the search narrowed to components that differ between PyTorch and MLX implementations. The main candidates:

1. **Linear layer weight convention** — verified identical (`[out, in]` in both frameworks)
2. **LayerNorm implementation** — custom implementation, but operates element-wise and wouldn't produce this magnitude of error
3. **GELU activation** — standard, no room for divergence
4. **Attention mask construction and application** — the causal mask

Comparing the attention implementations side by side:

**PyTorch (Karpathy):**
```python
att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
```
`masked_fill` sets positions to `-inf` **where the condition is True** (i.e., where `bias == 0`, which is the upper triangle / future positions).

**MLX:**
```python
att = mx.where(mask[:,:,:T,:T] == 0, att, float('-1e9'))
```
`mx.where(condition, x, y)` returns `x` where condition is True, `y` where False.

So this reads: where `mask == 0` (future positions), keep `att`; where `mask == 1` (valid positions), replace with `-1e9`.

**This was exactly backwards.** The model was attending to future tokens and masking out the valid past context.

### Step 9: Fix and Verification

The fix was a single argument swap:

```python
# Before (broken): keeps att where mask==0, masks where mask==1
att = mx.where(mask[:,:,:T,:T] == 0, att, float('-1e9'))

# After (correct): masks where mask==0, keeps att where mask==1
att = mx.where(mask[:,:,:T,:T] == 0, float('-1e9'), att)
```

**After fix:**
```
Logits max diff: 0.000005
Logits mean diff: 0.000002
Correlation: 1.00000000

PyTorch top 5: [(' ', 8.7713), ('e', 6.2195), (',', 4.0993), ('i', 3.7126), ("'", 3.1613)]
MLX top 5:     [(' ', 8.7713), ('e', 6.2194), (',', 4.0993), ('i', 3.7126), ("'", 3.1613)]
```

The residual difference of 0.000005 is attributable to floating-point precision differences between frameworks. The models now produce identical predictions.

Generated text after fix:
```
ANGELO:
And you have fought with saying tears, how my bawd,
The flower of our lovely have but either
As she, is a man that for once and say
The thought that only lies for the sea,
To same the kingdom of King of York...
```

Recognizable Shakespeare from a model at only iteration 1750/5000.

## Why this bug was tricky to identify

1. **The model still trained.** With the mask inverted, the model could still see future tokens during training — it was essentially doing bidirectional attention. It learned *something*, just not causal language modeling. Training loss decreased, creating the illusion of progress.

2. **Short sequences masked the problem.** For a 1-token input, the causal mask is `[[1]]` — there are no future positions to attend to, so the mask is irrelevant. For short sequences, the model's embedding and positional encoding carry enough signal to produce plausible next-token predictions even with broken masking. The error only compounds as the generated sequence grows.

3. **Multiple confounding issues.** The `mx.where` NaN bug with `-inf`, the `mx.random.categorical` repetition issue, and dropout-during-inference were all real bugs that produced similar symptoms (repetitive/degenerate output). Each fix improved behavior slightly, making it appear that progress was being made toward the real solution.

4. **`mx.where` argument order is unintuitive.** PyTorch's `masked_fill(condition, value)` fills matching positions with the value. `mx.where(condition, x, y)` returns `x` where True, `y` where False — a ternary operator. The semantic gap between "fill where condition" and "choose between x and y" makes this class of bug easy to introduce and hard to spot in review.

## Takeaway Notes

- **Cross-framework logit comparison as a debugging tool**:  Weight identity + logit divergence immediately localizes the bug to the forward pass, and the magnitude/pattern of divergence provides clues about which component is broken.

- **Testing with a known-good checkpoint**: If inference is broken, no amount of hyperparameter tuning will fix training.

- **Test at realistic sequence lengths**: Short-sequence tests can pass even with fundamental architectural bugs. The causal mask is trivial for 1-5 token sequences.

- **API differences in semantically similar functions**: `masked_fill`, `mx.where`, `np.where`, and `torch.where` all handle conditionals but with different argument conventions. When porting between frameworks, verify these at the call site, not just by referencing documentation.
