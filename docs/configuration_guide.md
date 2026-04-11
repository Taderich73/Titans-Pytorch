# Model Architecture Reference

---

## Model Architecture

**--dim 768**

The hidden dimension — the width of the vector that represents each token as it flows through the model. Every token gets converted from a word/subword into a 768-number vector. This vector is what the model actually manipulates. Bigger dim = richer representations but more compute and memory. 768 is a common "small-medium" size (GPT-2 base used 768, for reference).

**--num-layers 16**

How many transformer blocks are stacked on top of each other. Each layer takes the token representations, lets them attend to each other, then passes them through a feed-forward network. More layers = more opportunities for the model to refine its understanding, build abstractions, and compose features. Think of it as depth of reasoning — layer 1 might capture syntax, layer 8 might capture semantic relationships, layer 16 might capture higher-order patterns. 16 layers is moderate.

**--num-heads 16**

The number of attention heads per layer. Multi-head attention splits the 768-dim vector into 16 independent 48-dim "perspectives" (768 / 16 = 48). Each head learns to attend to different things — one might focus on nearby tokens, another on subject-verb agreement, another on coreference. After attending independently, the heads' outputs get concatenated back to 768 dims. More heads = more diverse attention patterns. Having heads = dim/48 is a common ratio.

---

## Attention Configuration

**--window-size 512**

The sliding window size for MAG and MAL variants. Each token can only attend to the previous 512 tokens rather than the entire sequence. This makes attention cost linear in sequence length (O(N × W) instead of O(N²)). Long-range context beyond the window is handled by the neural memory module, not attention. Larger windows give richer local context but cost more compute. Since the neural memory handles long-range patterns, you can often use smaller windows than you'd expect.

**--chunk-size 512**

For the MAC variant, the sequence gets split into chunks of this size. Full causal attention operates within each chunk, and the neural memory persists across chunks. For MAG/MAL, chunk_size controls how the sequence is segmented for processing — the sliding window attention operates within each chunk while memory state carries forward.

**--adaptive-window**

Enables per-layer learned adaptive window sizing (MAG and MAL only). Instead of a fixed window for all layers, each layer gets a lightweight predictor that learns its own effective window size from the input. A soft sigmoid falloff replaces the hard window cutoff, making the boundary differentiable. An efficiency regularization term encourages smaller windows where local context suffices.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--adaptive-window-min` | 64 | Minimum window size floor |
| `--adaptive-window-max` | None | Maximum window size ceiling (defaults to `--window-size`) |
| `--adaptive-window-temperature` | 10.0 | Sigmoid sharpness — higher = more binary mask |
| `--adaptive-window-lambda` | 0.01 | Regularization weight — higher = more pressure toward smaller windows |

The predictor input differs by variant: MAG uses `norm1(h)` (pre-attention hidden state), MAL uses `norm2(h_mid)` (memory-enriched hidden state, since memory runs before attention in MAL).

---

## Memory Configuration

**--num-persistent-tokens 16**

Learned prefix tokens prepended to every attention computation. These encode task-level knowledge that doesn't change with input. They bypass the sliding window constraint — all query positions can attend to persistent tokens regardless of distance.

**--num-memory-layers 2**

Depth of the neural long-term memory MLP. L=1 gives a linear memory (single weight matrix), L>=2 gives a deep memory with hidden layers and SiLU activations. Deeper memory can capture more complex associations but is more expensive.

**--memory-objective l2**

The attentional bias for memory updates. `l2` (default) uses standard squared error. `huber` (Yaad) uses a Huber loss with a learned per-position delta gate, making it robust to outlier tokens.

---

## Training Configuration

**--batch-size 2**

How many independent training examples the GPU processes in a single forward+backward pass. 2 is tiny — this means only 2 sequences at a time fit in GPU memory. This is where gradient accumulation comes in.

**--gradient-accumulation-steps 24**

The model does 24 forward+backward passes (each with batch-size 2), accumulating the gradients without updating the weights. After 24 steps, it averages them and does one weight update. The **effective batch size** is 2 × 24 = **48 sequences per update**. This is a standard trick to simulate a large batch on limited VRAM — you get the statistical stability of a 48-sequence batch while only ever holding 2 in memory.

Why this matters: small effective batches make training noisy and unstable. Large effective batches give smoother, more reliable gradient estimates. 48 is a reasonable effective batch for a model this size.

**--seq-len 3072**

The maximum sequence length — how many tokens the model sees in a single training example. 3072 tokens is roughly 2,000-4,000 words depending on tokenizer. This defines the model's context window during training. Longer sequences let the model learn longer-range dependencies but attention cost scales quadratically with sequence length (for MAC) or linearly with the window (for MAG/MAL), so memory and compute go up with length regardless.

---

## How They Interact

The total compute per training step scales roughly with: `batch_size × seq_len × num_layers × dim²`. Memory pressure comes mainly from `batch_size × seq_len × dim` for activations, plus `num_layers × dim²` for parameters.

For MAC: `chunk_size` controls the attention cost tradeoff — smaller chunks = cheaper attention but more reliance on the neural memory to bridge across chunks.

For MAG/MAL: `window_size` controls the attention cost. With `--adaptive-window`, the effective window size per layer is learned rather than fixed, so compute varies by layer and content. The regularization term (`--adaptive-window-lambda`) provides a tunable knob trading off quality vs. efficiency — sweep it to find the Pareto curve for your task.

## Three Tiers of Context Access

| Tier | Mechanism | Range | Cost |
|------|-----------|-------|------|
| Persistent memory | Learned prefix tokens | Always accessible | Fixed (num_persistent_tokens) |
| Sliding window attention | Local attention | Last W tokens (fixed or adaptive) | O(W) per token |
| Neural long-term memory | Compressed state | Entire history | Fixed per chunk |
