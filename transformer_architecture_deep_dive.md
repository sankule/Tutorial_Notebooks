# Transformers: A Complete End-to-End Guide
### For the ML Scientist Moving into GenAI

---

## Preface: Why This Guide Exists

You already know encoder-decoder networks. You've likely worked with RNNs, LSTMs, or GRUs. The transformer was designed to replace all of that — not by being incrementally better, but by throwing out the sequential processing bottleneck entirely and replacing it with something fundamentally different: **attention over the entire sequence, all at once**.

This guide walks you from "why transformers?" to "here's every matrix operation happening inside one" — sequentially, with no loose ends.

---

## Step 1 — The Problem: Why RNNs Weren't Enough

Before transformers, sequence-to-sequence tasks (translation, summarization, Q&A) were dominated by RNNs and LSTMs with attention mechanisms. They had three critical problems:

### Problem 1: Sequential Processing = No Parallelism
RNNs process tokens one at a time: `h_t = f(h_{t-1}, x_t)`. To process token #100, you must wait for tokens 1–99. This kills GPU utilization because GPUs thrive on parallel computation.

### Problem 2: The Vanishing Gradient / Long-Range Dependency Problem
Even with LSTMs, encoding a 500-word document into a single context vector causes information from early tokens to be "washed out" by the time you process later ones. The model struggles to connect *"The trophy didn't fit in the suitcase because **it** was too big"* — where "it" refers back to "trophy" many tokens earlier.

### Problem 3: Information Bottleneck
The entire source sequence is compressed into one fixed-size hidden state vector `h`. This is an architectural chokepoint.

**The transformer solution:** Let every token look at every other token simultaneously, with learned, dynamic weights — this is called **self-attention**.

---

## Step 2 — The 30,000-Foot View: What a Transformer Is

The original transformer (Vaswani et al., 2017 — "Attention Is All You Need") is an encoder-decoder architecture, but unlike RNN-based seq2seq, it replaces recurrence with attention entirely.

```
INPUT SEQUENCE                         OUTPUT SEQUENCE
    │                                       │
    ▼                                       ▼
┌─────────────────────┐         ┌─────────────────────────┐
│                     │         │                         │
│   ENCODER STACK     │────────▶│    DECODER STACK        │
│  (N=6 identical     │  Cross  │  (N=6 identical layers) │
│   layers)           │ Attn    │                         │
│                     │         │                         │
└─────────────────────┘         └─────────────────────────┘
```

Each encoder layer has two sub-layers:
1. **Multi-Head Self-Attention**
2. **Position-wise Feed-Forward Network**

Each decoder layer has three sub-layers:
1. **Masked Multi-Head Self-Attention**
2. **Multi-Head Cross-Attention** (attends to encoder output)
3. **Position-wise Feed-Forward Network**

Both use **Residual Connections** and **Layer Normalization** around each sub-layer.

Let's now build this up from scratch.

---

## Step 3 — Input Representation: Turning Words into Vectors

### Step 3a: Tokenization

Before a model sees text, it's converted into tokens — subword units. The sentence:

> "The cat sat on the mat"

...might be tokenized as: `["The", "cat", "sat", "on", "the", "mat"]` → token IDs `[464, 3797, 3332, 319, 262, 2603]`

Modern models use **Byte-Pair Encoding (BPE)** or **WordPiece**, so rare words get split: "unbelievable" → `["un", "believ", "able"]`.

### Step 3b: Token Embeddings

Each token ID maps to a learned embedding vector of dimension `d_model`. In the original paper, `d_model = 512`.

```
Token "cat" → ID 3797 → Embedding vector ∈ ℝ^512
```

The **Embedding Matrix** `E ∈ ℝ^(vocab_size × d_model)` is learned during training. You're essentially doing a lookup: `e_i = E[token_id_i]`.

For a sequence of length `n`, you get a matrix `X ∈ ℝ^(n × d_model)`.

**Example:** "The cat sat" → `X ∈ ℝ^(3 × 512)` — three row vectors, one per token.

---

## Step 4 — Positional Encoding: Injecting Order into a Position-Blind Model

Here's the critical insight: **attention has no notion of order**. If you shuffle the input tokens, self-attention produces the same weighted combinations — just in a different order. This is a disaster for language, where "dog bites man" ≠ "man bites dog".

The fix: **add a positional signal** to each embedding before it enters the transformer.

### The Formula

The paper uses fixed sinusoidal functions:

```
PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
```

**Breaking it down term by term:**
- `pos` — the position of the token in the sequence (0, 1, 2, ...)
- `i` — the index of the embedding dimension (0, 1, 2, ..., d_model/2 - 1)
- `d_model` — the embedding dimension (512)
- Even dimensions (2i) use **sine**, odd dimensions (2i+1) use **cosine**

**Why sinusoids?** They generate a unique pattern for each position, they're bounded between -1 and 1, and — crucially — the model can learn to attend to *relative* positions because `PE(pos + k)` can be expressed as a linear function of `PE(pos)`.

The final input to the model is:

```
X_input = TokenEmbedding(x) + PositionalEncoding(pos)
```

**Intuition:** Think of each dimension of the positional encoding as a different "frequency clock." Low dimensions oscillate slowly (capturing coarse position), high dimensions oscillate fast (capturing fine position). Together, they create a unique "fingerprint" for each position.

---

## Step 5 — The Heart of the Transformer: Self-Attention

This is the most important concept. Understand this deeply.

### The Core Idea

**Self-attention** lets each token in a sequence look at all other tokens and decide how much to "attend" to each one when building its representation.

Consider: *"The animal didn't cross the street because it was too tired."*

When processing the word **"it"**, the model needs to figure out whether "it" refers to "animal" or "street." Self-attention lets "it" query all other tokens and assign high weight to "animal" (because "tired" is more semantically consistent with animals than streets).

### The Mechanism: Queries, Keys, and Values

Self-attention uses three learnable linear projections of each token embedding:

- **Query (Q):** "What am I looking for?"
- **Key (K):** "What do I contain/represent?"
- **Value (V):** "What information will I pass along if selected?"

**Analogy:** Think of a library system.
- Your **query** is your search request: "I want books about machine learning."
- Each book has a **key** (its catalog description): "Deep learning, neural networks..."
- The **value** is the actual book content.
- Attention computes how well your query matches each key, then returns a weighted mix of the values.

### The Math

Given input matrix `X ∈ ℝ^(n × d_model)`, we project it into Q, K, V:

```
Q = X · W_Q        W_Q ∈ ℝ^(d_model × d_k)
K = X · W_K        W_K ∈ ℝ^(d_model × d_k)
V = X · W_V        W_V ∈ ℝ^(d_model × d_v)
```

Where `d_k = d_v = d_model / h` (h = number of heads; in the paper, h=8, d_k=64).

The attention output is:

```
Attention(Q, K, V) = softmax( Q · K^T / √d_k ) · V
```

Let's unpack every piece of this:

#### Part 1: `Q · K^T` — The Similarity Score Matrix

- `Q ∈ ℝ^(n × d_k)`, `K^T ∈ ℝ^(d_k × n)` → result is `S ∈ ℝ^(n × n)`
- Entry `S[i][j]` = dot product between query of token `i` and key of token `j`
- A high dot product means token `i` finds token `j` very relevant
- This gives you an **n×n attention score matrix** — every token scored against every other token

**Example:** For "The cat sat", S is 3×3:
```
         The    cat    sat
The  [ [2.1,  0.3,  0.1],
cat    [0.8,  3.5,  1.2],
sat ]  [0.2,  1.8,  2.9] ]
```
"cat" attends most to itself (3.5) and somewhat to "sat" (1.2).

#### Part 2: `/ √d_k` — The Scaling Factor

**Why divide by √d_k?**

Dot products grow in magnitude as `d_k` increases (because you're summing more terms). Large dot products push softmax into regions where gradients become vanishingly small (the "saturated" regime). Dividing by √d_k keeps the variance of the dot products at ~1, preventing this.

**Concrete example:** If d_k = 64, we divide by √64 = 8. If d_k = 512, we divide by √512 ≈ 22.6.

#### Part 3: `softmax(...)` — Converting Scores to Weights

Softmax is applied **row-wise** (over the key dimension), turning raw scores into a probability distribution:

```
softmax(s_i) = exp(s_ij) / Σ_j exp(s_ij)
```

After softmax, each row sums to 1. These are the **attention weights** `A ∈ ℝ^(n × n)`.

```
         The    cat    sat
The  [ [0.85,  0.10,  0.05],   ← "The" mostly attends to itself
cat    [0.15,  0.70,  0.15],   ← "cat" strongly attends to itself
sat ]  [0.05,  0.35,  0.60] ]  ← "sat" attends to itself and "cat"
```

#### Part 4: `· V` — Weighted Sum of Values

`A ∈ ℝ^(n × n)` × `V ∈ ℝ^(n × d_v)` → Output `∈ ℝ^(n × d_v)`

Each output token representation is a **weighted average of all Value vectors**, where the weights come from the attention matrix.

**What this means:** The new representation of "sat" is 60% of "sat"'s value vector + 35% of "cat"'s value vector + 5% of "the"'s value vector. The model has dynamically decided which tokens contribute to each token's updated representation.

---

## Step 6 — Multi-Head Attention: Running Attention in Parallel

Single-head attention learns one way of relating tokens to each other. But relationships are multi-faceted — syntactic dependencies, semantic roles, coreference — all simultaneously.

**Multi-head attention** runs `h` attention heads in parallel, each with its own `W_Q^i, W_K^i, W_V^i` projections, then concatenates the results:

```
head_i = Attention(X·W_Q^i, X·W_K^i, X·W_V^i)

MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_O
```

**Breaking down the terms:**
- `h` = number of heads (8 in the original paper)
- Each head operates in a lower-dimensional space: `d_k = d_model / h = 512/8 = 64`
- `W_O ∈ ℝ^(h·d_v × d_model)` — final projection back to d_model size

**Example — what different heads might learn:**

In the sentence *"The cat sat on the mat because it was comfortable"*:
- **Head 1** might learn syntactic dependencies: "sat" → "cat" (subject-verb)
- **Head 2** might learn coreference: "it" → "mat"
- **Head 3** might learn positional proximity: each word attends to its neighbors
- **Head 4** might learn semantic roles: "on" → "mat" (location preposition)

Each head captures a different relational pattern. The concatenation merges all these perspectives into one rich representation.

**Computational note:** Because each head operates in d_k=64 dimensions (not 512), the total computation is the same as one full-dimensional attention head — no extra cost, more expressiveness.

---

## Step 7 — Position-wise Feed-Forward Network (FFN)

After multi-head attention, each token's representation passes through a small, identical, **fully-connected feed-forward network** applied independently to each position:

```
FFN(x) = max(0, x · W_1 + b_1) · W_2 + b_2
```

**Breaking down the terms:**
- `x ∈ ℝ^d_model` — the output of attention for one token (512-dim)
- `W_1 ∈ ℝ^(d_model × d_ff)` — expands to a larger inner dimension
- `d_ff = 2048` in the paper (4× expansion)
- `max(0, ...)` — ReLU activation
- `W_2 ∈ ℝ^(d_ff × d_model)` — projects back to d_model

**What's the role of the FFN?**

Attention decides **which** tokens to integrate information from. The FFN then performs **nonlinear transformation** on the integrated representation — think of it as each token's "thinking step" after gathering context. Research has shown FFN layers act as key-value memory stores, encoding factual knowledge.

**Key fact:** W_1 and W_2 are **shared across positions** but each position is processed **independently**. This is why it's called "position-wise."

---

## Step 8 — Residual Connections and Layer Normalization

Each sub-layer in the transformer is wrapped with:
1. A **residual (skip) connection**
2. **Layer normalization**

The formula for each sub-layer is:

```
Output = LayerNorm(x + Sublayer(x))
```

### Residual Connections

`x + Sublayer(x)` — the input `x` is added back to the sub-layer's output.

**Why?** Residual connections (from ResNets) solve the vanishing gradient problem in deep networks. Gradients can flow directly backward through the identity connection without being filtered through every transformation. This is what makes it possible to stack 6, 12, 96, or even 1000+ layers.

### Layer Normalization

```
LayerNorm(x) = γ · (x - μ) / (σ + ε) + β
```

**Breaking down:**
- `μ` — mean of the d_model values for this token
- `σ` — standard deviation of the d_model values for this token
- `ε` — small constant for numerical stability (e.g., 1e-6)
- `γ, β` — learned scale and shift parameters (per dimension)

**Unlike Batch Norm**, LayerNorm normalizes across the **feature dimension** (d_model) for each token independently — not across the batch. This makes it robust to variable sequence lengths and works correctly at inference with batch size 1.

**Intuition:** LayerNorm keeps activations in a stable numerical range throughout the network, preventing explosions or vanishing — critical in 6+ layer deep architectures.

---

## Step 9 — The Full Encoder: Putting It Together

One encoder layer performs these operations in sequence:

```
# Input: X ∈ ℝ^(n × d_model)

# Sub-layer 1: Multi-Head Self-Attention
attn_out = MultiHeadAttention(X, X, X)      # Q=K=V=X (self-attention)
X = LayerNorm(X + attn_out)                 # Residual + Norm

# Sub-layer 2: Feed-Forward
ffn_out = FFN(X)
X = LayerNorm(X + ffn_out)                  # Residual + Norm

# Output: X ∈ ℝ^(n × d_model)  — same shape as input
```

The original paper stacks **N=6 identical encoder layers**. (GPT-3 uses 96 layers; BERT-large uses 24.)

The output of the final encoder layer is a matrix `Z ∈ ℝ^(n × d_model)` — a **contextualized representation** of every input token. Every token's representation has been enriched by attending to all other tokens, through 6 rounds of attention + FFN.

**Example trace for "The cat sat":**
- After layer 1: tokens begin incorporating local context
- After layer 3: syntactic structure fully captured
- After layer 6: rich semantic + syntactic representation — "cat" now encodes that it's a subject, it sat, it's a living thing

---

## Step 10 — The Decoder: Generation with Masked Attention and Cross-Attention

The decoder generates the output sequence **one token at a time**, auto-regressively. But during **training**, all target tokens are available simultaneously — so we process the whole target sequence at once, using a mask to prevent cheating.

Each decoder layer has **three sub-layers:**

### Sub-layer 1: Masked Multi-Head Self-Attention

The decoder attends to the tokens it has **already generated**. To prevent any position from attending to future positions (which don't exist at inference time), we apply a **causal mask**:

```
# The attention score matrix S ∈ ℝ^(n × n) gets a mask applied:
# For position i, set S[i][j] = -∞ for all j > i

Masked_S[i][j] = S[i][j]   if j ≤ i
                  -∞         if j > i
```

After softmax, `-∞` becomes `0` — those positions get zero attention weight.

**Example:** Generating "Le chat" (French for "The cat"):
- When generating "chat": can attend to "Le" and "chat" but not any future tokens
- When generating "Le": can only attend to "Le" (first position)

### Sub-layer 2: Multi-Head Cross-Attention (Encoder-Decoder Attention)

This is where the decoder **reads the encoder's output**. This is the bridge between encoder and decoder.

```
cross_attn = MultiHeadAttention(Q=decoder_state, K=encoder_output, V=encoder_output)
```

**Breaking this down:**
- **Queries (Q)** come from the **decoder** (what the decoder is currently building)
- **Keys (K) and Values (V)** come from the **encoder output Z** (the source representation)

For every decoder token position, this computes: "Given what I'm generating right now (Q), which parts of the source sentence (K/V) should I focus on?"

**Example — Machine Translation "The cat sat" → "Le chat s'est assis":**
- When the decoder generates "chat" (cat), cross-attention focuses on "cat" in the encoder output
- When generating "assis" (sat), cross-attention focuses on "sat" in the encoder output

This is the alignment mechanism — directly analogous to (but more powerful than) the Bahdanau attention you'd know from RNN seq2seq.

### Sub-layer 3: Feed-Forward Network

Identical to the encoder FFN — applied position-wise, same `d_ff=2048` inner dimension.

### Full Decoder Layer:

```
# Input: decoder sequence Y ∈ ℝ^(m × d_model), encoder output Z ∈ ℝ^(n × d_model)

# Sub-layer 1: Masked Self-Attention (attend only to past positions)
masked_attn = MaskedMultiHeadAttention(Y, Y, Y)
Y = LayerNorm(Y + masked_attn)

# Sub-layer 2: Cross-Attention (attend to encoder output)
cross_attn = MultiHeadAttention(Q=Y, K=Z, V=Z)
Y = LayerNorm(Y + cross_attn)

# Sub-layer 3: Feed-Forward
ffn_out = FFN(Y)
Y = LayerNorm(Y + ffn_out)
```

N=6 decoder layers are stacked identically.

---

## Step 11 — The Output Head: From Vectors to Probabilities

After the final decoder layer, we have `Y ∈ ℝ^(m × d_model)` — one vector per output token position.

To predict the next token at each position:

```
Logits = Y · W_vocab^T        W_vocab ∈ ℝ^(vocab_size × d_model)
Probs  = softmax(Logits)      Probs ∈ ℝ^(m × vocab_size)
```

At position `t`, `Probs[t]` is a probability distribution over the entire vocabulary — the model's predicted probability for each word being the correct next token.

**Interesting detail:** The output projection matrix `W_vocab` is often **tied** to the input embedding matrix `E`. Since both map between token space and d_model space, tying them reduces parameters and often improves performance.

---

## Step 12 — Training the Transformer

### Loss: Cross-Entropy

For each position `t`, the true next token is a one-hot vector. The loss is:

```
L = -Σ_t log(P(y_t | y_<t, X))
```

This is the average negative log-probability assigned to the correct tokens — **teacher forcing**: during training, the true target tokens are always fed as input (not the model's predictions), making training stable.

### Label Smoothing

Instead of one-hot targets `[0, 0, 1, 0, 0]`, use soft targets like `[0.025, 0.025, 0.9, 0.025, 0.025]` with ε=0.1. This prevents the model from being overconfident and improves calibration.

### Optimizer: Adam with Warmup

The original paper uses a custom learning rate schedule:

```
lr = d_model^(-0.5) · min(step^(-0.5), step · warmup_steps^(-1.5))
```

- **Warmup phase** (first 4000 steps): LR increases linearly → allows the model to explore before committing to a direction
- **Decay phase:** LR decreases proportionally to 1/√step

---

## Step 13 — Inference: How the Transformer Generates Text

At inference, there's no teacher forcing. The decoder generates **auto-regressively**:

```
Step 1: Feed [<START>] → predict first token → "Le" (highest probability)
Step 2: Feed [<START>, "Le"] → predict second token → "chat"
Step 3: Feed [<START>, "Le", "chat"] → predict third token → "s'est"
...
Step N: Generate <END> token → stop
```

Each step, the full attention computation re-runs. Modern systems use **KV-Cache** optimization: since Keys and Values from previous tokens don't change, they're cached, making each decoding step O(1) in attention computation (for already-generated tokens).

### Decoding Strategies

- **Greedy:** Pick argmax at each step. Fast, but can get stuck in repetitive loops.
- **Beam Search:** Keep top-k candidate sequences at each step. Better quality, still deterministic.
- **Sampling:** Sample from the probability distribution. `temperature > 1` = more random; `temperature < 1` = more deterministic.
- **Top-p / Nucleus Sampling:** Sample from the smallest subset of tokens whose cumulative probability exceeds p (e.g., 0.9). Balances creativity and coherence.

---

## Step 14 — The Variants That Came After

Understanding the original architecture unlocks understanding all modern variants:

| Model | Architecture | Key Change |
|-------|-------------|------------|
| **BERT** | Encoder only | Masked Language Modeling, bidirectional |
| **GPT series** | Decoder only | Causal LM, no encoder, no cross-attention |
| **T5** | Encoder-Decoder | Frames everything as text-to-text |
| **RoBERTa** | Encoder only | Better BERT training recipe |
| **Vision Transformer (ViT)** | Encoder only | Patches of images as tokens |
| **Llama / Mistral** | Decoder only | RoPE positional encoding, GQA, SwiGLU |

**Decoder-only models (GPT, Llama)** are just the transformer decoder, but without cross-attention (no encoder to attend to). They use masked self-attention throughout. This is the dominant architecture for modern LLMs.

---

## Step 15 — Everything in One Pass: End-to-End Example

**Task:** Translate "The cat sat" → "Le chat s'est assis"

### Encoder Path:

```
1. Tokenize:  ["The", "cat", "sat"] → IDs [464, 3797, 3332]

2. Embed:     X ∈ ℝ^(3 × 512)
              (lookup embedding matrix for each token)

3. Add PE:    X += PE(0), PE(1), PE(2)

4. Layer 1:
   a. Q=K=V=X
   b. Compute 8 attention heads in parallel
      Each head: score = softmax(QK^T / 8) · V   [d_k=64]
   c. Concat heads → project with W_O → ℝ^(3×512)
   d. Residual + LayerNorm
   e. FFN: expand to 2048 → ReLU → contract to 512
   f. Residual + LayerNorm

5. Repeat for Layers 2–6

6. Output: Z ∈ ℝ^(3 × 512)  ← rich contextualized source representation
```

### Decoder Path (Generating "Le"):

```
1. Input: [<START>] → embed + PE → Y ∈ ℝ^(1 × 512)

2. Layer 1:
   a. Masked self-attn on Y (trivial at step 1, only one token)
   b. Cross-attn: Q=Y, K=Z, V=Z → attends to "The", "cat", "sat"
      → focuses most on "The" (aligned to "Le")
   c. FFN → Residual + Norm

3. Repeat for Layers 2–6

4. Output head: Y · W_vocab^T → logits ∈ ℝ^vocab_size
               softmax → P("Le") = 0.74, P("Un") = 0.12, ...

5. Argmax → "Le" is generated
```

### Repeat for "chat", "s'est", "assis", then `<END>`.

---

## Summary: The Transformer's Key Insights

| Concept | Purpose | What It Replaces |
|---------|---------|-----------------|
| Self-Attention | Dynamic, context-aware token relationships | Recurrent hidden states |
| Positional Encoding | Inject sequence order | Sequential processing |
| Multi-Head Attention | Learn multiple relationship types simultaneously | Single attention perspective |
| FFN | Per-token nonlinear transformation | — |
| Residual + LayerNorm | Training stability in deep networks | — |
| Masking (decoder) | Prevent future token leakage | RNN's inherent causality |
| Cross-Attention | Decoder reads encoder's representation | Context vector bottleneck |

---

## What You Now Know

✅ Why transformers were invented (RNN limitations)  
✅ How tokenization and embeddings work  
✅ Why and how positional encoding is added  
✅ The full Q/K/V self-attention mechanism with math  
✅ Why we scale by √d_k and what it prevents  
✅ Multi-head attention and what different heads capture  
✅ The position-wise FFN and its role  
✅ Residual connections and LayerNorm — why both are needed  
✅ The full encoder stack  
✅ Masked self-attention in the decoder (causal masking)  
✅ Cross-attention: the encoder-decoder bridge  
✅ The output projection and probability generation  
✅ Training (teacher forcing, label smoothing, warmup)  
✅ Inference (autoregressive generation, KV cache, decoding strategies)  
✅ How all modern LLMs (GPT, BERT, Llama) relate to this foundation  

---

*"Attention Is All You Need" — Vaswani et al., 2017. The paper that started it all. Now you know exactly what they meant.*
