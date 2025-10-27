# Transformer Core
# Token + position embeddings

# Causal self-attention

# Feedforward network

# Repeated blocks

# Final LM head to predict the next token

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .model_config import ModelConfig

class CasualSelfAttention(nn.Module):
    """
    Causal Self-Attention module.

    This is the core building block of a Transformer, allowing it to weigh the
    importance of different words in the input sequence when processing a given word.
    A standard GPT-style causal self-attention head group.

    - We project the hidden states (d_model) into queries, keys, and values.
    - We split those into multiple heads.
    - Each token can only attend to tokens at positions <= itself (causal mask).
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        assert config.d_model % config.n_heads == 0, "d_model must be divisible by n_heads"
        # The dimension of each attention head.
        self.head_dim = config.d_model // config.n_heads
        self.n_heads = config.n_heads

        # A single linear layer to create the Query, Key, and Value projections for all heads at once.
        # This is more efficient than creating three separate linear layers.
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias = False)
        # - Query (Q): What I am looking for. (e.g., a verb for the current subject)
        # - Key (K): What information I have. (e.g., this word is a noun)
        # - Value (V): What I will pass on if you pay attention to me. (e.g., the actual meaning/embedding of the word)

        # A final linear layer to combine the outputs of all attention heads back into a single vector.
        # This is often called the "output projection".
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x: torch.Tensor):
        """
        x: (batch, seq_len, d_model)
        return: (batch, seq_len, d_model)
        """

        # B = Batch size (how many sequences we process at once)
        # T = Sequence length (how many tokens are in each sequence)
        # C = d_model (the size of the vector representing each token)
        B, T, C = x.shape

        # 1. Create Query, Key, and Value for all heads
        # Pass the input `x` through the linear layer to get a combined Q, K, V tensor.
        qkv = self.qkv(x)  # Shape: (B, T, 3 * C)
        # Split the combined tensor into three separate tensors for Q, K, and V.
        q, k, v = torch.split(qkv, C, dim=2)  # Each has shape: (B, T, C)

        # 2. Reshape for Multi-Head Attention
        # We split the embedding dimension `C` into `n_heads` smaller `head_dim` chunks.
        # This allows the model to focus on different aspects of the sequence simultaneously.
        # For example, one head might track subject-verb agreement, while another tracks semantic relationships.
        # The initial shape is (B, T, C). We want (B, n_heads, T, head_dim).
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # 3. Scaled Dot-Product Attention
        # This is where the magic happens. We calculate how much each token should "attend" to every other token.
        # We do this by taking the dot product of the Query of one token with the Key of another.
        # (B, n_heads, T, head_dim) @ (B, n_heads, head_dim, T) -> (B, n_heads, T, T)
        att_scores = torch.matmul(q, k.transpose(-2, -1))
        # We scale the scores by the square root of the head dimension. This prevents the dot products
        # from becoming too large, which would make the softmax output too "spiky" and hard to train.
        att_scores = att_scores / math.sqrt(self.head_dim)

        # 4. Apply Causal Mask
        # For a language model that predicts the next word, a token should NOT see future tokens.
        # We create a mask to hide all positions "to the right" (future tokens).
        # `torch.triu` creates an upper-triangular matrix. We set these positions to -infinity.
        mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool),
            diagonal=1  # The diagonal is 0, so a token can attend to itself.
        )
        # When we apply softmax, e^(-inf) becomes 0, so future tokens get zero attention.
        att_scores = att_scores.masked_fill(mask, float('-inf'))
        # `softmax` converts the raw scores into probabilities (attention weights) that sum to 1.
        att_weights = F.softmax(att_scores, dim=-1)  # Shape: (B, n_heads, T, T)

        # 5. Compute the weighted sum of Values
        # We multiply the attention weights by the Value vectors.
        # This gives us a new vector for each token, which is a weighted blend of all other token Values.
        # (B, n_heads, T, T) @ (B, n_heads, T, head_dim) -> (B, n_heads, T, head_dim)
        att_output = torch.matmul(att_weights, v)

        # 6. Combine Heads and Project Output
        # We need to put the heads back together into a single tensor.
        # .transpose(1, 2) -> (B, T, n_heads, head_dim)
        # .contiguous() ensures the tensor is stored in a contiguous block of memory.
        # .view(B, T, C) merges the `n_heads` and `head_dim` back into the original embedding dimension `C`.
        att_output = att_output.transpose(1, 2).contiguous().view(B, T, C)

        # Pass the combined output through a final linear layer.
        out = self.out_proj(att_output)  # Shape: (B, T, C)
        return out
    
class FeedForward(nn.Module):
    """
    Simple 2-layer MLP used inside each Transformer block.
    Typically: Linear(d_model -> d_ff) -> GELU -> Linear(d_ff -> d_model)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        # The "expand" layer: projects from d_model to a larger dimension d_ff.
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        # The "contract" layer: projects back from d_ff to d_model.
        self.fc2 = nn.Linear(config.d_ff, config.d_model)
        # A non-linear activation function. GELU is common in modern Transformers.
        # This non-linearity is crucial; without it, the two linear layers would just be one linear layer.
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor):
        # The forward pass: expand -> activate -> contract.
        # This is where the model does a lot of its "thinking" on a per-token basis.
        return self.fc2(self.activation(self.fc1(x)))
    
class TransformerBlock(nn.Module):
    """
    One Decoder block:
    x -> LayerNorm -> SelfAttention -> residual
      -> LayerNorm -> FeedForward -> residual
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        # Layer Normalization helps stabilize training by normalizing the inputs to a sub-layer.
        # This is a "pre-norm" architecture, where we normalize *before* the main operation.
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        # The causal self-attention sub-layer.
        self.attn = CasualSelfAttention(config)
        # The feed-forward sub-layer.
        self.ffn = FeedForward(config)
        # Dropout is a regularization technique. During training, it randomly sets some activations
        # to zero, which prevents the model from becoming too reliant on any single neuron.
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x: torch.Tensor):
        # First sub-layer: Self-Attention
        # The input `x` is passed through a "residual connection" or "skip connection".
        # We add the output of the attention layer back to the original input `x`.
        # This is crucial for training deep networks, as it allows gradients to flow more easily.
        attn_output = self.attn(self.ln1(x))
        x = x + self.dropout(attn_output)

        # Second sub-layer: Feed-Forward Network
        # We do the same thing again: normalize, pass through the FFN, apply dropout, and add to the input.
        ffn_output = self.ffn(self.ln2(x))
        x = x + self.dropout(ffn_output)
        return x
    
class Transformer(nn.Module):
    """
    Full decoder-only transformer LM:
    - token embedding
    - positional embedding
    - N transformer blocks
    - final layer norm
    - LM head to predict next-token logits
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # A dictionary-like layer that maps token IDs (integers) to dense vectors (embeddings).
        # This is the first step: turning words into numbers the model can understand.
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        # A layer to create embeddings for token positions. Since attention is position-agnostic,
        # we need to explicitly give the model information about the order of tokens.
        self.pos_embed = nn.Embedding(config.context_window, config.d_model)

        # A stack of N identical Transformer blocks. The output of one block is the input to the next.
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # A final layer normalization after all the blocks.
        self.ln_f = nn.LayerNorm(config.d_model)

        # The final linear layer that maps the model's output back to the vocabulary size.
        # This gives us a raw score (logit) for each possible next token.
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """
        idx: (batch, seq_len) integer token IDs
        targets: (batch, seq_len) integer token IDs we want to predict
                 If provided, we compute cross-entropy loss.

        returns:
            logits: (batch, seq_len, vocab_size)
            loss:   scalar or None
        """
        B, T = idx.shape  # Batch size, Sequence length
        if T > self.config.context_window:
            raise ValueError(f"Sequence length {T} exceeds model context window {self.config.context_window}")
        
        # 1. Get Token and Positional Embeddings
        token_embeddings = self.token_embed(idx)  # (B, T) -> (B, T, d_model)
        # Create position indices from 0 to T-1.
        position_indices = torch.arange(T, device=idx.device)  # (T)
        pos_embeddings = self.pos_embed(position_indices)  # (T) -> (T, d_model)
        # Add the two embeddings together. This infuses the token embeddings with positional information.
        x = token_embeddings + pos_embeddings  # (B, T, d_model)

        # 2. Pass through the stack of Transformer Blocks
        for block in self.blocks:
            x = block(x)
        
        # 3. Final Layer Norm and Language Model Head
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, d_model) -> (B, T, vocab_size)

        loss = None
        if targets is not None:
            # To calculate the loss, we want to predict the next token.
            # For an input `[t1, t2, t3]`, the model outputs predictions for `[t2, t3, t4]`.
            # So, we compare `logits` for `[t1, t2]` with `targets` `[t2, t3]`.
            # We ignore the last logit and the first target.
            logits_shifted = logits[:, :-1, :].contiguous()
            targets_shifted = targets[:, 1:].contiguous()
            loss = F.cross_entropy(
                logits_shifted.view(-1, logits_shifted.size(-1)),
                targets_shifted.view(-1)
            )
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int):
        """
        Generate new tokens autoregressively given a context.

        idx: (B, T) initial context token IDs
        max_new_tokens: number of tokens to generate

        returns:
            (B, T + max_new_tokens) generated token IDs
        """
        for _ in range(max_new_tokens):
            # If the current sequence is longer than our model's context window,
            # we must crop it to the most recent `context_window` tokens.
            # The model can only "see" this many tokens at a time.
            if idx.size(1) > self.config.context_window:
                idx_cond = idx[:, -self.config.context_window:]
            else:
                idx_cond = idx

            # Get the model's predictions (logits) for the next token.
            logits, _ = self.forward(idx_cond)
            # We only care about the predictions for the very last token in the sequence.
            next_token_logits = logits[:, -1, :]  # Shape: (B, vocab_size)

            # Convert logits to probabilities using softmax.
            probs = F.softmax(next_token_logits, dim=-1)
            # "Greedy decoding": simply pick the token with the highest probability.
            next_token = torch.argmax(probs, dim=-1, keepdim=True)  # Shape: (B, 1)

            # Append the newly predicted token to our sequence.
            idx = torch.cat([idx, next_token], dim=1)

        return idx
