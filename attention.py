from torch import nn
import torch

class CausalAttention(nn.Module):
    """Single-head causal attention: tokens cannot attend to future tokens.

    d_in: input embedding features per token.
    d_out: query, key, value, and output features per token.
    context_length: maximum sequence length.

    Input: (b, num_tokens, d_in). Output: (b, num_tokens, d_out).
    b is the batch size; num_tokens must be <= context_length.

    Example: input (1, 3, 4) means 1 sequence, 3 tokens, 4 features per token.
    With d_out=2 and context_length>=3, the output shape is (1, 3, 2).
    """
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        # A buffer moves with the model's device and is not a trainable parameter.
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        # x contains embeddings: (b, num_tokens, d_in).
        b, num_tokens, d_in = x.shape

        # Project embeddings into queries, keys, and values: (b, num_tokens, d_out).
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Compare queries and keys; rows are query tokens, columns are key tokens.
        # Scores: (b, num_tokens, num_tokens).
        attn_scores = queries @ keys.transpose(1, 2)

        # Mask future tokens; softmax turns their -inf scores into zero weights.
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)

        # Scale by sqrt(d_out), then normalize over keys for each query.
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        # During training, drop and rescale attention weights for regularization.
        attn_weights = self.dropout(attn_weights)

        # Weighted sums of values: (b, num_tokens, d_out).
        context_vec = attn_weights @ values
        return context_vec

class MultiHeadAttention(nn.Module):
    """Multi-head causal attention: tokens cannot attend to future tokens.

    d_in: input embedding features per token.
    d_out: total query, key, value, and output features per token across all heads.
    context_length: maximum sequence length.
    num_heads: number of parallel attention heads; must divide d_out evenly.
    head_dim: features per head, equal to d_out // num_heads.

    Input: (b, num_tokens, d_in). Output: (b, num_tokens, d_out).
    b is the batch size; num_tokens must be <= context_length.

    Example: input (1, 3, 4) means 1 sequence, 3 tokens, 4 features per token.
    With d_out=4, num_heads=2, and context_length>=3, each head uses 2 features
    and the output shape is (1, 3, 4).
    """
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        # A buffer moves with the model's device and is not a trainable parameter.
        self.register_buffer("mask",torch.triu(torch.ones(context_length, context_length),diagonal=1))

    def forward(self, x):
        # x contains embeddings: (b, num_tokens, d_in).
        b, num_tokens, d_in = x.shape

        # Project embeddings into queries, keys, and values: (b, num_tokens, d_out).
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Split projected features into heads; every head still sees every token.
        # (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim).
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compare queries and keys per head; rows are query tokens, columns are key tokens.
        # Scores: (b, num_heads, num_tokens, num_tokens).
        attn_scores = queries @ keys.transpose(2, 3)

        # Mask future tokens; softmax turns their -inf scores into zero weights.
        # The same mask broadcasts across all batches and heads.
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # Scale by sqrt(head_dim), then normalize over keys for each query.
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        # During training, drop and rescale attention weights for regularization.
        attn_weights = self.dropout(attn_weights)

        # Weighted sums of values: (b, num_heads, num_tokens, head_dim).
        # Move the token axis back before concatenating the heads.
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Transpose changes memory layout; contiguous() makes view() possible.
        # Join head features: (b, num_tokens, num_heads * head_dim).
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        # Mix features from all heads: (b, num_tokens, d_out).
        context_vec = self.out_proj(context_vec)

        return context_vec
