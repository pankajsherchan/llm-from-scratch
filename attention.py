from torch import nn
import torch 

class CausalAttention(nn.Module):
    """
    d_in: input embedding features per token.
    d_out: query, key, value, and output features per token.
    context_length: maximum sequence length

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
        # x contains embeddings: (batch_size, num_tokens, d_in).
        b, num_tokens, d_in = x.shape

        # Project embeddings into queries, keys, and values: (b, num_tokens, d_out).
        keys = self.W_key(x) 
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)

        # Mask future tokens; softmax turns their -inf scores into zero weights.
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)

        # Scale by sqrt(d_out), then normalize over keys for each query.
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights) 

        context_vec = attn_weights @ values
        return context_vec

class MultiHeadAttention(nn.Module):
    pass
