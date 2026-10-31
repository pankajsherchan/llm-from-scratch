import argparse

import torch

from llm.attention import MultiHeadAttention


def show(label, tensor):
    print(f"\n{label} | shape={tuple(tensor.shape)}")
    print(tensor)


@torch.no_grad()
def main(batch_size=2, d_in=4, d_out=4, context_length=3, dropout=0.0,
         num_heads=2, training=False):
    for name, value in {
        "batch_size": batch_size, "d_in": d_in, "d_out": d_out,
        "context_length": context_length, "num_heads": num_heads,
    }.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if d_out % num_heads != 0:
        raise ValueError("d_out must be divisible by num_heads.")
    if not 0 <= dropout <= 1:
        raise ValueError("dropout must be between 0 and 1.")

    torch.manual_seed(42)
    torch.set_printoptions(precision=3, sci_mode=False, linewidth=100)

    # Use the full context_length for each generated sequence.
    # Attention processes each sequence independently, using the same learned weights.
    inputs = torch.rand(batch_size, context_length, d_in)
    
    attention = MultiHeadAttention(
        d_in=d_in, d_out=d_out, context_length=context_length,
        dropout=dropout, num_heads=num_heads,
    )
    attention.train(training)
    b, num_tokens, _ = inputs.shape
    print(f"head_dim = d_out // num_heads = {d_out} // {num_heads} = {attention.head_dim}")
    print(f"Mode: {'training' if training else 'evaluation'}; dropout probability: {dropout}")
    show("1. Input embeddings", inputs)

    # Each projection learns a different transformation of the same embeddings.
    queries = attention.W_query(inputs)
    keys = attention.W_key(inputs)
    values = attention.W_value(inputs)
    show("2a. Queries = inputs @ W_query.T", queries)
    show("2b. Keys = inputs @ W_key.T", keys)
    show("2c. Values = inputs @ W_value.T", values)

    # Each head has vectors for every token; the causal mask limits attention later.
    # (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
    # -> (b, num_heads, num_tokens, head_dim).
    def split_heads(tensor):
        return tensor.view(b, num_tokens, attention.num_heads, attention.head_dim).transpose(1, 2)

    queries, keys, values = map(split_heads, (queries, keys, values))
    show("3a. Queries grouped by head", queries)
    show("3b. Keys grouped by head", keys)
    show("3c. Values grouped by head", values)

    # Scores: (b, num_heads, num_tokens, num_tokens).
    # Rows are query tokens; columns are key tokens within the same sequence.
    scores = queries @ keys.transpose(-2, -1)
    show("4. Scores = Q @ K.transpose(-2, -1) within each head", scores)

    mask = attention.mask.bool()[:num_tokens, :num_tokens]
    scores = scores.masked_fill(mask, -torch.inf)
    show("5. Future-token scores replaced by -inf", scores)

    scaled_scores = scores / attention.head_dim**0.5
    show(f"6. Scores divided by sqrt({attention.head_dim}), the square root of head_dim", scaled_scores)

    weights = torch.softmax(scaled_scores, dim=-1)
    show("7. Softmax over keys: each row sums to 1; future weights are 0", weights)
    # Save the RNG state to replay the same dropout mask in the full forward pass.
    dropout_rng_state = torch.get_rng_state()
    weights = attention.dropout(weights)
    if training and dropout > 0:
        show("8. Dropout zeros some weights and rescales survivors; rows may no longer sum to 1", weights)
    else:
        print("\n8. Dropout leaves weights unchanged (evaluation mode or dropout=0).")

    head_outputs = weights @ values
    show("9. Each head's weighted sums = attention weights @ values", head_outputs)
    # Each token combines only its own and earlier value vectors within its sequence.

    combined = head_outputs.transpose(1, 2).contiguous().view(b, num_tokens, attention.d_out)
    show(f"10. Concatenate {num_heads} heads x {attention.head_dim} features into {d_out} features per token", combined)

    # Each output feature can use features from both heads for the same token.
    # This learned linear layer preserves shape and does not mix token positions.
    output = attention.out_proj(combined)
    show("11. Output projection = combined @ out_proj.weight.T + out_proj.bias", output)
    torch.set_rng_state(dropout_rng_state)
    print("\nWalkthrough matches attention(inputs):", torch.allclose(output, attention(inputs), atol=1e-6))

    # Compare in evaluation mode so random dropout masks do not affect the result.
    attention.eval()
    batched_output = attention(inputs)
    separate_outputs = torch.cat([attention(sequence.unsqueeze(0)) for sequence in inputs])
    print("Batched output matches running each sequence separately (evaluation mode):",
          torch.allclose(batched_output, separate_outputs, atol=1e-6))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Walk through multi-head causal attention.")
    parser.add_argument("--batch-size", type=int, default=2, help="Number of sequences (default: 2).")
    parser.add_argument("--d-in", type=int, default=4, help="Input features per token (default: 4).")
    parser.add_argument("--d-out", type=int, default=4, help="Total output features per token (default: 4).")
    parser.add_argument("--context-length", type=int, default=3, help="Tokens per generated sequence and maximum context (default: 3).")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability from 0 to 1 (default: 0).")
    parser.add_argument("--num-heads", type=int, default=2, help="Number of heads; must divide d_out (default: 2).")
    parser.add_argument("--training", action="store_true", help="Enable training mode to demonstrate dropout.")
    args = parser.parse_args()
    try:
        main(**vars(args))
    except ValueError as error:
        parser.error(str(error))
