import torch

from llm.attention import CausalAttention


def main():
    torch.manual_seed(42)

    # One sequence of 3 tokens, each with 4 embedding features.
    # These are example embeddings, not token IDs.
    inputs = torch.tensor([
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 1.0, 1.1, 1.2],
    ]).unsqueeze(0)  # Add the batch dimension: (1, 3, 4).

    attention = CausalAttention(
        d_in=4, d_out=2, context_length=3, dropout=0.0
    )
    attention.eval()

    with torch.no_grad():
        output = attention(inputs)

    print("Input shape (batch, tokens, features):", tuple(inputs.shape))
    print("Output shape (batch, tokens, features):", tuple(output.shape))
    print("\nOutput for each token:\n", output[0])

if __name__ == "__main__":
    main()
