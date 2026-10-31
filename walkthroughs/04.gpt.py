import tiktoken
import torch

from llm.gpt import GPT_CONFIG_124M, GPTConfig, GPTModel


@torch.no_grad()
def main():
    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding("gpt2")
    texts = ["The river flows past", "A robot learns to"]
    # Both prompts have four tokens, so no padding is needed.
    batch = torch.tensor([tokenizer.encode(text) for text in texts])

    # Keep the GPT-2 vocabulary but use a small model for this walkthrough.
    config: GPTConfig = {
        **GPT_CONFIG_124M,
        "context_length": 16,
        "emb_dim": 32,
        "n_heads": 4,
        "n_layers": 2,
    }
    model = GPTModel(config).eval()  # Disable dropout for repeatable results.

    print("Prompts:", texts)
    print("Token IDs:\n", batch)
    print("Input shape (batch, tokens):", tuple(batch.shape))

    positions = torch.arange(batch.shape[1])
    x = model.token_emb(batch) + model.pos_emb(positions)
    x = model.drop_emb(x)
    print("Token + position embeddings:", tuple(x.shape))

    for index, block in enumerate(model.trf_blocks, start=1):
        x = block(x)
        print(f"Transformer block {index}:", tuple(x.shape))

    # Normalize each token's features independently; the shape stays unchanged.
    print("\nFirst token's feature mean before final_norm:", x[0, 0].mean().item())
    x = model.final_norm(x)
    print("First token's feature mean after final_norm:", x[0, 0].mean().item())
    print("First token's feature variance after final_norm:", x[0, 0].var(unbiased=False).item())

    logits = model.out_head(x)
    print("\nVocabulary scores (batch, tokens, vocabulary):", tuple(logits.shape))
    torch.testing.assert_close(logits, model(batch))

    # The last position predicts the token that follows each prompt.
    next_ids = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    extended = torch.cat((batch, next_ids), dim=1)
    for token_ids in extended.tolist():
        print(repr(tokenizer.decode(token_ids)))


if __name__ == "__main__":
    main()
