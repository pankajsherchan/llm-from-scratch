from pathlib import Path

import tiktoken
import torch

from llm.data_loader import create_dataloader
from llm.generation import generate, text_to_token_ids, token_ids_to_text
from llm.gpt import GPT_CONFIG_124M, GPTConfig, GPTModel
from llm.training import evaluate_model, load_checkpoint, save_checkpoint, train_model_simple


def main():
    num_epochs = 2
    checkpoint_path = Path("checkpoints/gpt.pt")

    torch.manual_seed(123)
    device = torch.device("cpu")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    tokenizer = tiktoken.get_encoding("gpt2")
    config: GPTConfig = {**GPT_CONFIG_124M, "context_length": 32,
                         "emb_dim": 32, "n_heads": 4, "n_layers": 2}

    text = (Path(__file__).resolve().parents[1] / "data/input.txt").read_text(encoding="utf-8")

    # Split before creating windows so training and validation never share tokens.
    split = int(0.9 * len(text))
    loader_options = dict(batch_size=2, max_length=config["context_length"],
                          stride=config["context_length"], drop_last=False)
    train_loader = create_dataloader(text[:split], shuffle=True, **loader_options)
    val_loader = create_dataloader(text[split:], shuffle=False, **loader_options)
    model = GPTModel(config).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
    initial = evaluate_model(model, train_loader, val_loader, device, eval_iter=5)
    print(f"Before training: train {initial[0]:.3f} | validation {initial[1]:.3f}")

    train_model_simple(model, train_loader, val_loader, optimizer, device,
                       num_epochs=num_epochs, eval_freq=20, eval_iter=5)

    final = evaluate_model(model, train_loader, val_loader, device, eval_iter=5)
    print(f"After training: train {final[0]:.3f} | validation {final[1]:.3f}")

    model.eval()
    prompt = text_to_token_ids("Alice was", tokenizer).to(device)
    for temperature in (0.0, 0.8):
        generated = generate(model, prompt, max_new_tokens=30,
                             temperature=temperature, top_k=25)
        print(f"Temperature {temperature}: {token_ids_to_text(generated, tokenizer)!r}")

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(checkpoint_path, model, optimizer, config)
    restored = GPTModel(config).to(device).eval()
    load_checkpoint(checkpoint_path, restored, device=device)
    with torch.no_grad():
        torch.testing.assert_close(model(prompt), restored(prompt))
    print(f"Saved and verified checkpoint: {checkpoint_path}")

if __name__ == "__main__":
    main()
