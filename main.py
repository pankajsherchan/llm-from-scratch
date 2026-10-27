"""Inspect a batch of text prepared for next-token prediction."""

from pathlib import Path

from data_loader import create_dataloader


def main():
    text_path = Path(__file__).resolve().parent / "data" / "input.txt"
    raw_text = text_path.read_text(encoding="utf-8")
    print("Total number of characters:", len(raw_text))

    dataloader = create_dataloader(
        raw_text, batch_size=2, max_length=4, stride=4, shuffle=False
    )
    inputs, targets = next(iter(dataloader))
    print("Inputs:\n", inputs)
    print("Targets:\n", targets)
    print("Batch shape (batch_size, max_length):", inputs.shape)


if __name__ == "__main__":
    main()
