import torch
from torch.utils.data import Dataset


class GPTDataset(Dataset):
    """Split text into fixed-length windows, moving by stride tokens."""

    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        if len(token_ids) <= max_length:
            raise ValueError(
                "At least max_length + 1 tokens are required to create an input sequence "
                "and its next-token targets."
            )

        # Each target is the next token after the corresponding input token.
        # Example with max_length = 4:
        # token_ids:    [10, 20, 30, 40, 50]
        # input_chunk:  [10, 20, 30, 40]
        # target_chunk: [20, 30, 40, 50]  (shifted one token ahead)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]
