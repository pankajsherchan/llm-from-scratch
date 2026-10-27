# GPT-2 from scratch

A learning project to build a GPT-2-style language model from scratch using Python
and PyTorch, covering text preparation, attention, transformer blocks, training,
and text generation.  

## Getting started

Install dependencies with [uv](https://docs.astral.sh/uv/):

```sh
uv sync
uv run python main.py
```

## Dataset

The sample text is *Alice's Adventures in Wonderland* by Lewis Carroll, sourced
from [Project Gutenberg](https://www.gutenberg.org/ebooks/11).
The original license is included in `data/input.txt`.
