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

Run the attention examples from the project root:

```sh
uv run python -m examples.causal_attention
uv run python -m examples.multi_head_attention
```

The multi-head example prints each calculation, its tensor shape, and how the
causal mask keeps future tokens from affecting earlier outputs.

Run the Chapter 4 GPT walkthrough:

```sh
uv run python -m examples.gpt
```

The example traces embedding and transformer shapes, shows the effect of final
LayerNorm, and appends one predicted token. It uses 32 features and two blocks
with the GPT-2 tokenizer; all weights are untrained.

## Dataset

The sample text is *Alice's Adventures in Wonderland* by Lewis Carroll, sourced
from [Project Gutenberg](https://www.gutenberg.org/ebooks/11).
The original license is included in `data/input.txt`.
