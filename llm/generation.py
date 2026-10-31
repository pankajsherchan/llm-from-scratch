import torch


def text_to_token_ids(text, tokenizer):
    return torch.tensor(tokenizer.encode(text, allowed_special={"<|endoftext|>"})).unsqueeze(0)


def token_ids_to_text(token_ids, tokenizer):
    return tokenizer.decode(token_ids.squeeze(0).tolist())

@torch.no_grad()
def generate(model, token_ids, max_new_tokens, temperature=0.0, top_k=None, eos_id=None):
    if temperature < 0:
        raise ValueError("temperature must be nonnegative")
    if top_k is not None and not 1 <= top_k <= model.out_head.out_features:
        raise ValueError("top_k must be between 1 and the vocabulary size")
    context_size = model.pos_emb.num_embeddings
    finished = torch.zeros(token_ids.shape[0], dtype=torch.bool, device=token_ids.device)

    for _ in range(max_new_tokens):
        token_ids_cond = token_ids[:, -context_size:]
        logits = model(token_ids_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            values, indices = torch.topk(logits, top_k, dim=-1)
            logits = torch.full_like(logits, -torch.inf).scatter(-1, indices, values)

        if temperature > 0:
            probabilities = torch.softmax(logits / temperature, dim=-1)
            next_id = torch.multinomial(probabilities, num_samples=1)
        else:
            next_id = logits.argmax(dim=-1, keepdim=True)

        if eos_id is not None:
            next_id = torch.where(finished[:, None], eos_id, next_id)
            finished |= next_id.squeeze(1).eq(eos_id)

        token_ids = torch.cat((token_ids, next_id), dim=1)
        if eos_id is not None and finished.all():
            break
    return token_ids
