"""Chapter 5: measure next-token prediction loss and update GPT weights."""

import torch
import torch.nn.functional as F


def calc_loss_batch(input_batch, target_batch, model, device):
    logits = model(input_batch.to(device))
    return F.cross_entropy(logits.flatten(0, 1), target_batch.to(device).flatten())


@torch.no_grad()
def calc_loss_loader(data_loader, model, device, num_batches=None):
    num_batches = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))
    
    total_loss, total_tokens = 0.0, 0
    for index, (inputs, targets) in enumerate(data_loader):
        if index >= num_batches:
            break
        loss = calc_loss_batch(inputs, targets, model, device)
        total_loss += loss.item() * targets.numel()
        total_tokens += targets.numel()

    return total_loss / total_tokens


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    was_training = model.training
    model.eval()
    try:
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    finally:
        model.train(was_training)
    return train_loss, val_loss


def train_model_simple(model, train_loader, val_loader, optimizer, device,
                       num_epochs, eval_freq=5, eval_iter=5):
    history = []
    tokens_seen = 0
    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(inputs, targets, model, device)
            loss.backward()
            optimizer.step() # Update model weights using loss gradients

            tokens_seen += inputs.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                history.append(dict(step=global_step, tokens_seen=tokens_seen,
                                    train_loss=train_loss, val_loss=val_loss))
                print(f"Epoch {epoch + 1} | step {global_step:04d} | "
                      f"train {train_loss:.3f} | validation {val_loss:.3f}")
    return history


def save_checkpoint(path, model, optimizer, config):
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "config": config}, path)


def load_checkpoint(path, model, optimizer=None, device="cpu"):
    """Restore model weights and optionally optimizer state into matching objects."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["config"]
