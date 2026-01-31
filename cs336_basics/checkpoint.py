import torch
from omegaconf import ListConfig
from omegaconf.base import ContainerMetadata

def save_checkpoint(model, optimizer, iteration, out):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(src, model, optimizer):
    try:
        with torch.serialization.safe_globals([ListConfig, ContainerMetadata]):
            checkpoint = torch.load(src, weights_only=True)
    except Exception:
        # Fallback for trusted checkpoints that include non-tensor metadata.
        checkpoint = torch.load(src, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]