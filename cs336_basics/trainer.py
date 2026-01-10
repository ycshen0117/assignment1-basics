import argparse
import os
import time

import numpy as np
import torch

from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW, cross_entropy, gradient_clipping, learning_rate_schedule


def data_loading(
    x: np.ndarray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    x_tensor = torch.as_tensor(x, dtype=torch.long)
    num_possible_starting_indices = x_tensor.shape[0] - context_length
    starts = torch.randint(0, num_possible_starting_indices, (batch_size,), dtype=torch.long)
    offsets = torch.arange(context_length, dtype=torch.long)
    idx = starts[:, None] + offsets[None, :]
    x_batch = x_tensor[idx]
    y_batch = x_tensor[idx + 1]
    if device != "cpu":
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
    return x_batch, y_batch


def save_checkpoint(model, optimizer, iteration, out):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(src, model, optimizer):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]


def _load_dataset(path: str, dtype: str, fmt: str) -> np.ndarray:
    if fmt == "npy":
        return np.load(path, mmap_mode="r")
    return np.memmap(path, dtype=np.dtype(dtype), mode="r")


def _resolve_checkpoint_path(base_path: str, iteration: int) -> str:
    if base_path.endswith(os.sep) or os.path.isdir(base_path):
        os.makedirs(base_path, exist_ok=True)
        return os.path.join(base_path, f"ckpt_{iteration}.pt")
    os.makedirs(os.path.dirname(base_path) or ".", exist_ok=True)
    return base_path


def _maybe_init_wandb(args):
    if not args.use_wandb:
        return None
    try:
        import wandb
    except ImportError:
        print("wandb not installed; skipping external logging.")
        return None
    return wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=vars(args),
    )


def evaluate(
    model: torch.nn.Module,
    data: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    batches: int,
) -> float:
    model.eval()
    total_loss = 0.0
    with torch.inference_mode():
        for _ in range(batches):
            x_batch, y_batch = data_loading(data, batch_size, context_length, device)
            logits = model(x_batch)
            loss = cross_entropy(logits, y_batch)
            total_loss += loss.item()
    model.train()
    return total_loss / max(batches, 1)


def main() -> None:
    # Argument parsing
    parser = argparse.ArgumentParser(description="Train a TransformerLM on tokenized data.")
    parser.add_argument("--train-data", required=True, help="Path to training tokens.")
    parser.add_argument("--val-data", default=None, help="Path to validation tokens.")
    parser.add_argument("--data-format", choices=["memmap", "npy"], default="memmap")
    parser.add_argument("--data-dtype", default="int32")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--max-iters", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1337)

    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--theta", type=float, default=10000.0)
    parser.add_argument("--eps", type=float, default=1e-5)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-iters", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="cs336-basics")
    parser.add_argument("--wandb-run-name", default=None)

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Pick device and load datasets
    device = torch.device(args.device)
    train_data = _load_dataset(args.train_data, args.data_dtype, args.data_format)
    val_data = None
    if args.val_data:
        val_data = _load_dataset(args.val_data, args.data_dtype, args.data_format)

    # Initialize model and optimizer
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.theta,
        eps=args.eps,
        device=device,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )

    # Resume from a checkpoint
    start_iter = 0
    if args.resume_from:
        start_iter = load_checkpoint(args.resume_from, model, optimizer)

    # Initialize Weights & Biases logging
    run = _maybe_init_wandb(args)

    # Training loop
    model.train()
    start_time = time.time()

    for iteration in range(start_iter, args.max_iters):
        lr = learning_rate_schedule(
            iteration,
            alpha_max=args.lr,
            alpha_min=args.min_lr,
            T_w=args.warmup_iters,
            T_c=args.max_iters,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        # Sample a random training batch
        x_batch, y_batch = data_loading(train_data, args.batch_size, args.context_length, args.device)

        # Forward pass + loss
        logits = model(x_batch)
        loss = cross_entropy(logits, y_batch)
        
        # Backward pass + optimization step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            gradient_clipping(model.parameters(), args.grad_clip)
        optimizer.step()

        # Periodic logging (console + wandb)
        if iteration % args.log_interval == 0:
            elapsed = time.time() - start_time
            tokens = args.batch_size * args.context_length * max(1, iteration - start_iter + 1)
            tps = tokens / max(elapsed, 1e-6)
            msg = f"iter {iteration} | loss {loss.item():.4f} | lr {lr:.2e} | tok/s {tps:.1f}"
            print(msg)
            if run is not None:
                run.log({"train/loss": loss.item(), "lr": lr, "tokens_per_s": tps}, step=iteration)

        # Periodic evaluation on validation set
        if val_data is not None and iteration % args.eval_interval == 0 and iteration != start_iter:
            val_loss = evaluate(
                model,
                val_data,
                args.batch_size,
                args.context_length,
                args.device,
                args.eval_batches,
            )
            print(f"iter {iteration} | val_loss {val_loss:.4f}")
            if run is not None:
                run.log({"val/loss": val_loss}, step=iteration)

        # Periodic checkpointing
        if args.checkpoint_interval > 0 and iteration % args.checkpoint_interval == 0:
            ckpt_path = _resolve_checkpoint_path(args.checkpoint_path, iteration)
            save_checkpoint(model, optimizer, iteration, ckpt_path)
    
    # Final checkpoint at the end of training
    ckpt_path = _resolve_checkpoint_path(args.checkpoint_path, args.max_iters)
    save_checkpoint(model, optimizer, args.max_iters, ckpt_path)

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
