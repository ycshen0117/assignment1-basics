import torch


def decoding(
    model,
    prompt,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    if max_new_tokens <= 0:
        if torch.is_tensor(prompt):
            return prompt.to(dtype=torch.long)
        return torch.tensor(prompt, dtype=torch.long)

    if torch.is_tensor(prompt):
        input_ids = prompt.to(dtype=torch.long)
    else:
        input_ids = torch.tensor(prompt, dtype=torch.long)

    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = input_ids.device
    input_ids = input_ids.to(device)

    finished = None
    eos_token = None
    if eos_token_id is not None:
        finished = torch.zeros((input_ids.shape[0],), dtype=torch.bool, device=device)
        eos_token = torch.tensor(eos_token_id, device=device, dtype=input_ids.dtype)

    # Turn off training behaviors and gradients
    model.eval()
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            logits = model(input_ids) # shape (B, T, V)
            next_logits = logits[:, -1, :] # shape (B, V)

            if temperature is None or temperature == 0.0:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True) # shape (B, 1)
            else:
                scaled = next_logits / temperature
                probs = torch.softmax(scaled, dim=-1) # shape (B, V)
                if top_p < 1.0: # top-p sampling
                    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
                    cumulative = torch.cumsum(sorted_probs, dim=-1)
                    cutoff = cumulative > top_p
                    cutoff[..., 0] = False
                    sorted_probs = sorted_probs.masked_fill(cutoff, 0.0)
                    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                    next_token = torch.multinomial(sorted_probs, num_samples=1)
                    next_token = sorted_idx.gather(-1, next_token)
                else:
                    next_token = torch.multinomial(probs, num_samples=1)
            
            # Handle finished sequences
            if finished is not None:
                next_token = torch.where(finished.unsqueeze(-1), eos_token, next_token)

            input_ids = torch.cat([input_ids, next_token], dim=1) # append to sequence

            if finished is not None:
                finished |= (next_token.squeeze(-1) == eos_token_id)
                if bool(torch.all(finished)):
                    break

    return input_ids
