import numpy as np
import torch

def data_loading(
    x: np.ndarray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    # Generate random starting indices for the batches
    start_indices = np.random.randint(0, len(x) - context_length, size=(batch_size,))

    # Create indices for contiguous blocks of size context_length + 1
    offsets = np.arange(context_length + 1)
    block_indices = start_indices[:, None] + offsets

    # Index the dataset once to get all data blocks
    data_blocks = torch.from_numpy(x[block_indices].astype(np.int64))

    # Create x and y by slicing the data blocks. This is a very fast operation.
    x = data_blocks[:, :-1]
    y = data_blocks[:, 1:]

    # Move tensors to the specified device
    x, y = x.to(device), y.to(device)
    return x, y