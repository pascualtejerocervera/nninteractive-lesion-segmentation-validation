import torch

def get_device() -> str:
    """
    Automatically detect the best available compute device.

    Returns:
        str: Selected device name. One of:
            - "cuda"
            - "mps"
            - "cpu"
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"