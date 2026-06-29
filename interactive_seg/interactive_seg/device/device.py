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

def sync_device(device: str) -> None:
    """Synchronizes the specified device to ensure all operations are complete.

    GPU backends (CUDA, MPS) execute operations asynchronously: a call like
    ``model.add_interaction(...)`` enqueues work on the device and returns
    control to the CPU immediately, before the GPU has actually finished
    running it. This makes async dispatch fast for normal inference, but it
    breaks naive timing — if you wrap ``time.perf_counter()`` around an
    operation without synchronizing, you measure only how long it took to
    *schedule* the work on the device, not how long the device took to
    *execute* it. The real computation can still be running in the
    background for milliseconds (or more) after the timer has already
    stopped, making the recorded duration artificially small and unreliable.

    Calling this function forces the CPU to block until every operation
    queued on the device has actually completed, so that a
    ``perf_counter()`` call placed immediately after it reflects real
    finished work rather than an in-flight launch. To get an accurate
    duration, this must be called both immediately before starting the
    timer (to drain any leftover work from a previous step that could
    otherwise leak into the next measurement) and immediately before
    stopping it (to guarantee the current step's work has finished).

    CPU execution is already synchronous, so no equivalent call is needed
    there — ops block until done by default.

    Args:
        device (str): The device to synchronize. One of:
            - "cuda"
            - "mps"
            - "cpu" (no synchronization needed)
    """
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
