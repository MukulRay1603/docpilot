"""Tiny helper kept dependency-free so it can be unit-tested."""


def gpu_session_active(session_providers: list[str]) -> bool:
    """True only if ONNX Runtime actually placed the session on CUDA.

    `ort.get_available_providers()` lists what the installed wheel was built
    with, not what can load. If CUDA/cuDNN are missing, the session falls back
    to CPU and `session.get_providers()` starts with CPUExecutionProvider.
    """
    return bool(session_providers) and session_providers[0] == "CUDAExecutionProvider"
