import os


def env_flag(name: str, default: bool = False) -> bool:
    """Parse common truthy environment variable values."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
