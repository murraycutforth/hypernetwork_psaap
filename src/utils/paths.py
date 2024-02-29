from pathlib import Path


def project_dir() -> Path:
    return Path(__file__).parent.parent.parent