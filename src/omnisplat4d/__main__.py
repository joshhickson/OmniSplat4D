"""Entry point for `python -m omnisplat4d` and the `omnisplat4d` CLI script."""

import sys
from pathlib import Path


def main() -> None:
    """Thin wrapper that delegates to run_pipeline.py for CLI usage."""
    # Locate run_pipeline.py relative to the installed package
    repo_root = Path(__file__).resolve().parent.parent.parent
    run_script = repo_root / "run_pipeline.py"
    if not run_script.exists():
        print("run_pipeline.py not found. Run from the repo root.", file=sys.stderr)
        sys.exit(1)
    # Execute run_pipeline in-process by importing it
    import importlib.util

    spec = importlib.util.spec_from_file_location("run_pipeline", run_script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
