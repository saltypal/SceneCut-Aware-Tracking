"""Execute one project notebook in place using the isolated project kernel."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: execute_notebook.py NOTEBOOK_PATH")
    root = Path(__file__).resolve().parents[1]
    notebook_path = Path(sys.argv[1]).resolve()
    kernel_prefix = root / ".runtime" / "jupyter"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "ipykernel",
            "install",
            "--prefix",
            str(kernel_prefix),
            "--name",
            "scenecut-tracking",
            "--display-name",
            "SceneCut Tracking Python 3.13",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    os.environ["JUPYTER_PATH"] = str(kernel_prefix / "share" / "jupyter")
    notebook = nbformat.read(notebook_path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=1800,
        kernel_name="scenecut-tracking",
        resources={"metadata": {"path": str(root)}},
    )
    client.execute()
    nbformat.write(notebook, notebook_path)
    print(f"Executed {notebook_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
