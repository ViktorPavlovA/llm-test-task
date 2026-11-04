import subprocess
import sys
from pathlib import Path

URL = (
    "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/"
    "Qwen3.5-4B-UD-Q4_K_XL.gguf?download=true"
)
FILE_NAME = "Qwen3.5-4B-UD-Q4_K_XL.gguf"
# Size of the file already published for this quant (UD-Q4_K_XL).
MIN_BYTES = 2_912_109_728

script_dir = Path(__file__).parent
output_path = script_dir / FILE_NAME


def complete(path: Path) -> bool:
    """
    Return True when the GGUF file exists and is large enough.

    Parameters
    ----------
    path : Path
        Expected weight file.

    Returns
    -------
    ready : bool
        True if size is at least ``MIN_BYTES``.

    """
    return path.is_file() and path.stat().st_size >= MIN_BYTES


if complete(output_path):
    print(f"Model already present: {output_path}")
    print(f"Size: {output_path.stat().st_size / (1024**3):.2f} GB")
    sys.exit(0)

if output_path.is_file() and output_path.stat().st_size < 1024 * 1024:
    print(f"Removing damaged file: {output_path}")
    output_path.unlink()

print(f"Downloading model to: {output_path}")

try:
    subprocess.run(
        [
            "wget",
            "--continue",
            "--progress=bar:force",
            "-O",
            str(output_path),
            URL,
        ],
        check=True,
    )
except FileNotFoundError:
    print("Error: wget is not installed.")
    sys.exit(1)
except subprocess.CalledProcessError as exc:
    print(f"Download failed: {exc}")
    sys.exit(exc.returncode or 1)

if not complete(output_path):
    size = output_path.stat().st_size if output_path.is_file() else 0
    print(f"Incomplete file: {size} bytes, need at least {MIN_BYTES}")
    sys.exit(1)

print(f"\nDone: {output_path}")
print(f"Size: {output_path.stat().st_size / (1024**3):.2f} GB")
