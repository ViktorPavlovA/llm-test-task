#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_HOST_CONFIG = os.path.join(_HERE, "models", "config.yaml")
CONFIG_PATH = os.environ.get(
    "MODEL_CONFIG",
    _HOST_CONFIG if os.path.isfile(_HOST_CONFIG) else "/models/config.yaml",
)

# Two GPUs on the host; Tesla V100 is often unsupported, so it is left unused.
ALLOW_CPU = os.environ.get("ALLOW_CPU", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

GGUF_NATIVE_TEMPLATES = {
    "",
    "auto",
    "gguf",
    "from-gguf",
    "qwen",
    "qwen2",
    "qwen2.5",
    "qwen3",
    "qwen3.5",
    "qwen3.8",
}

MODEL_SCRIPT = "download_model_script.py"
MODEL_FILENAME = "Qwen3.5-4B-UD-Q4_K_XL.gguf"


def load_config(path: str) -> dict:
    """
    Load the llama-server YAML config.

    Parameters
    ----------
    path : str
        Path to ``config.yaml``.

    Returns
    -------
    data : dict
        Mapping of launcher options.

    """
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        sys.exit(f"config must be a mapping: {path}")
    return data


def gpu_layers(value) -> str:
    """
    Normalize ``n_gpu_layers`` for the ``--n-gpu-layers`` flag.

    Parameters
    ----------
    value : Any
        Config value: auto, all, -1, or a layer count.

    Returns
    -------
    layers : str
        Flag value for llama-server.

    """
    if value in (None, "", "auto", "AUTO"):
        return "auto"
    if value in (-1, "-1", "all", "ALL"):
        return "all"
    return str(value)


def flash_attn_mode(value) -> str:
    """
    Normalize flash-attention config to on/off/auto.

    Parameters
    ----------
    value : Any
        Boolean or string from YAML.

    Returns
    -------
    mode : str
        ``on``, ``off``, or ``auto``.

    """
    if value is True:
        return "on"
    if value is False:
        return "off"
    if value in (None, "", "auto", "AUTO"):
        return "auto"
    return str(value)


def as_bool(value, default: bool = False) -> bool:
    """
    Coerce a YAML or env-like value to bool.

    Parameters
    ----------
    value : Any
        Raw config value.
    default : bool, optional
        Result when ``value`` is None. Default is False.

    Returns
    -------
    flag : bool
        Parsed boolean.

    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def resolve_path(value: str, base_dir: str) -> str:
    """
    Expand ``~`` and join relative paths to ``base_dir``.

    Parameters
    ----------
    value : str
        File path from config.
    base_dir : str
        Directory that contains the config file.

    Returns
    -------
    path : str
        Absolute path.

    """
    value = os.path.expanduser(str(value))
    if os.path.isabs(value):
        return value
    return os.path.abspath(os.path.join(base_dir, value))


def find_script(script_name: str, models_dir: str) -> str | None:
    """
    Locate a download script next to the weights or this launcher.

    Parameters
    ----------
    script_name : str
        File name, for example ``download_model_script.py``.
    models_dir : str
        Directory of the model file.

    Returns
    -------
    path : str or None
        First existing candidate, or None.

    """
    candidates = [
        os.path.join(models_dir, script_name),
        os.path.join(os.path.dirname(os.path.abspath(CONFIG_PATH)), script_name),
        os.path.join(_HERE, "models", script_name),
        os.path.join(_HERE, script_name),
    ]
    seen = set()
    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        if candidate in seen:
            continue
        seen.add(candidate)
        if os.path.isfile(candidate):
            return candidate
    return None


def ensure_weight(
    path: str,
    script_name: str,
    known_name: str,
    label: str,
) -> None:
    """
    Download a known weight file when it is missing or incomplete.

    The script returns immediately once the file is complete and resumes
    a partial wget. Any other filename is left to the operator.

    Parameters
    ----------
    path : str
        Destination GGUF path.
    script_name : str
        Download helper to run.
    known_name : str
        Basename this launcher is allowed to fetch.
    label : str
        Log label, for example ``model``.

    """
    base = os.path.basename(path)
    if base != known_name:
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            print(f"{label} ready: {path}", flush=True)
            return
        sys.exit(
            f"{label} not found: {path}. "
            f"This launcher only auto-downloads {known_name}."
        )

    script = find_script(script_name, os.path.dirname(path))
    if script is None:
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            print(f"{label} ready: {path} (no download script)", flush=True)
            return
        sys.exit(f"{label} not found: {path}, and {script_name} is missing")

    print(f"{label}: {script}", flush=True)
    try:
        subprocess.run([sys.executable, script], check=True)
    except subprocess.CalledProcessError as exc:
        sys.exit(f"download failed ({script}), exit code {exc.returncode}")
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        sys.exit(f"{label} still missing after download: {path}")


def require_cuda() -> None:
    """
    Abort when llama.cpp cannot see a CUDA device, unless ``ALLOW_CPU=1``.

    """
    llama_server = shutil.which("llama-server")
    if not llama_server:
        sys.exit("llama-server is not on PATH")

    print("=== llama.cpp devices ===", flush=True)
    result = subprocess.run(
        [llama_server, "--list-devices"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output = result.stdout or ""
    print(output, end="" if output.endswith("\n") else "\n", flush=True)
    if result.returncode != 0:
        sys.exit(
            "llama-server --list-devices failed "
            f"(exit {result.returncode}). The log above is the real error."
        )
    if ALLOW_CPU:
        return
    if "CUDA" not in output.upper():
        sys.exit(
            "No CUDA device visible to llama.cpp. "
            "Refusing CPU fallback. Set ALLOW_CPU=1 to override, "
            "or pass host GPU 0 into the container (device_ids: ['0'])."
        )


def build_command(cfg: dict, config_dir: str) -> list[str]:
    """
    Assemble the ``llama-server`` argv from YAML.

    Parameters
    ----------
    cfg : dict
        Loaded config mapping.
    config_dir : str
        Directory that contains the config file.

    Returns
    -------
    cmd : list of str
        Argument vector for ``os.execvp``.

    """
    model_value = cfg.get("model_path")
    if not model_value:
        sys.exit("model_path is missing from config")
    model_path = resolve_path(model_value, config_dir)
    ensure_weight(model_path, MODEL_SCRIPT, MODEL_FILENAME, "model")

    cmd = [
        "llama-server",
        "--model",
        model_path,
        "--host",
        os.environ.get("HOST", "0.0.0.0"),
        "--port",
        os.environ.get("PORT", "8686"),
        "--ctx-size",
        str(cfg.get("context_length", 8192)),
        "--n-gpu-layers",
        gpu_layers(cfg.get("n_gpu_layers", "auto")),
        "--flash-attn",
        flash_attn_mode(cfg.get("flash_attn", "auto")),
        "--alias",
        str(cfg.get("alias", "local-llm")),
        "--metrics",
    ]
    split_mode = cfg.get("split_mode")
    if split_mode:
        cmd.extend(["--split-mode", str(split_mode)])
    if cfg.get("main_gpu") is not None:
        cmd.extend(["--main-gpu", str(cfg.get("main_gpu"))])
    tensor_split = cfg.get("tensor_split")
    if tensor_split:
        cmd.extend(["--tensor-split", str(tensor_split)])
    device = cfg.get("device")
    if device:
        cmd.extend(["--device", str(device)])

    if as_bool(cfg.get("webui", False), default=False):
        cmd.append("--webui")
    else:
        cmd.append("--no-webui")
    if as_bool(cfg.get("jinja", True), default=True):
        cmd.append("--jinja")
    else:
        cmd.append("--no-jinja")

    template_file = cfg.get("chat_template_file")
    if template_file:
        template_file = resolve_path(str(template_file), config_dir)
        if not os.path.isfile(template_file):
            sys.exit(f"chat template file not found: {template_file}")
        cmd.extend(["--chat-template-file", template_file])
    else:
        chat_format = str(cfg.get("chat_format") or "auto").strip().lower()
        if chat_format not in GGUF_NATIVE_TEMPLATES:
            cmd.extend(["--chat-template", chat_format])

    reasoning = cfg.get("reasoning_format")
    if reasoning:
        cmd.extend(["--reasoning-format", str(reasoning)])

    reasoning_mode = cfg.get("reasoning")
    if reasoning_mode is not None:
        if isinstance(reasoning_mode, bool):
            reasoning_mode = "on" if reasoning_mode else "off"
        cmd.extend(["--reasoning", str(reasoning_mode)])

    reasoning_budget = cfg.get("reasoning_budget")
    if reasoning_budget is not None:
        cmd.extend(["--reasoning-budget", str(reasoning_budget)])

    kwargs = cfg.get("chat_template_kwargs")
    if kwargs:
        if isinstance(kwargs, dict):
            kwargs = json.dumps(kwargs, ensure_ascii=False)
        elif not isinstance(kwargs, str):
            sys.exit("chat_template_kwargs must be a mapping or a JSON string")
        cmd.extend(["--chat-template-kwargs", kwargs])

    append_lora(cmd, cfg, config_dir)

    extra = cfg.get("extra_args") or []
    if extra:
        if not isinstance(extra, list) or not all(
            isinstance(item, str) for item in extra
        ):
            sys.exit("extra_args must be a list of strings")
        cmd.extend(extra)
    return cmd


def append_lora(cmd: list[str], cfg: dict, config_dir: str) -> None:
    """
    Append LoRA flags when ``lora_path`` is set.

    Parameters
    ----------
    cmd : list of str
        Argument vector, mutated in place.
    cfg : dict
        Loaded config mapping.
    config_dir : str
        Directory that contains the config file.

    """
    lora_path = cfg.get("lora_path")
    if not lora_path:
        return
    lora_path = resolve_path(str(lora_path), config_dir)
    if not os.path.isfile(lora_path):
        sys.exit(f"LoRA adapter not found: {lora_path}")
    scale = cfg.get("lora_scale")
    if scale is None:
        cmd.extend(["--lora", lora_path])
    else:
        cmd.extend(["--lora-scaled", f"{lora_path}:{scale}"])


def main() -> None:
    """
    Load config, ensure weights, and exec llama-server.

    """
    if not os.path.isfile(CONFIG_PATH):
        sys.exit(f"config not found: {CONFIG_PATH}")
    cfg = load_config(CONFIG_PATH)
    cmd = build_command(cfg, os.path.dirname(os.path.abspath(CONFIG_PATH)))
    require_cuda()
    print("starting:", " ".join(cmd), flush=True)
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
