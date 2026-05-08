#!/usr/bin/env python3
"""Instance setup script.

Usage:
    python3 setup.py <group_name>

Reads groups/<group_name>.yaml, installs custom ComfyUI nodes and downloads
model files into the standard ComfyUI directories.

Environment variables (optional overrides):
    CUSTOM_NODES_DIR  — default: /workspaces/ComfyUI/custom_nodes
    MODELS_DIR        — default: /workspaces/ComfyUI/models
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap pyyaml if not present
# ---------------------------------------------------------------------------
try:
    import yaml
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyyaml", "-q"])
    import yaml  # type: ignore[no-redef]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CUSTOM_NODES_DIR = "/workspaces/ComfyUI/custom_nodes"
DEFAULT_MODELS_DIR = "/workspaces/ComfyUI/models"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _run(cmd: str, timeout: int = 600) -> bool:
    """Run a shell command, stream output to logger, return True on success."""
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    assert proc.stdout
    while True:
        line = await proc.stdout.readline()
        if not line:
            break
        logger.info("[CMD] %s", line.decode(errors="replace").rstrip())
    await asyncio.wait_for(proc.wait(), timeout=timeout)
    return proc.returncode == 0


# ---------------------------------------------------------------------------
# Node installation
# ---------------------------------------------------------------------------

async def install_nodes(nodes: list[dict], custom_nodes_dir: Path) -> None:
    custom_nodes_dir.mkdir(parents=True, exist_ok=True)
    for node in nodes:
        repo: str = node.get("repo", "")
        pip_deps: list[str] = node.get("pip") or []
        custom_subdir: str = node.get("custom_dir", "")
        if not repo:
            continue
        name = Path(repo).name.removesuffix(".git")
        target = custom_nodes_dir / (custom_subdir or name)
        if target.exists() and (target / ".git").exists():
            logger.info("[nodes] SKIP (already cloned): %s", name)
        else:
            logger.info("[nodes] Cloning: %s → %s", repo, target)
            ok = await _run(f'git clone --depth 1 "{repo}" "{target}"', timeout=300)
            if not ok:
                logger.error("[nodes] git clone failed: %s", repo)
                continue
        if pip_deps:
            deps = " ".join(f'"{d}"' for d in pip_deps)
            logger.info("[nodes] pip install %s", " ".join(pip_deps))
            await _run(f"pip install {deps} --quiet", timeout=180)


# ---------------------------------------------------------------------------
# Model downloading
# ---------------------------------------------------------------------------

async def download_models(models: list[dict], models_dir: Path) -> None:
    for model in models:
        url: str = model.get("url", "")
        filename: str = model.get("filename", "")
        subfolder: str = model.get("subfolder", "")
        if not url or not filename:
            continue
        dest_dir = models_dir / subfolder if subfolder else models_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / filename
        if dest_file.exists():
            size_mb = dest_file.stat().st_size / 1024 / 1024
            logger.info("[models] SKIP (exists, %.1f MB): %s", size_mb, filename)
            continue
        logger.info("[models] Downloading: %s → %s", filename, dest_dir)
        ok = await _run(
            f'curl -L --retry 3 --retry-delay 5 -# -o "{dest_file}" "{url}"',
            timeout=3600,
        )
        if ok and dest_file.exists():
            size_mb = dest_file.stat().st_size / 1024 / 1024
            logger.info("[models] Done: %s (%.1f MB)", filename, size_mb)
        else:
            logger.error("[models] Failed: %s", filename)
            dest_file.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def setup(group_name: str) -> None:
    config_path = REPO_ROOT / "groups" / f"{group_name}.yaml"
    if not config_path.exists():
        logger.error("Group config not found: %s", config_path)
        sys.exit(1)

    with config_path.open() as f:
        cfg = yaml.safe_load(f) or {}

    nodes: list[dict] = cfg.get("nodes") or []
    models: list[dict] = cfg.get("models") or []

    custom_nodes_dir = Path(os.getenv("CUSTOM_NODES_DIR", DEFAULT_CUSTOM_NODES_DIR))
    models_dir = Path(os.getenv("MODELS_DIR", DEFAULT_MODELS_DIR))

    logger.info("=== Setup: group=%s, nodes=%d, models=%d ===", group_name, len(nodes), len(models))

    if nodes:
        await install_nodes(nodes, custom_nodes_dir)
    if models:
        await download_models(models, models_dir)

    logger.info("=== Setup complete: group=%s ===", group_name)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} <group_name>", file=sys.stderr)
        sys.exit(1)
    asyncio.run(setup(sys.argv[1]))
