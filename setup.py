#!/usr/bin/env python3
"""Instance setup + agent runner.

Usage:
    python3 setup.py <group_name>

Phase 1 – Provision: installs custom ComfyUI nodes and downloads model files.
Phase 2 – Agent loop: starts heartbeat + polling loops immediately and
           executes ComfyUI tasks until the process is terminated.

Environment variables:
    CUSTOM_NODES_DIR  — default: /workspace/ComfyUI/custom_nodes
    MODELS_DIR        — default: /workspace/ComfyUI/models
    SERVER_URL        — scheduler server base URL (default: http://localhost:8000)
    CONTAINER_ID      — instance identifier
    API_SECRET        — HMAC secret for request signing (optional)
    HEARTBEAT_INTERVAL — seconds between heartbeats (default: 5)
    INSTANCE_GROUP_NAME — group name passed at container launch
    GPU_NAME          — GPU display name passed at container launch
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Bootstrap dependencies if not present
# ---------------------------------------------------------------------------
for _pkg in ("pyyaml", "httpx"):
    try:
        __import__("yaml" if _pkg == "pyyaml" else _pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", _pkg, "-q"])

import yaml          # type: ignore[no-redef]
import httpx         # type: ignore[no-redef]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CUSTOM_NODES_DIR = "/workspace/ComfyUI/custom_nodes"
DEFAULT_MODELS_DIR = "/workspace/ComfyUI/models"


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
    pending = ""
    try:
        while True:
            # Read fixed-size chunks so commands that print long progress lines
            # (without newlines) do not trigger StreamReader line-length errors.
            chunk = await proc.stdout.read(4096)
            if not chunk:
                break

            pending += chunk.decode(errors="replace").replace("\r", "\n")

            while "\n" in pending:
                line, pending = pending.split("\n", 1)
                line = line.rstrip()
                if line:
                    logger.info("[CMD] %s", line)

            # Flush oversized partial content to avoid unbounded buffer growth.
            if len(pending) > 8192:
                line = pending.rstrip()
                if line:
                    logger.info("[CMD] %s", line)
                pending = ""

        if pending.strip():
            logger.info("[CMD] %s", pending.rstrip())

        await asyncio.wait_for(proc.wait(), timeout=timeout)
        return proc.returncode == 0
    except asyncio.TimeoutError:
        logger.error("[CMD] Timeout after %ss: %s", timeout, cmd)
        proc.kill()
        await proc.wait()
        return False


def _make_auth_headers() -> dict[str, str]:
    """生成 X-Timestamp + X-Signature 认证头（API_SECRET 为空时跳过认证）。"""
    secret = os.getenv("API_SECRET", "")
    if not secret:
        return {}
    ts = str(int(time.time()))
    sig = hashlib.md5(f"{ts}{secret}".encode()).hexdigest()
    return {"X-Timestamp": ts, "X-Signature": sig}


# ---------------------------------------------------------------------------
# Phase 1: Node installation
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
# Phase 1: Model downloading
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


def _load_group_setup_config(group_name: str) -> tuple[list[dict], list[dict], Path, Path]:
    config_path = REPO_ROOT / "groups" / f"{group_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Group config not found: {config_path}")

    with config_path.open() as f:
        cfg = yaml.safe_load(f) or {}

    nodes: list[dict] = cfg.get("nodes") or []
    models: list[dict] = cfg.get("models") or []
    custom_nodes_dir = Path(os.getenv("CUSTOM_NODES_DIR", DEFAULT_CUSTOM_NODES_DIR))
    models_dir = Path(os.getenv("MODELS_DIR", DEFAULT_MODELS_DIR))
    return nodes, models, custom_nodes_dir, models_dir


async def _run_provision_jobs(nodes: list[dict], models: list[dict], custom_nodes_dir: Path, models_dir: Path) -> None:
    jobs: list[asyncio.Task[None]] = []
    if nodes:
        jobs.append(asyncio.create_task(install_nodes(nodes, custom_nodes_dir)))
    if models:
        jobs.append(asyncio.create_task(download_models(models, models_dir)))

    if jobs:
        await asyncio.gather(*jobs)
    else:
        logger.info("[setup] no nodes/models to provision")


async def provision_in_background(group_name: str, on_finished: Callable[[], None] | None = None) -> None:
    try:
        nodes, models, custom_nodes_dir, models_dir = _load_group_setup_config(group_name)
        logger.info(
            "=== Background setup: group=%s, nodes=%d, models=%d ===",
            group_name,
            len(nodes),
            len(models),
        )
        await _run_provision_jobs(nodes, models, custom_nodes_dir, models_dir)

        logger.info("=== Background setup complete: group=%s ===", group_name)
    except Exception:
        logger.exception("[setup] background provisioning failed")
    finally:
        if on_finished:
            on_finished()


# ---------------------------------------------------------------------------
# Phase 2: Agent loop
# ---------------------------------------------------------------------------

@dataclass
class ComfyUIMessage:
    prompt_id: str
    status: str  # "pending" | "in_progress" | "completed" | "error"
    outputs: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class ComfyUIClient:
    def __init__(self, ip: str = "127.0.0.1", port: int = 18188, timeout: float = 120.0):
        self._base = f"http://{ip}:{port}"
        self._timeout = timeout

    async def submit_prompt(self, graph: dict[str, Any], client_id: str) -> str:
        async with httpx.AsyncClient(timeout=self._timeout) as c:
            resp = await c.post(f"{self._base}/prompt", json={"prompt": graph, "client_id": client_id})
            resp.raise_for_status()
        prompt_id: str = resp.json().get("prompt_id", "")
        logger.info("[ComfyUI] prompt accepted, id=%s", prompt_id)
        return prompt_id

    async def get_history(self, prompt_id: str) -> ComfyUIMessage:
        async with httpx.AsyncClient(timeout=self._timeout) as c:
            resp = await c.get(f"{self._base}/history/{prompt_id}")
            resp.raise_for_status()
            data = resp.json()
        entry = data.get(prompt_id, {})
        outputs = entry.get("outputs", {})
        status = "completed" if entry.get("status") else "in_progress"
        error: str | None = None
        images: list[dict] = []
        for node_out in outputs.values():
            if isinstance(node_out, dict):
                if node_out.get("error"):
                    error = node_out["error"]
                    status = "error"
                for img in node_out.get("images", []):
                    images.append({
                        "filename": img.get("filename", ""),
                        "subfolder": img.get("subfolder", ""),
                        "type": img.get("type", "output"),
                    })
        return ComfyUIMessage(prompt_id=prompt_id, status=status, outputs={"images": images}, error=error)


class InstanceAgent:
    _AGENT_VERSION = "1.0.0"

    def __init__(
        self,
        server_url: str,
        heartbeat_interval: int = 5,
        poll_interval: int = 5,
        group_name: str = "",
        gpu_name: str = "",
    ):
        self._server = server_url.rstrip("/")
        self._instance_id = os.getenv("CONTAINER_ID", "")
        self._group_name = group_name
        self._gpu_name = gpu_name or os.getenv("GPU_NAME", "")
        self._heartbeat_interval = heartbeat_interval
        self._poll_interval = poll_interval
        self._status = "provisioning"
        self._current_task_id: str | None = None
        self._current_task: dict | None = None
        self._stop = asyncio.Event()
        self._active_tasks: list[asyncio.Task] = []

    def mark_provisioning_done(self) -> None:
        if self._status == "provisioning":
            self._status = "idle"
            logger.info("[agent] provisioning done, switch status to idle")

    def add_background_task(self, task: asyncio.Task) -> None:
        self._active_tasks.append(task)

    async def start(self) -> None:
        logger.info(
            "Agent 已启动: instance_id=%s, server=%s, group=%s, gpu=%s",
            self._instance_id,
            self._server,
            self._group_name,
            self._gpu_name,
        )
        heartbeat_loop = asyncio.create_task(self._heartbeat_loop())
        fetch_task_loop = asyncio.create_task(self._fetch_task_loop())
        try:
            await self._stop.wait()
        finally:
            heartbeat_loop.cancel()
            fetch_task_loop.cancel()
            for t in self._active_tasks:
                t.cancel()
            await asyncio.gather(*self._active_tasks, return_exceptions=True)
            logger.info("Agent 已停止")

    async def _heartbeat_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self._send_heartbeat()
            except Exception as exc:
                logger.warning("[心跳] 失败: %s", exc)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._heartbeat_interval)
            except asyncio.TimeoutError:
                pass

    async def _send_heartbeat(self) -> None:
        payload = {
            "instance_id": self._instance_id,
            "status": self._status,
            "current_task_id": self._current_task_id,
            "group_name": self._group_name,
            "gpu_name": self._gpu_name,
            "agent_version": self._AGENT_VERSION,
        }
        async with httpx.AsyncClient(base_url=self._server, timeout=httpx.Timeout(10.0, connect=5.0)) as c:
            resp = await c.post("/instance/heartbeat", json=payload, headers=_make_auth_headers())
            if resp.status_code != 200:
                logger.warning("[心跳] 异常: %d %s", resp.status_code, resp.text)

    async def _fetch_task_loop(self) -> None:
        while not self._stop.is_set():
            if self._status == "idle":
                try:
                    async with httpx.AsyncClient(base_url=self._server, timeout=httpx.Timeout(10.0)) as c:
                        resp = await c.post("/instance/fetch_task", json={"instance_id": self._instance_id}, headers=_make_auth_headers())
                        if resp.status_code == 200:
                            tasks = resp.json().get("tasks") or []
                            if tasks:
                                self._current_task = tasks[0]
                                self._status = "computing"
                                t = asyncio.create_task(self._execute_comfyui_task())
                                self._active_tasks.append(t)
                except Exception as exc:
                    logger.warning("[轮询] 失败: %s", exc)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._poll_interval)
            except asyncio.TimeoutError:
                pass

    @staticmethod
    def _load_workflow_file(workflow_file: str, params: dict) -> dict:
        """从本地 workflows/ 目录加载 JSON，并将 params 覆盖到匹配的节点 inputs。"""
        wf_path = REPO_ROOT / "workflows" / "json" / workflow_file
        if not wf_path.exists():
            raise FileNotFoundError(f"Workflow file not found: {wf_path}")
        workflow_file_content = json.loads(wf_path.read_text())
        for key, value in params.items():
            for node in workflow_file_content.values():
                if isinstance(node, dict) and key in node.get("inputs", {}):
                    node["inputs"][key] = value
        return workflow_file_content

    async def _execute_comfyui_task(self) -> None:
        task = self._current_task
        if not task:
            return
        task_id: str = task["task_id"]
        self._current_task_id = task_id
        comfyui = ComfyUIClient(ip="localhost", port=18188)
        try:
            logger.info("执行任务: %s", task_id)
            workflow_file: str = task.get("workflow_file", "")
            params: dict = task.get("payload", {})
            if not workflow_file:
                raise ValueError("任务缺少 workflow_file 字段")
            workflow_file_content = self._load_workflow_file(workflow_file, params)
            prompt_id = await comfyui.submit_prompt(workflow_file_content, client_id=f"container-{self._instance_id}")
            if not prompt_id:
                raise ValueError("ComfyUI 未返回 prompt_id")
            msg = await self._poll_history(comfyui, prompt_id)
            result: dict = {"status": msg.status, "prompt_id": prompt_id, "outputs": msg.outputs}
            if msg.error:
                result["error"] = msg.error
            if not await self._push_result(task_id, result):
                await self._push_result(task_id, {"status": "failed", "error": "result push failed"})
        except Exception as exc:
            logger.error("任务 %s 失败: %s", task_id, exc)
            await self._push_result(task_id, {"status": "failed", "error": str(exc)})
        finally:
            self._current_task = None
            self._current_task_id = None
            self._status = "idle"
            self._active_tasks = [t for t in self._active_tasks if not t.done()]

    async def _poll_history(self, comfyui: ComfyUIClient, prompt_id: str, max_attempts: int = 360) -> ComfyUIMessage:
        for attempt in range(max_attempts):
            try:
                msg = await comfyui.get_history(prompt_id)
                if msg.status != "in_progress":
                    return msg
            except Exception as exc:
                logger.debug("[历史轮询] 第%d次异常: %s", attempt + 1, exc)
            if attempt > 0 and attempt % 12 == 0:
                logger.info("[历史轮询] prompt_id=%s 等待中... (%ds)", prompt_id, attempt * 5)
            await asyncio.sleep(5)
        return ComfyUIMessage(prompt_id=prompt_id, status="error", error="ComfyUI 任务执行超时")

    async def _push_result(self, task_id: str, result: dict, max_retries: int = 3) -> bool:
        payload = {
            "task_id": task_id,
            "instance_id": self._instance_id,
            "status": result.get("status", "failed"),
            "result_path": f"/results/{task_id}",
            "error_message": result.get("error"),
        }
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=30.0) as c:
                    resp = await c.post(f"{self._server}/instance/result", json=payload, headers=_make_auth_headers())
                    if resp.status_code == 200:
                        logger.info("任务 %s 结果已推送", task_id)
                        return True
                    logger.warning("推送失败 (%d/%d): %d", attempt + 1, max_retries, resp.status_code)
            except Exception as exc:
                logger.warning("推送异常 (%d/%d): %s", attempt + 1, max_retries, exc)
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** (attempt + 1))
        return False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} <group_name>", file=sys.stderr)
        sys.exit(1)

    group_name = sys.argv[1]

    async def main() -> None:
        # Validate config early. Provisioning itself runs in background.
        try:
            _load_group_setup_config(group_name)
        except FileNotFoundError as exc:
            logger.error("%s", exc)
            sys.exit(1)
        agent = InstanceAgent(
            server_url=os.getenv("SERVER_URL", "http://localhost:8000"),
            heartbeat_interval=int(os.getenv("HEARTBEAT_INTERVAL", "5")),
            group_name=group_name,
            gpu_name=os.getenv("GPU_NAME", ""),
        )
        setup_task = asyncio.create_task(
            provision_in_background(group_name, on_finished=agent.mark_provisioning_done)
        )
        agent.add_background_task(setup_task)
        await agent.start()

    asyncio.run(main())
