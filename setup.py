#!/usr/bin/env python3
"""Instance setup + agent runner.

Usage:
    python3 setup.py <group_id>

Phase 1 – Provision: installs custom ComfyUI nodes and downloads model files.
Phase 2 – Agent loop: registers with the scheduler server, then heartbeats and
           executes ComfyUI tasks until the process is terminated.

Environment variables:
    CUSTOM_NODES_DIR  — default: /workspaces/ComfyUI/custom_nodes
    MODELS_DIR        — default: /workspaces/ComfyUI/models
    SERVER_URL        — scheduler server base URL (default: http://localhost:8000)
    CONTAINER_ID      — instance identifier
    PUBLIC_IPADDR     — public IP reported during registration
    API_SECRET        — HMAC secret for request signing (optional)
    HEARTBEAT_INTERVAL — seconds between heartbeats (default: 30)
    INSTANCE_GROUP_ID — group identifier passed at container launch
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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

    def __init__(self, server_url: str, heartbeat_interval: int = 30, poll_interval: int = 5, group_id: str = ""):
        self._server = server_url.rstrip("/")
        self._instance_id = os.getenv("CONTAINER_ID", "")
        self._group = group_id
        self._heartbeat_interval = heartbeat_interval
        self._poll_interval = poll_interval
        self._status = "idle"
        self._current_task_id: str | None = None
        self._current_task: dict | None = None
        self._stop = asyncio.Event()
        self._active_tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        if not await self._register():
            logger.error("无法注册到服务端，agent 退出")
            return
        logger.info("Agent 已启动: instance_id=%s, server=%s", self._instance_id, self._server)
        hb = asyncio.create_task(self._heartbeat_loop())
        poll = asyncio.create_task(self._poll_loop())
        try:
            await self._stop.wait()
        finally:
            hb.cancel()
            poll.cancel()
            for t in self._active_tasks:
                t.cancel()
            logger.info("Agent 已停止")

    async def _register(self) -> bool:
        payload = {
            "instance_id": self._instance_id,
            "ip_address": os.getenv("PUBLIC_IPADDR", ""),
            "agent_version": self._AGENT_VERSION,
            "group_id": self._group,
        }
        try:
            async with httpx.AsyncClient(base_url=self._server, timeout=httpx.Timeout(10.0, connect=5.0)) as c:
                resp = await c.post("/instance/register", json=payload, headers=_make_auth_headers())
                if resp.status_code == 200:
                    return True
                logger.error("注册失败: %d %s", resp.status_code, resp.text)
        except Exception as exc:
            logger.error("注册异常: %s", exc)
        return False

    async def _heartbeat_loop(self) -> None:
        while not self._stop.is_set():
            try:
                async with httpx.AsyncClient(base_url=self._server, timeout=httpx.Timeout(10.0)) as c:
                    resp = await c.post("/instance/health", headers=_make_auth_headers(), json={
                        "instance_id": self._instance_id,
                        "status": self._status,
                        "current_task_id": self._current_task_id,
                    })
                    if resp.status_code != 200:
                        logger.warning("[心跳] 异常: %d", resp.status_code)
            except Exception as exc:
                logger.warning("[心跳] 失败: %s", exc)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._heartbeat_interval)
            except asyncio.TimeoutError:
                pass

    async def _poll_loop(self) -> None:
        while not self._stop.is_set():
            if self._status == "idle" and not self._current_task:
                try:
                    async with httpx.AsyncClient(base_url=self._server, timeout=httpx.Timeout(10.0)) as c:
                        resp = await c.post("/instance/fetch_task", json={"instance_id": self._instance_id}, headers=_make_auth_headers())
                        if resp.status_code == 200:
                            tasks = resp.json().get("tasks") or []
                            if tasks:
                                self._current_task = tasks[0]
                                self._status = "computing"
                                t = asyncio.create_task(self._execute_task())
                                self._active_tasks.append(t)
                except Exception as exc:
                    logger.warning("[轮询] 失败: %s", exc)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._poll_interval)
            except asyncio.TimeoutError:
                pass

    @staticmethod
    def _load_graph(workflow_file: str, params: dict) -> dict:
        """从本地 workflows/ 目录加载 JSON，并将 params 覆盖到匹配的节点 inputs。"""
        wf_path = REPO_ROOT / "workflows" / workflow_file
        if not wf_path.exists():
            raise FileNotFoundError(f"Workflow file not found: {wf_path}")
        graph = json.loads(wf_path.read_text())
        graph = copy.deepcopy(graph)
        for key, value in params.items():
            for node in graph.values():
                if isinstance(node, dict) and key in node.get("inputs", {}):
                    node["inputs"][key] = value
        return graph

    async def _execute_task(self) -> None:
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
            graph = self._load_graph(workflow_file, params)
            prompt_id = await comfyui.submit_prompt(graph, client_id=f"container-{self._instance_id}")
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
        print(f"Usage: python3 {sys.argv[0]} <group_id>", file=sys.stderr)
        sys.exit(1)

    group_id = sys.argv[1]

    async def main() -> None:
        await setup(group_id)
        agent = InstanceAgent(
            server_url=os.getenv("SERVER_URL", "http://localhost:8000"),
            heartbeat_interval=int(os.getenv("HEARTBEAT_INTERVAL", "30")),
            group_id=group_id,
        )
        await agent.start()

    asyncio.run(main())
