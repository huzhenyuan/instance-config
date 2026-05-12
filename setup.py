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
    INSTANCE_API_SECRET — Bearer token for /instance/* API authentication (preferred)
    HEARTBEAT_INTERVAL — seconds between heartbeats (default: 5)
    INSTANCE_GROUP_NAME — group name passed at container launch
    GPU_NAME          — GPU display name passed at container launch
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
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
COMFYUI_ROOT = Path("/workspace/ComfyUI")
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
    """生成 Authorization: Bearer 认证头（未配置时跳过认证）。"""
    secret = os.getenv("INSTANCE_API_SECRET", "")
    if not secret:
        return {}
    return {"Authorization": f"Bearer {secret}"}


# ---------------------------------------------------------------------------
# Phase 1: Node installation
# ---------------------------------------------------------------------------

CM_CLI = Path("/workspace/ComfyUI/custom_nodes/ComfyUI-Manager/cm-cli.py")
VENV_PYTHON = Path("/venv/main/bin/python")


async def install_nodes(nodes: list[dict], custom_nodes_dir: Path) -> None:
    custom_nodes_dir.mkdir(parents=True, exist_ok=True)
    python = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable
    # Ensure setuptools/importlib_metadata are present (required by some node install.py scripts)
    await _run(f'"{python}" -m pip install setuptools importlib_metadata -q', timeout=600)
    for node in nodes:
        repo: str = node.get("repo", "")
        if not repo:
            continue
        name = Path(repo).name.removesuffix(".git")
        if not CM_CLI.exists():
            logger.error("[nodes] cm-cli.py not found at %s; skipping %s", CM_CLI, name)
            continue
        logger.info("[nodes] Installing via cm-cli: %s", repo)
        ok = await _run(
            f'"{python}" "{CM_CLI}" install "{repo}"',
            timeout=600,
        )
        if not ok:
            logger.error("[nodes] cm-cli install failed: %s", repo)

    # Reboot ComfyUI to load newly installed nodes
    logger.info("[nodes] Rebooting ComfyUI to load new nodes...")
    try:
        async with httpx.AsyncClient(timeout=10.0) as c:
            await c.post("http://localhost:18188/api/manager/reboot")
        logger.info("[nodes] ComfyUI reboot triggered")
    except Exception as exc:
        logger.warning("[nodes] ComfyUI reboot request failed (may be normal if not running): %s", exc)


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


def _comfyui_is_ready() -> bool:
    """Return True once ComfyUI source tree has been initialized."""
    required_markers = ("main.py", "server.py", "README.md")
    return COMFYUI_ROOT.is_dir() and any((COMFYUI_ROOT / marker).exists() for marker in required_markers)


async def _wait_for_comfyui_ready(timeout_sec: int = 300, interval_sec: int = 2) -> bool:
    """Avoid creating /workspace/ComfyUI subdirs before image bootstrap finishes."""
    elapsed = 0
    while elapsed <= timeout_sec:
        if _comfyui_is_ready():
            return True
        await asyncio.sleep(interval_sec)
        elapsed += interval_sec
    return False


async def provision_in_background(group_name: str, on_finished: Callable[[], None] | None = None) -> None:
    try:
        if not await _wait_for_comfyui_ready():
            logger.error(
                "[setup] ComfyUI not initialized under %s after timeout; skip provisioning to avoid clobbering bootstrap",
                COMFYUI_ROOT,
            )
            return

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
            if not resp.is_success:
                logger.error("[ComfyUI] submit_prompt failed %d: %s", resp.status_code, resp.text)
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
    _LOG_FILE = "/var/log/gpus-agent.log"
    _RUNTIME_REFRESH_SEC = 15

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
        self._public_ip = os.getenv("PUBLIC_IPADDR", "")
        self._heartbeat_interval = heartbeat_interval
        self._poll_interval = poll_interval
        self._status = "provisioning"
        self._current_task_id: str | None = None
        self._current_task: dict | None = None
        self._runtime_info_cache: dict[str, str] = {}
        self._runtime_info_cache_at: float = 0.0
        self._stop = asyncio.Event()
        self._active_tasks: list[asyncio.Task] = []

    @staticmethod
    def _tail_log_file(path: str, max_lines: int = 200) -> str:
        try:
            proc = subprocess.run(
                ["tail", "-n", str(max_lines), path],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
            out = (proc.stdout or "").strip()
            if out:
                return out

            if proc.stderr:
                return f"[log unavailable] {proc.stderr.strip()}"

            # Fallback: if tail returns empty unexpectedly, keep previous behavior.
            with open(path, encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            return "".join(lines[-max_lines:]).strip()
        except Exception as exc:
            return f"[log unavailable] {exc}"

    @staticmethod
    async def _run_capture(cmd: list[str], timeout: float = 5.0) -> str:
        def _run_sync() -> str:
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
                out = (proc.stdout or "").strip()
                err = (proc.stderr or "").strip()
                return out or err
            except Exception as exc:
                return f"[command failed: {' '.join(cmd)}] {exc}"

        return await asyncio.to_thread(_run_sync)

    async def _collect_runtime_info(self) -> dict[str, str]:
        now = asyncio.get_running_loop().time()
        if self._runtime_info_cache and (now - self._runtime_info_cache_at) < self._RUNTIME_REFRESH_SEC:
            return self._runtime_info_cache

        top_out = await self._run_capture(["top", "-bn1"], timeout=6.0)
        mem_out = await self._run_capture(["free", "-h"], timeout=4.0)

        runtime_info = {
            "agent_log_tail": self._tail_log_file(self._LOG_FILE, 200),
            "disk_free": await self._run_capture(["df", "-h"], timeout=5.0),
            "cpu_usage": top_out or "[cpu unavailable]",
            "memory_usage": mem_out or "[memory unavailable]",
            "gpu_info": await self._run_capture(["nvidia-smi"], timeout=8.0),
        }

        self._runtime_info_cache = runtime_info
        self._runtime_info_cache_at = now
        return runtime_info

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
        runtime_info = await self._collect_runtime_info()
        payload = {
            "instance_id": self._instance_id,
            "status": self._status,
            "current_task_id": self._current_task_id,
            "group_name": self._group_name,
            "gpu_name": self._gpu_name,
            "ip_address": self._public_ip,
            "agent_version": self._AGENT_VERSION,
            "runtime_info": runtime_info,
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
    def _build_endpoint_map() -> dict[str, str]:
        """Scan all workflow YAMLs and build endpoint -> workflow_file mapping.

        Each workflow YAML may declare an ``endpoint`` field that links it to a
        gateway API endpoint.  This replaces the server-side workflows_mapping
        config: the instance resolves the mapping locally from instance-config.
        """
        mapping: dict[str, str] = {}
        yaml_dir = REPO_ROOT / "workflows"
        for yaml_path in yaml_dir.glob("*.yaml"):
            if yaml_path.name.startswith("_"):
                continue
            try:
                with yaml_path.open() as f:
                    cfg = yaml.safe_load(f) or {}
                endpoint: str = cfg.get("endpoint", "")
                workflow_id: str = cfg.get("workflow_id", "")
                if endpoint and workflow_id:
                    mapping[endpoint] = f"{workflow_id}.json"
            except Exception:
                pass
        return mapping

    @staticmethod
    async def _prepare_workflow(workflow_file: str, params: dict, task_id: str) -> dict:
        """Load workflow JSON and apply bindings from the companion YAML.

        Each binding resolves its ``source`` field:
          request.<key>  → value from the task params dict
          runtime.<var>  → auto-generated variable (save_prefix, seed)

        A binding may carry an optional ``type`` field:
          url_image / url_video → download the URL and substitute the local
                                  relative path into the node input

        Auto-generated runtime vars:
          save_prefix → "{workflow_id}/{task_id}"
          seed        → random 63-bit integer
        """
        wf_path = REPO_ROOT / "workflows" / "json" / workflow_file
        if not wf_path.exists():
            raise FileNotFoundError(f"Workflow file not found: {wf_path}")
        graph = json.loads(wf_path.read_text())

        # --- Load companion YAML (optional) ----------------------------------
        workflow_id = workflow_file.removesuffix(".json")
        yaml_path = REPO_ROOT / "workflows" / f"{workflow_id}.yaml"
        bindings: list[dict] = []
        comfy_input_root = Path("/workspace/ComfyUI/input")
        yaml_workflow_id = workflow_id

        if yaml_path.exists():
            with yaml_path.open() as f:
                wf_cfg = yaml.safe_load(f) or {}
            bindings = wf_cfg.get("bindings") or []
            comfy_input_root = Path(wf_cfg.get("comfy_input_root", str(comfy_input_root)))
            yaml_workflow_id = wf_cfg.get("workflow_id", workflow_id)

        input_dir = comfy_input_root
        input_dir.mkdir(parents=True, exist_ok=True)

        # Auto-generated runtime vars.
        runtime: dict[str, Any] = {
            "save_prefix": f"{yaml_workflow_id}/{task_id}",
            "seed": int.from_bytes(__import__("os").urandom(8), "big") & 0x7FFF_FFFF_FFFF_FFFF,
        }

        # --- Apply bindings to graph ----------------------------------------
        for binding in bindings:
            node_id: str = str(binding.get("node", ""))
            input_key: str = str(binding.get("input", ""))
            source: str = str(binding.get("source", ""))
            binding_type: str = str(binding.get("type", ""))
            default = binding.get("default")

            if not node_id or not input_key or not source:
                continue
            node = graph.get(node_id)
            if not isinstance(node, dict):
                continue

            # Resolve raw value from source.
            value: Any = None
            if source.startswith("request."):
                value = params.get(source[len("request."):], default)
            elif source.startswith("runtime."):
                value = runtime.get(source[len("runtime."):], default)
            else:
                value = default

            if value is None:
                continue

            # For URL inputs, download and replace value with local relpath.
            if binding_type in ("url_image", "url_video"):
                url = str(value)
                param_key = source[len("request."):] if source.startswith("request.") else input_key
                ext = Path(url.split("?")[0]).suffix or (".mp4" if binding_type == "url_video" else ".jpg")
                local_path = input_dir / f"{task_id}_{param_key}{ext}"
                if not local_path.exists():
                    logger.info("[binding] Downloading %s → %s", url, local_path)
                    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
                        resp = await client.get(url)
                        resp.raise_for_status()
                        local_path.write_bytes(resp.content)
                value = str(local_path.relative_to(comfy_input_root))
                logger.info("[binding] node %s .inputs.%s = %s", node_id, input_key, value)

            inputs_dict = node.get("inputs")
            if isinstance(inputs_dict, dict):
                inputs_dict[input_key] = value
                logger.debug("[binding] node %s .inputs.%s = %r", node_id, input_key, value)

        return graph

    async def _execute_comfyui_task(self) -> None:
        task = self._current_task
        if not task:
            return
        task_id: str = task["task_id"]
        self._current_task_id = task_id
        comfyui = ComfyUIClient(ip="localhost", port=18188)
        try:
            logger.info("执行任务: %s", task_id)
            endpoint: str = task.get("endpoint", "")
            params: dict = task.get("payload", {})
            if not endpoint:
                raise ValueError("任务缺少 endpoint 字段")
            # Resolve endpoint to workflow file using local instance-config
            endpoint_map = self._build_endpoint_map()
            workflow_file = endpoint_map.get(endpoint, "")
            if not workflow_file:
                raise ValueError(f"未知端点，无法映射到工作流: {endpoint}")
            workflow_file_content = await self._prepare_workflow(workflow_file, params, task_id)
            # print workflow_file_content for current task in debug log
            logger.info("Prepared workflow for task %s (endpoint=%s): %s", task_id, endpoint, json.dumps(workflow_file_content, indent=2))
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
        # Upload output file(s) to server before pushing result metadata
        images: list[dict] = result.get("outputs", {}).get("images", [])
        server_result_path: str | None = None
        if result.get("status") == "completed" and images:
            comfy_output_root = Path("/workspace/ComfyUI/output")
            files_to_upload: list[Path] = []
            for img in images:
                subfolder = img.get("subfolder", "")
                filename = img.get("filename", "")
                if filename:
                    p = comfy_output_root / subfolder / filename if subfolder else comfy_output_root / filename
                    if p.exists():
                        files_to_upload.append(p)
            if files_to_upload:
                try:
                    async with httpx.AsyncClient(timeout=120.0) as c:
                        upload_files = [
                            ("file", (fp.name, fp.read_bytes(), "image/png" if fp.suffix == ".png" else "video/mp4"))
                            for fp in files_to_upload
                        ]
                        resp = await c.post(
                            f"{self._server}/instance/result_file/{task_id}",
                            files=upload_files,
                            headers=_make_auth_headers(),
                        )
                        if resp.status_code == 200:
                            server_result_path = resp.json().get("primary")
                            logger.info("[upload] 任务 %s 文件已上传: %s", task_id, server_result_path)
                        else:
                            logger.warning("[upload] 上传失败 %d: %s", resp.status_code, resp.text)
                except Exception as exc:
                    logger.warning("[upload] 上传异常: %s", exc)

        payload = {
            "task_id": task_id,
            "instance_id": self._instance_id,
            "status": result.get("status", "failed"),
            "result_path": server_result_path or f"/results/{task_id}",
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
