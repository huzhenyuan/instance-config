"""实例和任务的状态枚举定义。

所有状态字符串集中于此，避免各模块散乱的魔法字符串。
与服务端 src/gpu_scheduler/status.py 保持一致。
"""

from __future__ import annotations

from enum import Enum


class InstanceStatus(str, Enum):
    """实例生命周期状态。

    启动流程: provisioning → idle
    工作流程: idle ↔ busy/computing
    缩容流程: idle → draining → stopping → stopped
    """

    # --- 启动阶段 ---
    PROVISIONING = "provisioning"   # VastAI 容器刚创建，还未就绪

    # --- 正常运行阶段 ---
    IDLE = "idle"                   # 空闲，可接任务
    BUSY = "busy"                   # 任务已派发，处理中
    COMPUTING = "computing"         # 计算中（实例自报）

    # --- 缩容阶段 ---
    DRAINING = "draining"           # 标记为排水，不再接新任务
    STOPPING = "stopping"           # 正在执行终止操作
    STOPPED = "stopped"             # 已停止

    @property
    def can_fetch_task(self) -> bool:
        """只有 idle 状态的实例才允许领取新任务。"""
        return self == InstanceStatus.IDLE


class TaskStatus(str, Enum):
    """任务生命周期状态。

    正常流程: queued → dispatched → running → completed
    失败流程: running/dispatched → failed
    """

    QUEUED = "queued"          # 等待被实例领取
    DISPATCHED = "dispatched"  # 已派发给实例，等待确认
    RUNNING = "running"        # 实例正在执行
    COMPLETED = "completed"    # 执行成功
    FAILED = "failed"          # 失败（含超时）
