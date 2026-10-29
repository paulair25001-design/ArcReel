"""Native version selection shared by HTTP and agent hosts."""

import asyncio
import logging
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from lib.artifacts.artifact_activation import (
    forget_current_resource_artifact,
    forget_unbound_grid_artifacts,
    forget_unbound_storyboard_artifacts,
    register_current_resource_artifact,
)
from lib.artifacts.artifact_version_provenance import parse_image_version_basis
from lib.artifacts.formal_write import project_metadata_lock
from lib.artifacts.version_manager import MANUAL_UPLOAD_VERSION_SOURCE, VersionManager
from lib.infra.api_errors import BadRequestError
from lib.infra.async_thread import run_sync_transaction
from lib.project.asset_derivatives import DerivativeSheetTarget, split_derivative_artifact_id
from lib.project.project_change_hints import project_change_source
from lib.project.project_manager import ProjectManager, get_project_manager
from lib.project.resource_paths import CHARACTER_DERIVATIVE_RESOURCE_TYPE
from lib.script.grid.grid_access import ensure_grid_writable
from server.services.currency.artifact_version_restore import (
    is_typed_media_restore_resource,
)
from server.services.currency.typed_media_restore import resolve_media_file, restore_typed_media_resource
from server.services.tasks.derivative_sheet_tasks import point_derivative_at_sheet

# 经此路由可还原的资源类型（API 面策略）。路径形状委托 lib.project.resource_paths，本路由
# 仅放行有还原后元数据同步分支的这几类。grids 的还原只换回联合图文件并复位宫格记录的
# 切分态，不触发切分、不碰任何分镜图——落格由宫格切分端点显式执行。
_RESTORABLE_RESOURCE_TYPES = frozenset(
    {
        "storyboards",
        "videos",
        "audio",
        "characters",
        "scenes",
        "props",
        "products",
        "reference_videos",
        "grids",
        CHARACTER_DERIVATIVE_RESOURCE_TYPE,
    }
)


# resource_type（复数，URL 段）→ asset_type（单数，ASSET_SPECS 键）
_RESOURCE_TO_ASSET_TYPE: dict[str, str] = {
    "characters": "character",
    "scenes": "scene",
    "props": "prop",
    "products": "product",
}
# 资产图、衍生资产图与分镜图：还原到手动上传记录时按上传字节重新登记。
_UPLOAD_CLAIMED_RESOURCE_TYPES = frozenset(
    {*_RESOURCE_TO_ASSET_TYPE, CHARACTER_DERIVATIVE_RESOURCE_TYPE, "storyboards"}
)


def resolve_resource_path(
    resource_type: str,
    resource_id: str,
    project_path: Path,
) -> tuple[Path, str]:
    """返回 (current_file_absolute, relative_file_path)；资源类型不可还原或 ID 越界时抛出领域异常。"""
    if resource_type not in _RESTORABLE_RESOURCE_TYPES:
        raise BadRequestError("unsupported_resource_type", resource_type=resource_type)
    return resolve_media_file(resource_type, resource_id, project_path)


def _sync_grid_record(
    project_path: Path,
    resource_id: str,
    *,
    on_commit: Callable[[], None] | None = None,
    on_miss: Callable[[], None] | None = None,
) -> None:
    """还原联合图后复位宫格记录：内容已换回历史版本，旧的落格结果不再对应当前图。

    只动宫格记录自身（split_at / 失败态），不同步剧本或分镜——落格由切分端点显式执行；
    frame_chain 原样保留。记录缺失/损坏时跳过：联合图文件已还原成功，记录属 best-effort。

    还原本身不设在途闸门（与分镜图还原同口径），复位口径见
    ``GridGeneration.mark_composite_replaced``：生成在途时保留在途态，否则记录会谎报空闲。
    """
    from lib.script.grid.grid_manager import GridManager

    manager = GridManager(project_path)
    manager.update_formal(
        resource_id,
        lambda grid: grid.mark_composite_replaced(),
        on_commit=on_commit,
        on_miss=on_miss,
        ignore_invalid=True,
    )


def _commit_non_typed_restore_claim(
    *,
    resource_type: str,
    resource_id: str,
    file_path: str,
    project_path: Path,
    record: Mapping[str, Any] | None,
    owner_present: bool,
) -> None:
    """Apply the claim half of a non-typed restore inside its metadata locks."""

    if not owner_present:
        if resource_type == "storyboards":
            forget_unbound_storyboard_artifacts(project_path, resource_id)
        elif resource_type == "grids":
            forget_unbound_grid_artifacts(project_path, resource_id)
        else:
            forget_current_resource_artifact(
                project_path,
                resource_type=resource_type,
                resource_id=resource_id,
            )
        return

    if resource_type in _UPLOAD_CLAIMED_RESOURCE_TYPES and (record or {}).get("source") == MANUAL_UPLOAD_VERSION_SOURCE:
        # 还原到作为成品带入的资产图、衍生资产图（作者上传或从资产库应用）或作者上传的分镜图：
        # 选中的已是这条手动上传记录，规划器按上传字节投影依据，投影不出即遗忘。
        register_current_resource_artifact(
            project_path,
            resource_type=resource_type,
            resource_id=resource_id,
        )
        return

    try:
        basis = parse_image_version_basis(resource_type, resource_id, record or {})
    except (TypeError, ValueError):
        basis = None
    if basis is None:
        forget_current_resource_artifact(
            project_path,
            resource_type=resource_type,
            resource_id=resource_id,
        )
        return
    register_current_resource_artifact(
        project_path,
        resource_type=resource_type,
        resource_id=resource_id,
        artifact_path=file_path,
        basis=basis,
    )


def restore_image_resource_version_sync(
    *,
    versions: VersionManager,
    resource_type: str,
    project_name: str,
    resource_id: str,
    version: int,
    current_file: Path,
    file_path: str,
    project_path: Path,
    project_manager: ProjectManager | None = None,
) -> dict[str, Any]:
    """Restore metadata → version bytes/pointer → claim in canonical lock order.

    Asset rename, grid split, formal generation, and typed restore all acquire
    script/file/project locks before the versions lock. Non-typed restore enters
    the same metadata transaction first so none of those paths can form an ABBA
    cycle while their nested rollback scopes remain intact.
    """

    project_manager = project_manager or get_project_manager()
    restored: list[dict[str, Any]] = []

    def _restore(*, owner_present: bool) -> None:
        def _commit_claim(record: dict[str, Any]) -> None:
            _commit_non_typed_restore_claim(
                resource_type=resource_type,
                resource_id=resource_id,
                file_path=file_path,
                project_path=project_path,
                record=record,
                owner_present=owner_present,
            )

        restored.append(
            versions.restore_version(
                resource_type=resource_type,
                resource_id=resource_id,
                version=version,
                current_file=current_file,
                on_restore=_commit_claim,
            )
        )

    if resource_type == "storyboards":
        scripts_dir = project_path / "scripts"
        script_names = [path.name for path in sorted(scripts_dir.glob("*.json"))] if scripts_dir.exists() else []
        with project_change_source("webui"):
            project_manager.update_scene_asset_across_scripts(
                project_name,
                script_names,
                resource_id,
                "storyboard_image",
                file_path,
                on_commit=lambda: _restore(owner_present=True),
                on_miss=lambda: _restore(owner_present=False),
            )
    elif resource_type == CHARACTER_DERIVATIVE_RESOURCE_TYPE:
        owner_key, derivative_key = split_derivative_artifact_id(resource_id)
        try:
            with project_change_source("webui"):
                point_derivative_at_sheet(
                    project_name=project_name,
                    target=DerivativeSheetTarget(owner_key=owner_key, derivative_key=derivative_key),
                    on_commit=lambda _project_file: _restore(owner_present=True),
                    project_manager=project_manager,
                )
        except KeyError:
            with project_metadata_lock(project_path):
                _restore(owner_present=False)
    elif (asset_type := _RESOURCE_TO_ASSET_TYPE.get(resource_type)) is not None:
        try:
            with project_change_source("webui"):
                project_manager._update_asset_sheet(
                    asset_type,
                    project_name,
                    resource_id,
                    file_path,
                    on_commit=lambda _project_file: _restore(owner_present=True),
                )
        except KeyError:
            with project_metadata_lock(project_path):
                _restore(owner_present=False)
    elif resource_type == "grids":
        _sync_grid_record(
            project_path,
            resource_id,
            on_commit=lambda: _restore(owner_present=True),
            on_miss=lambda: _restore(owner_present=False),
        )
    else:
        with project_metadata_lock(project_path):
            _restore(owner_present=True)

    if len(restored) != 1:
        raise RuntimeError("non-typed restore metadata transaction skipped version selection")
    return restored[0]


async def restore_resource_version(
    project_name: str,
    resource_type: str,
    resource_id: str,
    version: int,
    *,
    project_manager: ProjectManager | None = None,
    versions: VersionManager | None = None,
) -> dict[str, Any]:
    """版本还原的实现。资源 id 带层级（角色衍生）的路由不能走单段路径参数，共用此函数。"""
    try:
        project_manager = project_manager or await asyncio.to_thread(get_project_manager)
        if versions is None:
            project_path = await asyncio.to_thread(project_manager.get_project_path, project_name)
            versions = VersionManager(project_path)
        if is_typed_media_restore_resource(resource_type):
            with project_change_source("webui"):
                return await restore_typed_media_resource(
                    project_manager=project_manager,
                    versions=versions,
                    project_name=project_name,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    version=version,
                )

        def _sync():
            # 还原同样是一条联合图写入路径（换回历史联合图 + 复位宫格记录），
            # 与重生成/切分/上传共用准入判定；漏掉这里，被封禁项目就能从还原绕过。
            if resource_type == "grids":
                ensure_grid_writable(project_manager.load_project(project_name))

            vm = versions
            project_path = project_manager.get_project_path(project_name)
            current_file, file_path = resolve_resource_path(resource_type, resource_id, project_path)

            result = restore_image_resource_version_sync(
                versions=vm,
                resource_type=resource_type,
                project_name=project_name,
                resource_id=resource_id,
                version=version,
                current_file=current_file,
                file_path=file_path,
                project_path=project_path,
                project_manager=project_manager,
            )

            # 计算还原后文件的 fingerprint
            asset_fingerprints: dict[str, int] = {}
            if current_file.exists():
                asset_fingerprints[file_path] = current_file.stat().st_mtime_ns

            return {
                "success": True,
                **result,
                "file_path": file_path,
                "asset_fingerprints": asset_fingerprints,
            }

        return await run_sync_transaction(_sync)

    except ValueError as e:
        # 非法项目名 / 越界 resource_id 等坏请求，str(e) 可能含路径只进日志
        logger.warning("版本还原请求非法: %s", e)
        raise BadRequestError("request_invalid") from e
