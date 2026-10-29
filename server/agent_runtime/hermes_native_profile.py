"""Expose the embedded ArcReel profile to Hermes without duplicating its workflow."""

from __future__ import annotations

import hashlib
import json
import os
import posixpath
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

import yaml

from lib.agent.agent_profile import agent_profile_dir
from lib.agent.profile_manifest import ContentMode, resolve_profile_files_for_mode

DEFAULT_READ_PROFILE_TOOL = "read_agent_profile"
DEFAULT_DISPATCH_TOOL = "dispatch_agent"
_NAME = re.compile(r"[a-z0-9][a-z0-9-]*")
_READ_TOOLS = (
    "mcp__arcreel__get_project_content",
    "mcp__arcreel__list_source_files",
    "mcp__arcreel__get_source_text",
    "mcp__arcreel__get_episode_script",
    "mcp__arcreel__get_script_plan_content",
    "mcp__arcreel__list_project_files",
    "mcp__arcreel__read_project_file",
)


def _frontmatter(content: str) -> dict[str, Any]:
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    try:
        end = next(index for index in range(1, len(lines)) if lines[index].strip() == "---")
    except StopIteration as error:
        raise ValueError("Native profile frontmatter is not terminated") from error
    metadata = yaml.safe_load("\n".join(lines[1:end]))
    if metadata is None:
        return {}
    if not isinstance(metadata, dict) or any(not isinstance(key, str) for key in metadata):
        raise ValueError("Native profile frontmatter must be a mapping")
    return metadata


def _names(value: object, *, comma_separated: bool = False) -> tuple[str, ...]:
    if value is None:
        return ()
    if comma_separated and isinstance(value, str):
        values = [part.strip() for part in value.split(",")]
    elif isinstance(value, list):
        values = value
    else:
        raise ValueError("Native profile tools and skills must be name lists")
    if any(not isinstance(name, str) or not name.strip() for name in values):
        raise ValueError("Native profile tool or skill name is invalid")
    return tuple(dict.fromkeys(values))


def _is_document(path: str) -> bool:
    return path == "CLAUDE.md" or (
        path.endswith(".md") and path.startswith((".claude/skills/", ".claude/references/", ".claude/agents/"))
    )


def _read_document_bytes(root: Path, source_path: str) -> bytes:
    path = root / source_path
    if not path.is_relative_to(root) or any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError("Native profile document path must not contain symlinks")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as stream:
        opened, current = os.fstat(stream.fileno()), path.lstat()
        if not stat.S_ISREG(current.st_mode) or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino):
            raise ValueError("Native profile document identity changed while reading")
        return stream.read()


@dataclass(frozen=True)
class NativeProfileDocument:
    path: str
    source_path: str
    source_sha256: str
    content: str

    @property
    def metadata(self) -> dict[str, Any]:
        return _frontmatter(self.content)

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "content": self.content,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class NativeAgentSpec:
    name: str
    document: NativeProfileDocument
    skills: tuple[NativeProfileDocument, ...]
    # None inherits the parent project's approved MCP surface; an explicit list
    # remains a restriction, so review roles cannot acquire generation tools.
    allowed_tools: tuple[str, ...] | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "agent": self.document.as_dict(),
            "skills": [skill.as_dict() for skill in self.skills],
            "allowed_tools": list(self.allowed_tools) if self.allowed_tools is not None else None,
        }


@dataclass(frozen=True)
class HermesNativeProfile:
    content_mode: ContentMode
    documents: Mapping[str, NativeProfileDocument]
    source_sha256: str

    def read_document(self, path: str, *, relative_to: str | None = None) -> NativeProfileDocument:
        """Resolve only an enumerated current-mode Markdown document or its link."""
        if not path or "\\" in path or PurePosixPath(path).is_absolute():
            raise ValueError("Native profile document path must be relative")
        if relative_to is not None and not path.startswith(".claude/") and path != "CLAUDE.md":
            if relative_to not in self.documents:
                raise ValueError("Native profile link origin is unavailable")
            path = posixpath.join(posixpath.dirname(relative_to), path)
        logical = posixpath.normpath(path)
        if logical.startswith("../") or logical not in self.documents:
            raise ValueError("Native profile document is not available in this content mode")
        return self.documents[logical]

    def read_skill(self, name: str) -> NativeProfileDocument:
        if _NAME.fullmatch(name) is None:
            raise ValueError("Native skill name is invalid")
        return self.read_document(f".claude/skills/{name}/SKILL.md")

    def read_agent(self, name: str) -> NativeProfileDocument:
        if _NAME.fullmatch(name) is None:
            raise ValueError("Native agent name is invalid")
        return self.read_document(f".claude/agents/{name}.md")

    @property
    def agent_names(self) -> tuple[str, ...]:
        return tuple(
            PurePosixPath(path).stem
            for path in self.documents
            if path.startswith(".claude/agents/") and PurePosixPath(path).parent.as_posix() == ".claude/agents"
        )

    def catalog(self) -> list[dict[str, str]]:
        """List all definitions without injecting their bodies into main context."""
        return [
            {
                "path": document.path,
                "source_path": document.source_path,
                "source_sha256": document.source_sha256,
                "description": str(document.metadata.get("description") or ""),
            }
            for document in self.documents.values()
        ]

    def agent_spec(self, name: str, *, read_profile_tool: str = DEFAULT_READ_PROFILE_TOOL) -> NativeAgentSpec:
        document = self.read_agent(name)
        metadata = document.metadata
        if metadata.get("name", name) != name:
            raise ValueError("Native agent name differs from its profile document")
        tools: tuple[str, ...] | None = None
        if "tools" in metadata:
            approved: list[str] = [read_profile_tool]
            for tool in _names(metadata["tools"], comma_separated=True):
                if tool in {"Read", "Glob", "Grep"}:
                    approved.extend(_READ_TOOLS)
                    if tool == "Read":
                        approved.append("read_file")
                elif tool.startswith("mcp__arcreel__"):
                    approved.append(tool)
                else:
                    raise ValueError(f"Native agent tool has no approved Hermes adapter: {tool}")
            tools = tuple(dict.fromkeys(approved))
        skills = tuple(self.read_skill(skill) for skill in _names(metadata.get("skills")))
        return NativeAgentSpec(name=name, document=document, skills=skills, allowed_tools=tools)

    def to_manifest(
        self,
        project_name: str,
        *,
        read_profile_tool: str = DEFAULT_READ_PROFILE_TOOL,
        dispatch_tool: str = DEFAULT_DISPATCH_TOOL,
    ) -> dict[str, Any]:
        """Serialize a source-bound snapshot for the dependency-isolated child."""
        agents: dict[str, Any] = {}
        for name in self.agent_names:
            spec = self.agent_spec(name, read_profile_tool=read_profile_tool)
            metadata = spec.document.metadata
            agents[name] = {
                "name": name,
                "description": str(metadata.get("description") or ""),
                "path": spec.document.path,
                "source_sha256": spec.document.source_sha256,
                "prompt": self.agent_prompt(
                    name, project_name, read_profile_tool=read_profile_tool, dispatch_tool=dispatch_tool
                ),
                "tools": list(spec.allowed_tools) if spec.allowed_tools is not None else None,
                "native_tools": list(_names(metadata.get("tools"), comma_separated=True)),
                "skills": list(_names(metadata.get("skills"))),
            }
        return {
            "schema": "arcreel-hermes-native-profile/v1",
            "project": project_name,
            "content_mode": self.content_mode,
            "source_sha256": self.source_sha256,
            "read_profile_tool": read_profile_tool,
            "dispatch_tool": dispatch_tool,
            "documents": {path: document.as_dict() for path, document in self.documents.items()},
            "agents": agents,
        }

    def bootstrap_prompt(
        self,
        project_name: str,
        *,
        read_profile_tool: str = DEFAULT_READ_PROFILE_TOOL,
        dispatch_tool: str = DEFAULT_DISPATCH_TOOL,
    ) -> str:
        claude = self.read_document("CLAUDE.md")
        catalog = json.dumps(self.catalog(), ensure_ascii=False, separators=(",", ":"))
        return (
            f"[ArcReel 原生配置 mode={self.content_mode}; source_sha256={self.source_sha256}]\n"
            + claude.content
            + "\n\n"
            + self.host_instructions(project_name, read_profile_tool=read_profile_tool, dispatch_tool=dispatch_tool)
            + "\n\n## 原生配置文档目录\n"
            + catalog
            + f"\n开始工作流前，先用 {read_profile_tool} 读取 `.claude/skills/video-workflow/SKILL.md`。"
            "按原文链接和当前动作读取对应 skill、reference、agent；目录不是这些文档的替代摘要。"
        )

    def agent_prompt(
        self,
        name: str,
        project_name: str,
        *,
        read_profile_tool: str = DEFAULT_READ_PROFILE_TOOL,
        dispatch_tool: str = DEFAULT_DISPATCH_TOOL,
    ) -> str:
        spec = self.agent_spec(name, read_profile_tool=read_profile_tool)
        documents = (spec.document, *spec.skills)
        return "\n\n".join(
            [
                self.read_document("CLAUDE.md").content,
                self.host_instructions(project_name, read_profile_tool=read_profile_tool, dispatch_tool=dispatch_tool),
                "当前是原生子智能体：只完成父会话给出的聚焦目标，完成即返回；不能再派发子智能体。",
                *[
                    f"[原生文档 {document.path}; SHA256={document.source_sha256}]\n{document.content}"
                    for document in documents
                ],
            ]
        )

    @staticmethod
    def host_instructions(
        project_name: str,
        *,
        read_profile_tool: str = DEFAULT_READ_PROFILE_TOOL,
        dispatch_tool: str = DEFAULT_DISPATCH_TOOL,
    ) -> str:
        """Map host affordances while leaving creative and review text unchanged."""
        return (
            "## Hermes 宿主工具适配\n"
            f"当前项目仅为 {json.dumps(project_name, ensure_ascii=False)}；所有业务 MCP 请求显式传入该 project。\n"
            "以上及按需读取的文档是 ArcReel 自带原文。只将其中宿主工具名按下列对应关系解释，"
            "其工作流顺序、质量标准、审图审片、修复和授权要求保持原意：\n"
            f"- Skill、读取 CLAUDE.md 或 `.claude/` 文档：调用 {read_profile_tool}，path 取目录中的逻辑路径；"
            "相对链接附上当前文档路径作为 relative_to。不得用项目业务文件工具读取宿主文件。\n"
            "- Read 项目内容：project.json 用 mcp__arcreel__get_project_content，正式集脚本用 "
            "mcp__arcreel__get_episode_script，源文用 mcp__arcreel__get_source_text，脚本规划用 "
            "mcp__arcreel__get_script_plan_content；待修草稿用 mcp__arcreel__open_draft；"
            "其他允许的业务文本用 mcp__arcreel__read_project_file。\n"
            "- Glob/Grep：用 mcp__arcreel__list_project_files 或 list_source_files 定位，"
            "再经上述读取工具取得内容；不启用通用文件系统工具。\n"
            "- read_file：仅分页读取当前 ArcReel 隔离会话 home/cache/spillover/ 下的工具溢出缓存，"
            "按原生 offset/limit 继续读取；它不提供通用文件访问。项目数据仍用业务 MCP，"
            f"原生配置仍用 {read_profile_tool}，不能借缓存读取工具访问其他 home、项目、配置或源代码。\n"
            "- 长任务：此宿主的业务 MCP 可能立即返回 generation_batch 或 accepted/queued/running task，"
            "这只是已受理。保留 batch/task 身份回报父会话并结束本次子任务，等待由 ArcReel 原生后台 driver 完成；"
            "不让子代理用模型反复轮询，不把受理或本轮结束说成媒体制作完成。\n"
            "- 本地能力：实际可用模式、时长、尺寸和参考图限制以 get_video_capabilities 为准，优先于原文示例；"
            "TTS 不支持 numeric speed 参数时使用原生自然语言演法指令，不能伪造参数、改用云端或付费语音。\n"
            f"- Task/dispatch：调用 {dispatch_tool}，agent_type 取原生 agent 名，prompt 传原文要求；"
            "由隔离的本地 Hermes 子会话执行该 agent 原文及其预载 skills，并执行原生 tools 白名单。"
            "子智能体不能再次派发；主会话不能用自称已审替代应派发的 review-images/review-footage。\n"
            "- AskUserQuestion：通过当前会话提问并等待答复；TodoWrite：用当前会话进度说明记录；"
            "这两项不授予额外权限。原生自动入口的明确总体授权仍由该入口上下文决定，本适配说明不授予自主制作权限。\n"
            "- Write/Edit/Bash、宿主脚本与外网工具：本宿主不直接提供；项目变更只用已有业务 MCP 工具。"
            "需要的动作没有对应受控工具时报告缺项，不执行任意脚本、文件写入、网络或旧工作流。\n"
            "按需读回原文再执行，不将文档目录、工具成功、文件存在或文本摘要视为已经完成原生审图审片。"
        )


def load_hermes_native_profile(content_mode: ContentMode, *, profile_root: Path | None = None) -> HermesNativeProfile:
    """Snapshot the same mode projection used by native profile materialization."""
    root = (profile_root if profile_root is not None else agent_profile_dir()).resolve(strict=True)
    mapping = resolve_profile_files_for_mode(root, content_mode)
    documents: dict[str, NativeProfileDocument] = {}
    for logical, source in sorted(mapping.items()):
        if not _is_document(logical):
            continue
        raw = _read_document_bytes(root, source)
        document = NativeProfileDocument(logical, source, hashlib.sha256(raw).hexdigest(), raw.decode("utf-8"))
        _frontmatter(document.content)
        documents[logical] = document
    for required in ("CLAUDE.md", ".claude/skills/video-workflow/SKILL.md"):
        if required not in documents:
            raise ValueError(f"Native Hermes profile is missing its entry document: {required}")
    fingerprint = json.dumps(
        {"content_mode": content_mode, "sources": {key: value.source_sha256 for key, value in documents.items()}},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return HermesNativeProfile(content_mode, MappingProxyType(documents), hashlib.sha256(fingerprint).hexdigest())
