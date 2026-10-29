"""Native production read model and the small recovery operations it exposes."""

from server.agent_toolset.declaration import BLOCKED, Exempt, ToolDeclaration
from server.media_tools.production import (
    EpisodeProductionStatusRequest,
    RetryArtifactDownloadRequest,
    SelectImageVersionRequest,
    get_episode_production_status,
    retry_artifact_download,
    select_image_version,
)

PRODUCTION_TOOLS = (
    ToolDeclaration(
        name="get_episode_production_status",
        request_model=EpisodeProductionStatusRequest,
        description="只读本集正式制作事实：精确单元、配音实测时长、首尾帧、图像历史版本、视频、所有在途任务、失败与成片。goal_complete_technical只认可覆盖本集全部单元的current带旁白烧字幕成片，不以视频片段齐全替代交付。",
        migration=Exempt("只读正式制作事实；迁移缺口原样返回，不写文件、不推理。"),
        domain_key="episode_production_status",
        handler=get_episode_production_status,
    ),
    ToolDeclaration(
        name="select_image_version",
        request_model=SelectImageVersionRequest,
        description="使用与Web相同的原生版本恢复事务，将现有资产图或分镜首图选回指定历史版本。不生成新图；已有end_frame_image是独立快照，不受首图版本恢复影响。常用于生成并绑定尾图后恢复原首图。",
        migration=BLOCKED,
        domain_key="select_image_version",
        handler=select_image_version,
    ),
    ToolDeclaration(
        name="retry_artifact_download",
        request_model=RetryArtifactDownloadRequest,
        description="仅恢复本项目符合原生下载恢复资格的失败任务，复用原provider_job_id与执行检查点下载、记账、登记，不重新提交推理。仍在运行、未接受供应商任务、非下载错误或其它项目的任务均拒绝。",
        migration=Exempt("只沿既有供应商任务恢复下载；保持HTTP原生下载恢复准入。"),
        domain_key="artifact_download_retry",
        handler=retry_artifact_download,
    ),
)
