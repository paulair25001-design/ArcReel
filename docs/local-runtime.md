# 本地 Hermes 与 L20 运行方式

本分支默认启用 `ARCREEL_LOCAL_ONLY=true`。项目、素材、分镜、任务队列、版本和剪辑导出全部由 ArcReel 管理；对话与后台文本任务调用本机 Hermes，图片和视频通过原生 ComfyUI API 调用自有 L20，旁白调用免费本地 Qwen3-TTS。任何本地失败都会保留错误，不转云端。L20 provider 的推理图、请求和资源会话代码都在本仓库内，不加载其他制作项目、factory、catalog 或技能工作流。

## 启动

在仓库根目录执行：

```bash
uv sync --locked
pnpm --dir frontend install --frozen-lockfile
pnpm --dir frontend build
bash scripts/start_local.sh
```

打开 <http://127.0.0.1:1241>。默认保留登录认证，用户名为 `admin`；首次启动自动生成的密码保存在仓库 `.env` 的 `AUTH_PASSWORD` 中。启动脚本只监听本机回环地址。停止时在启动终端按 Ctrl+C，应用会先停止任务服务，再结束本应用拥有的 L20 会话。

启动工作台不会提交图像、视频、配音或文本生成任务，也不会加载、重启 oMLX。Hermes 的模型必须已由本机 oMLX 提供。L20 在首次媒体任务进入本项目资源会话后才开始工作。

## 实际模型与输入约束

| 用途 | 路线 | 当前约束 |
| --- | --- | --- |
| 对话、规划、结构化文本及图片理解 | 本机 Hermes → oMLX | 固定配置的本地模型；Hermes 只挂载当前项目的 ArcReel MCP 工具 |
| 文生图、图片编辑 | L20 Qwen-Image 2.1 INT8 | `1K` 画布预设；16:9 为 1024×576，9:16 为 576×1024；图片编辑最多两张参考图 |
| 图生视频 | L20 FastH3 V2 | 576p、24 fps、1～10 秒；必须有首帧和明确的目标尾帧；不支持参考生视频或参考音频 |
| 旁白 | 本机 Qwen3-TTS VoiceDesign | 免费、离线；用声音描述和逐段演法控制配音，不支持数值语速或自动付费回退 |

其他画幅按相同比例和 32 像素网格计算。图像与视频共用画布，首尾帧尺寸必须与请求一致。图片编辑会保留原图，把发送的参考图按画布等比缩放并补边，记录转换和两份 SHA-256。超过两张参考图会拒绝，不静默丢弃。

本地流程使用逐镜生成的完整画布分镜。上游宫格功能切出的单格尺寸较小，不能直接进入 FastH3；需要先为每镜重新生成完整画布图片，再设置相同尺寸的目标尾帧。

在分镜界面选择或上传目标尾帧；Hermes 也可用 `set_storyboard_end_frame` 从项目内已有图片建立尾帧快照。尾帧表达镜头结束时的实际画面，不自动复制首帧。缺少尾帧时，视频任务在入队前被拒绝。

FastH3 原生生成音轨，不接受参考音频，也不支持关闭原生音轨；关闭音频的项目设置会在生成前被拒绝。旁白可在后期按时间线混音。

FastH3 的原生对齐帧数可能超过声明时长，适配器保留原片，并截取请求时长所需的前段帧，不改变播放速度。文件可解码、帧数和哈希通过，只代表技术检查，不代表画质、表演、听感或口型已经通过审核。

## 资源与会话

图像和视频共享本项目的 L20 资源会话与跨进程锁。ArcReel 保存原 llama router 的模型 loaded 集合和身份，核实所有 child slots 连续两轮空闲后，通过 router 的 `/models/unload` 释放模型；router 进程保持运行。制作期间不会在每个镜头之间反复恢复原模型。模型权重、已安装 ComfyUI 和纯推理节点属于运行依赖；项目队列、审核与剪辑仍由 ArcReel 管理。

当整个项目的媒体任务完成，或停止制作且任务队列已空时，让 Hermes 调用 `finish_local_session`。应用等待自己的 ComfyUI 队列清空、释放推理模型后，通过 `/models/load` 恢复原先的 loaded 集合，并核对健康状态与空闲 slots。恢复失败时保留持久记录与恢复责任，不报告完成。

点击停止或删除对话会话时，应用也会等待该项目的排队和运行任务结束，再结束本项目资源会话；停止对话不等于取消已提交的媒体作业。继续制作会撤销尚未执行的恢复请求；同项目另有工作中的对话时，也会保留制作会话。应用正常退出及异常关闭的收尾路径都会先等待工作任务，再结束本应用拥有的资源会话。

每个媒体任务在提交前冻结实际推理图、输入 SHA-256 和 provider handle，然后只提交一次 `/prompt`。已取得 `prompt_id` 的任务直接查询 `/history` 并通过 `/view` 下载。提交响应丢失时，只从队列和历史记录中按同一 handle、图内容与图哈希精确定位；无匹配或有多笔匹配均保留故障，不盲目重投。完成历史中的 graph 必须与冻结请求完全一致。

Hermes 配置、会话绑定和项目专用 MCP 密钥保存在本应用数据根的 `runtime/hermes/`；独立 L20 provider 的状态和恢复记录在 `runtime/l20-direct/`。不会修改全局 Hermes 配置或其他制作项目的授权。后台文本调用每次使用独立、无工具上下文；交互会话可以恢复，但当前不支持改写历史消息。

Hermes 的 HTTP 资源屏障由本仓库的 `lib/local_runtime/resource_queue.py` 提供，锁文件保存在本应用数据根的 `runtime/resources/`。每个本地文本 HTTP 请求只在实际请求和响应消费期间持有共享锁，工具执行不占锁；配音持有独占锁，等待中的配音优先于后来的文本请求。锁等待可取消或超时，不停止其他服务。

本地配音使用项目中的 `narration_voice_description`（声线描述）和 `narration_instructions`（项目默认演法），未设置时使用本地默认值。创建或修改项目的 HTTP 接口接受这两个字段；Hermes 可通过 `create_project` 设置，或通过 `patch_project` 的 `settings` 修改。传入空字符串或 `null` 可清除已存值，恢复本地默认值。任务会冻结实际使用的描述与演法，并记入版本依据；修改后，旧配音会按依据变化重新判定时效。界面目前没有编辑这两个自然语言参数的输入框。

后台结构化文本任务向 oMLX 提交原始 JSON schema，并同时请求 `response_format` 与显式 `structured_outputs`。每个新 worker 的 `wire-audit.json` 只保存实际 HTTP 出站的端点、模型匹配标记及两处 schema 的哈希，不保存提示词、密钥、HTTP 头或响应正文。约束在传输过程中缺失或改变时，worker 在发送前拒绝请求。该回执证明实际提交的约束，不证明服务端 logits processor 已执行；后者仍需服务端运行证据。历史任务不会补写回执。

本地配音原始执行回执保存在 `runtime/tts/receipts/`，版本记录绑定原始回执哈希及实际提交的 WAV。回执中的「未听审」状态保持原样。

配音由本仓库的 `lib/local_runtime/qwen_tts_worker.py` 直接调用已安装的独立 MLX Audio 引擎库。模型必须是已存在的本地 Qwen3-TTS VoiceDesign；引擎运行在禁止网络的 macOS 沙箱中，并关闭模型下载、参考音频转录、自动播放与云端回退。声音输入经权限为 `0600` 的项目私有请求文件传入，不出现在命令行或正常标准输出中；完整 PCM 验证后才登记版本。项目代码不调用外部 skill wrapper、旧制作流程或它们的锁文件。

本地配音进程非零退出时，`runtime/tts/receipts/failures/` 保存私有失败回执、脱敏 stdout／stderr 和命令摘要，权限为 `0600`。失败提示仅显示识别出的保护原因和回执路径，不显示旁白原词、声线描述、演法或密钥。保护锁忙时保留当前任务，等待资源释放；不停止其他服务，不自动重试或转付费。未保存诊断的历史失败无法事后确定具体原因。

## 可选环境配置

配置写入仓库 `.env`，重启本应用后生效；不要把本机绝对路径写进共享工作流。

```dotenv
ARCREEL_LOCAL_ONLY=true
# 本机 oMLX 的回环地址，禁止配置公网地址
ARCREEL_HERMES_BASE_URL=http://127.0.0.1:8000/v1
# 若本机服务启用认证，填写它已有的本地 API 密钥
# ARCREEL_HERMES_API_KEY=
ARCREEL_HERMES_MODEL=Qwen3.8-Flash-Next-Uncensored-oQ5e-mtp
# 以下默认使用当前机器已有安装，可覆盖安装路径
# ARCREEL_HERMES_SOURCE=/path/to/hermes-agent
# ARCREEL_HERMES_PYTHON=/path/to/hermes-agent/venv/bin/python
# ARCREEL_TTS_PYTHON=/path/to/offline/python3.11
# ARCREEL_TTS_SITE_PACKAGES=/path/to/installed/mlx-audio/site-packages
# ARCREEL_TTS_MODEL=/path/to/qwen3-tts-voice-design
# 本项目管理的 SSH 回环转发；不接受云端 ComfyUI URL
ARCREEL_L20_SSH_TARGET=my-l20
ARCREEL_L20_SSH_OPTIONS_JSON='["-o","BatchMode=yes","-o","StrictHostKeyChecking=yes"]'
ARCREEL_L20_COMFY_URL=http://127.0.0.1:18191
ARCREEL_L20_ROUTER_URL=http://127.0.0.1:18080
# ARCREEL_L20_DIRECT_STATE_DIR=/path/to/arcreel-data/runtime/l20-direct
# ARCREEL_L20_COMFY_PYTHON=/path/to/installed/comfyui/python
# ARCREEL_L20_COMFY_ENGINE=/path/to/installed/ComfyUI
```

`my-l20` 应替换为已配置并验证主机密钥的 SSH 目标；需要跳板时，在 JSON 参数中显式加入 `-J` 和跳板名。SSH 连接与回环转发由本项目代码创建，远端使用独立的 ArcReel 输入、输出与服务目录。当前接入绑定已验证的 L20 主机 UID、`/nvme1` 挂载、模型目录与 router 端口；迁往其他服务器前应先调整并验证 `l20_remote_resources.py` 的主机配置。

本模式禁用其他供应商的连接测试、模型发现和端点试运行，关闭官方目录同步及官方服务。无需填写云端密钥。显式设置 `ARCREEL_LOCAL_ONLY=false` 仅用于上游兼容性开发与测试；本地启动脚本始终强制开启本地模式。
