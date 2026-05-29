# 非交互式自回归运行脚本变更说明

对比对象：

- 官方示例脚本：`/data3/dulingyi/worldmodel/HY-WorldPlay/run.sh`
- 当前本地脚本：`/data3/dulingyi/worldmodel/my_worldplay/HY-WorldPlay/test.sh`

本文只关注非交互式 autoregressive 运行路径，也就是不启用 interactive/web 控制，由脚本一次性给定输入图、pose、帧数和模型参数后启动生成。

## 1. 运行目标变化

官方 `run.sh` 中实际启用的是 autoregressive distilled model：

```bash
--action_ckpt $AR_DISTILL_ACTION_MODEL_PATH
--few_step true
--num_inference_steps 4
--model_type 'ar'
--transformer_resident_ar_rollout true
```

本地 `test.sh` 中实际启用的是普通 autoregressive model，并且 action checkpoint 指向 `ar_rl_model`：

```bash
AR_ACTION_MODEL_PATH=../ckpts/HY-WorldPlay/ar_rl_model/diffusion_pytorch_model.safetensors

--action_ckpt $AR_ACTION_MODEL_PATH
--few_step false
--model_type 'ar'
--transformer_resident_ar_rollout false
--offloading
```

影响：

- 官方脚本默认跑蒸馏 AR，推理步数少，偏向快速示例。
- 本地脚本改成普通 AR/RL checkpoint，`few_step=false`，更接近完整 AR 推理流程。
- 本地脚本关闭 `transformer_resident_ar_rollout` 并启用 `--offloading`，更偏向降低显存压力。

## 2. 输入方式变化

官方脚本只处理单张图片：

```bash
IMAGE_PATH=./assets/img/test.png
```

本地脚本改为批量图片列表：

```bash
IMAGE_PATHS=(
  /data3/dulingyi/worldmodel/my_worldplay/test_image/close_1.png
  ...
  /data3/dulingyi/worldmodel/my_worldplay/test_image/img19.png
)
```

并且使用循环逐张运行：

```bash
for IMAGE_PATH in "${IMAGE_PATHS[@]}"; do
  ...
done
```

影响：

- `test.sh` 可以一次批量生成多组结果。
- 每张图都会创建独立输出目录、终端日志和 metadata。
- 缺失图片会被跳过并记录到失败列表，最后以非零退出码结束。

## 3. 输出目录和日志变化

官方脚本输出到固定目录：

```bash
OUTPUT_PATH=./outputs/
```

本地脚本改为带时间戳和图片名的独立目录：

```bash
RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_ROOT=../outputs/test
OUTPUT_PATH="${OUTPUT_ROOT}/test_${IMAGE_PATH_TAG}_${RUN_TIMESTAMP}"
LOG_PATH="$OUTPUT_PATH/terminal_$(date +%Y%m%d_%H%M%S).log"
METADATA_PATH="$OUTPUT_PATH/run_metadata.json"
```

同时把 `torchrun` 输出写入终端日志：

```bash
2>&1 | tee "$LOG_PATH"
```

影响：

- 每次运行不会直接混到同一个 `./outputs/` 下。
- 可以按图片和运行时间追踪结果。
- 终端日志保存在对应输出目录中，便于排查失败原因。

## 4. Metadata 记录新增

本地脚本在每张图片运行前写入 `run_metadata.json`，记录：

- 当前状态：`running` / `failed` / `finished`
- 输入图片绝对路径
- 输出目录
- 终端日志路径
- prompt
- pose 拆分后的 actions
- 模型路径和 action checkpoint
- seed、分辨率、帧数、宽高、GPU 数、rewrite/SR 开关
- `CUDA_VISIBLE_DEVICES`
- prompt rewrite 相关 vLLM 地址和模型名

运行成功后还会扫描输出目录，把生成出来的视频文件路径写回：

```json
"output_videos": [...]
```

影响：

- 运行结果可追溯性明显增强。
- 批量实验时可以不用只依赖目录名或终端输出判断参数。
- 如果某张图片失败，metadata 会标记为 `failed`。

## 5. Prompt 和 rewrite 配置变化

官方脚本使用完整英文 prompt：

```bash
PROMPT='A paved pathway leads towards a stone arch bridge ...'
```

本地脚本把官方 prompt 注释掉，实际设置为空格：

```bash
PROMPT=" "
```

同时 rewrite 服务从占位符改成了本机 vLLM 地址和本地 Qwen 模型路径：

```bash
export T2V_REWRITE_BASE_URL="http://localhost:8000/v1"
export I2V_REWRITE_BASE_URL="http://localhost:8000/v1"
export T2V_REWRITE_MODEL_NAME="/data3/dulingyi/worldmodel/models/Qwen3.5-9B/Qwen/Qwen3.5-9B"
export I2V_REWRITE_MODEL_NAME="/data3/dulingyi/worldmodel/models/Qwen3.5-9B/Qwen/Qwen3.5-9B"
```

不过当前仍然设置：

```bash
REWRITE=false
```

影响：

- 当前非交互式 AR 实际不会做 prompt rewrite。
- 生成主要依赖输入图和 camera/action 条件，文本 prompt 基本为空。
- rewrite 环境变量已经配置好，但只有把 `REWRITE=true` 后才会参与生成。

## 6. Pose、帧数和 GPU 配置变化

官方脚本：

```bash
POSE='w-31'
NUM_FRAMES=125
N_INFERENCE_GPU=8
```

本地脚本：

```bash
POSE='w-46,left-41'
NUM_FRAMES=349
N_INFERENCE_GPU=4
export CUDA_VISIBLE_DEVICES=2,5,6,7
```

影响：

- 轨迹从单段前进 `w-31` 改为两段动作 `w-46,left-41`。
- 输出帧数从 125 增加到 349，生成长度更长。
- 并行 GPU 数从 8 改为 4，并显式指定使用物理 GPU `2,5,6,7`。

## 7. 路径配置变化

官方脚本使用绝对路径：

```bash
MODEL_PATH=/data3/dulingyi/worldmodel/HY-WorldPlay/HunyuanVideo-1.5
AR_ACTION_MODEL_PATH=/data3/dulingyi/worldmodel/HY-WorldPlay/HY-WorldPlay/ar_model/diffusion_pytorch_model.safetensors
```

本地脚本改成相对路径：

```bash
MODEL_PATH=../ckpts/HunyuanVideo-1.5
AR_ACTION_MODEL_PATH=../ckpts/HY-WorldPlay/ar_rl_model/diffusion_pytorch_model.safetensors
BI_ACTION_MODEL_PATH=../ckpts/HY-WorldPlay/bidirectional_model/diffusion_pytorch_model.safetensors
AR_DISTILL_ACTION_MODEL_PATH=../ckpts/HY-WorldPlay/ar_distilled_action_model/diffusion_pytorch_model.safetensors
```

影响：

- `test.sh` 默认需要从 `my_worldplay/HY-WorldPlay` 目录附近的 checkpoint 布局运行。
- 如果从其他工作目录直接执行，`../ckpts/...` 这类相对路径可能解析到错误位置；更稳妥的方式是在脚本所在目录执行，或者把路径改成基于脚本目录的绝对化路径。

## 8. Torchrun 参数变化

本地脚本新增或改变的关键参数：

```bash
--master_port=29511
--with-ui true
--use_vae_parallel false
--transformer_resident_ar_rollout false
--offloading
```

对比官方实际运行的 distilled AR 命令，本地脚本移除了：

```bash
--num_inference_steps 4
--use_sageattn false
--use_fp8_gemm false
--transformer_resident_ar_rollout true
```

影响：

- `--master_port=29511` 降低和其他 torchrun 任务端口冲突的概率。
- `--with-ui true` 被传给 `generate.py`，但脚本本身仍是非交互式批处理；这里不是 interactive/web 模式。
- `--offloading` 配合关闭 resident rollout，说明当前脚本更关注长视频/多图批处理下的显存控制。

## 9. 失败处理变化

本地脚本启用：

```bash
set -o pipefail
FAILED_IMAGES=()
```

并在 `tee` 管道后检查 `torchrun` 的真实退出码：

```bash
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
  ...
fi
```

影响：

- 即使用了 `tee`，也能正确捕获 `torchrun` 失败。
- 缺失图片和推理失败都会进入 `FAILED_IMAGES`。
- 只要有失败项，脚本最后会 `exit 1`。

## 10. `hyvideo/generate.py` 的非交互式路径变化

除了 bash 脚本，两个仓库里的 `hyvideo/generate.py` 也有差异。和非交互式 AR 直接相关的主要变化是：本地版本新增了 `--prompt_schedule_json`，并在 `generate_video(args)` 进入正式推理前改写 pose、video_length 和 prompt。

官方 `generate.py` 的 `generate_video(args)` 直接使用命令行传入的参数：

```python
viewmats, Ks, action = pose_to_input(args.pose, (args.video_length - 1) // 4 + 1)

out = pipe(
    prompt=args.prompt,
    video_length=args.video_length,
    viewmats=viewmats.unsqueeze(0),
    Ks=Ks.unsqueeze(0),
    action=action.unsqueeze(0),
    chunk_latent_frames=4 if args.model_type == "ar" else 16,
    model_type=args.model_type,
    transformer_resident_ar_rollout=args.transformer_resident_ar_rollout,
    **extra_kwargs,
)
```

本地 `generate.py` 在这之前增加了 prompt schedule 处理：

```python
prompt_schedule = None
if args.prompt_schedule_json:
    scheduled_pose, scheduled_video_length, prompt_schedule = load_prompt_schedule_json(
        args.prompt_schedule_json, args.prompt
    )
    args.pose = scheduled_pose
    args.video_length = scheduled_video_length
    if prompt_schedule:
        args.prompt = prompt_schedule[0]["prompt"]
```

之后调用 pipeline 时多传了一个参数：

```python
prompt_schedule=prompt_schedule
```

影响：

- 如果不传 `--prompt_schedule_json`，本地 `generate_video(args)` 的非交互式主干和官方基本一致，仍然是单次 `pose_to_input(...)` 加单次 `pipe(...)`。
- 如果传了 `--prompt_schedule_json`，本地版本会忽略 bash 中原本的 `--pose` 和 `--video_length`，改用 JSON segments 拼出来的完整 pose 和自动计算出的帧数。
- `args.prompt` 会被替换成第一个 segment 的 prompt，后续 prompt 切换通过 `prompt_schedule` 交给 pipeline。
- 这属于非交互式 prompt 分段能力，不需要 `--interactive true` 或 web 控制。

`load_prompt_schedule_json(...)` 支持的 JSON 形态是：

```json
{
  "segments": [
    {
      "pose": "w-46",
      "prompt": "first prompt"
    },
    {
      "pose": "left-41",
      "prompt": "second prompt"
    }
  ]
}
```

它的计算逻辑：

- 每个 segment 必须有非空 `pose`。
- `prompt` 可省略；省略时沿用上一个 prompt。
- 所有 segment 的 pose 会拼成一个逗号分隔的 pose 字符串。
- `video_length` 自动计算为 `latent_num * 4 - 3`。
- 每个 prompt 记录一个 `start_latent`，后续由 pipeline 在 AR chunk 边界应用。

需要注意：当前 `test.sh` 并没有传入 `--prompt_schedule_json`，所以这次对比的 `test.sh` 实际运行时不会触发上述分段 prompt 逻辑。

## 11. `hyvideo/generate.py` 的交互式新增能力

本地 `generate.py` 还新增了大量 interactive/web 相关代码：

- 新增 `--interactive`，用于选择 `generate_video_interactive(args)`。
- 新增 `--interactive_video_length`、`--initial_pose`、`--display_window`。
- 新增 `--web`、`--web_host`、`--web_port`，提供浏览器控制界面。
- 新增 VLM prompt cache 参数，例如 `--vlm_prompt_cache`、`--vlm_base_url`、`--vlm_model` 等。
- 新增 `incremental_poses_to_input(...)`，用于交互模式中把每一轮增量 camera poses 转成 viewmats/K/action。
- 交互模式会维护 `history_latents`、`history_cond_latents`、`history_viewmats`、`history_Ks`、`history_action`，并把它们传回 pipeline 实现跨轮连续生成。

这些改动对当前 `test.sh` 的非交互式运行没有直接影响，因为 `test.sh` 没有传 `--interactive true`。不过它们解释了为什么本地 `generate.py` 比官方文件大很多。

## 12. 当前 `test.sh` 实际会走的 generate 路径

当前 `test.sh` 的命令没有传：

```bash
--prompt_schedule_json
--interactive true
--web true
```

所以实际执行路径是：

```python
main()
  -> generate_video(args)
  -> pose_to_input(args.pose, (args.video_length - 1) // 4 + 1)
  -> pipe(..., prompt_schedule=None)
```

也就是说，当前 `test.sh` 的非交互式 AR 行为主要仍由 bash 中传入的这些参数决定：

- `--prompt "$PROMPT"`
- `--image_path "$IMAGE_PATH"`
- `--video_length $NUM_FRAMES`
- `--pose "$POSE"`
- `--action_ckpt $AR_ACTION_MODEL_PATH`
- `--few_step false`
- `--model_type 'ar'`
- `--transformer_resident_ar_rollout false`
- `--offloading`

`generate.py` 层面对这条路径的实质差异是：pipeline 调用多了 `prompt_schedule=None` 这个可选参数；在未启用 schedule 时，它不应改变单 prompt 非交互式生成逻辑。

## 13. 总结

`test.sh` 相比官方 `run.sh`，已经从一个单图、蒸馏 AR、固定输出目录的示例脚本，改成了一个面向批量实验的非交互式普通 AR 运行脚本。

主要变化是：

- 单图输入改为多图批处理。
- 官方 distilled AR 改为普通 AR/RL action checkpoint。
- 输出按图片和时间戳隔离。
- 新增终端日志和结构化 metadata。
- 新增失败检测、缺失图片跳过和最终失败汇总。
- 轨迹和帧数从 `w-31` / 125 帧改为 `w-46,left-41` / 349 帧。
- GPU 从默认 8 卡改为显式使用 `CUDA_VISIBLE_DEVICES=2,5,6,7` 的 4 卡。

`generate.py` 层面的主要变化是：

- 新增非交互式 `--prompt_schedule_json` 支持，可从 JSON segments 自动组装 pose、video_length 和 prompt schedule。
- `generate_video(args)` 在启用 schedule 时会覆盖 `args.pose`、`args.video_length`、`args.prompt`。
- pipeline 调用新增 `prompt_schedule` 参数。
- 新增 interactive/web/VLM prompt cache 运行路径，但当前 `test.sh` 不会触发。

需要特别注意的是，当前 `test.sh` 的 prompt 实际是一个空格，`REWRITE=false`，并且没有传 `--prompt_schedule_json`，所以这份脚本当前的非交互式 AR 结果主要由输入图片、动作轨迹、模型 checkpoint 和随机种子控制。

## 14. 除 `generate.py` 外的 Python 源码差异

按源码文件比较，排除 `hyvideo/generate.py` 后，官方目录和本地目录只有这些 Python 源文件不同：

- `hyvideo/generate_custom_trajectory.py`
- `hyvideo/pipelines/worldplay_video_pipeline.py`
- `hyvideo/utils/rewrite/clients.py`
- `hyvideo/display.py`，本地新增

对比输出里还有很多 `__pycache__/*.pyc` 差异，那些只是 Python 编译缓存，不是源码改动。

## 15. `generate_custom_trajectory.py`：相机状态持久化

这个文件的变化和 `interactive_changes.md` 中“修改点1：相机变换矩阵持久化”一致。

官方版本的 `generate_camera_trajectory_local(motions)` 每次调用都从单位矩阵开始：

```python
poses = []
T = np.eye(4)
poses.append(T.copy())
```

本地版本新增了 `CameraState`：

```python
class CameraState:
    def __init__(self):
        self.T = np.eye(4)
        self.intrinsic = [...]
        self.last_c2w = None
```

并把函数签名改成：

```python
def generate_camera_trajectory_local(motions, camera_state=None):
```

函数内部从 `camera_state.T` 开始，结束时写回：

```python
T = camera_state.T.copy()
poses = [T.copy()]
...
camera_state.T = T
camera_state.last_c2w = T.copy()
```

影响：

- 交互模式可以让下一轮动作从上一轮相机位置继续，而不是每轮回到原点。
- `CameraState.save()` / `restore()` 允许 Web/交互逻辑临时复制和恢复相机状态。
- 对当前 `test.sh` 的非交互式运行要特别注意：`test.sh` 每次启动一个新的 Python 进程，单进程内只调用一次 `pose_to_input(...)`，因此通常不会受到全局 `_global_camera_state` 的跨调用累积影响。
- 如果未来在同一个 Python 进程里连续多次调用 `pose_string_to_json(...)` 或 `pose_to_input(...)`，不显式传新的 `CameraState` 时可能会继承上一次的全局相机状态。这是和官方版本不同的行为。

## 16. `worldplay_video_pipeline.py`：历史上下文、prompt schedule 和返回 latent

这个文件是除 `generate.py` 外最关键的改动。它同时服务于两类能力：

- 交互式跨轮续写：历史 latent / pose / action 进入当前轮 AR context。
- 非交互式 prompt schedule：在 AR chunk 边界切换 prompt 条件。

### 16.1 Pipeline 输出新增中间状态

官方 `HunyuanVideoPipelineOutput` 只返回视频和可选 SR 视频：

```python
videos
sr_videos
```

本地版本新增：

```python
latents
cond_latents
viewmats
Ks
action
```

并在 `__call__` 结束时把这些张量 detach 后放到 CPU 返回。

影响：

- 交互模式每轮结束后可以保存本轮 latent、条件 latent、相机矩阵、内参和动作。
- 下一轮可以把这些历史张量再传回 pipeline。
- 对普通非交互式生成，主要影响是返回对象多了字段，视频生成本身不应改变。

### 16.2 `__call__` 新增历史参数

本地 `__call__` 新增参数：

```python
history_latents=None
history_cond_latents=None
history_viewmats=None
history_Ks=None
history_action=None
start_latent_idx=0
```

这些参数会继续传给 `ar_rollout(...)`。

影响：

- 交互式第二轮以后，`start_latent_idx` 表示当前 chunk 在全局 latent 时间线中的起点。
- 历史张量用于构造当前 chunk 的 context KV cache。
- 当前 `test.sh` 没传这些参数，所以它们都是默认值，不触发跨轮历史逻辑。

### 16.3 避免交互续写每轮重新吃原图条件

本地版本新增：

```python
if model_type == "ar" and start_latent_idx > 0:
    cond_latents.zero_()
```

这是 `interactive_changes.md` 后半部分提到的“避免每轮从原图第一帧重新开始”修复。

影响：

- 在交互模式中，第一轮仍可使用 i2v 参考图条件。
- 后续轮次不再把本轮局部第 0 个 latent 当作全局第一帧，减少每个 chunk 被原图拉回的问题。
- 当前 `test.sh` 的 `start_latent_idx=0`，所以不会触发这段 `cond_latents.zero_()`。

### 16.4 AR rollout 支持跨轮历史 context

官方 AR rollout 只在当前一次 pipeline 调用内部选择历史帧：

```python
selected_frame_indices = select_aligned_memory_frames(
    viewmats[0],
    chunk_start_idx,
    ...
)
context_latents = latents[:, :, selected_frame_indices]
```

本地版本会先判断是否有历史：

```python
has_history = history_latent_frames > 0
combined_viewmats = torch.cat([history_viewmats, viewmats.to(device)], dim=1)
```

然后用全局索引做 FOV 检索：

```python
selected_history_frame_id = select_aligned_memory_frames(
    combined_viewmats[0].cpu().detach().numpy(),
    start_latent_idx + chunk_start_idx,
    ...
)
```

检索结果再拆成两类：

```python
history_indices = [x for x in selected_frame_indices if x < start_latent_idx]
local_indices = [
    x - start_latent_idx
    for x in selected_frame_indices
    if start_latent_idx <= x < global_frame_idx
]
```

影响：

- 交互模式下，当前 chunk 可以注意到之前轮次生成的 latent。
- FOV 检索范围从“本次 pipeline 调用内的时间线”扩展为“历史 + 当前”的全局时间线。
- 这不是复用上一轮 transformer KV cache，而是每次按需选历史 latent，重新编码到当前 KV cache。
- 当前 `test.sh` 没传历史张量，非交互式内部 AR chunk 仍然按当前调用内的 latents 做上下文选择。

### 16.5 AR rollout 支持 prompt condition schedule

本地版本在 `__call__` 中读取：

```python
raw_prompt_schedule = kwargs.get("prompt_schedule")
```

如果存在，会为每个 schedule prompt 单独编码：

- `prompt_embeds`
- `prompt_mask`
- ByT5 embedding
- classifier-free guidance 下的 negative/positive 拼接

然后转换成按 chunk 生效的 `prompt_condition_schedule`：

```python
"start_chunk": (start_latent + chunk_latent_frames - 1) // chunk_latent_frames
```

在 `_ar_rollout_inner(...)` 里，每个 chunk 选择当前 prompt condition；一旦 prompt condition 变化，就重新缓存 text KV：

```python
prompt_condition_idx, prompt_condition = select_prompt_condition(chunk_i)
if prompt_condition_idx != active_prompt_condition_idx:
    cache_text_condition(prompt_condition)
    active_prompt_condition_idx = prompt_condition_idx
```

影响：

- 非交互式模式也可以在一个完整 AR 生成过程中按 segment 切换 prompt。
- prompt 切换按 AR chunk 对齐，不是在任意 latent/frame 处立即切换。
- 当前 `test.sh` 没传 `--prompt_schedule_json`，所以 `prompt_schedule=None`，不会触发。

## 17. `hyvideo/display.py`：本地新增实时显示模块

这是本地新增文件，官方目录没有。它对应 `interactive_changes.md` 中“修改点6：视频帧实时显示”。

主要内容：

- `VideoDisplay` 类
- `tensor_to_display_frame(...)`：把 torch tensor 转成 OpenCV BGR frame
- `draw_keyboard_overlay(...)`：绘制 WASD 和方向键叠加层
- `display_frame(...)`：调用 `cv2.imshow(...)` 展示单帧
- `get_display(...)` / `close_display(...)` / 全局 `display_frame(...)`
- `parse_keyboard_input(...)`：把 OpenCV key code 转成动作字典

影响：

- 只服务交互模式中的 OpenCV 窗口预览。
- 当前 `test.sh` 是非交互式批处理，不 import 也不调用该模块。

## 18. `hyvideo/utils/rewrite/clients.py`：rewrite 图片输入兼容 PIL Image

这个文件也有源码改动，但和 `interactive_changes.md` 的主线关系较弱。

官方版本的图片编码逻辑只接受图片路径：

```python
with Image.open(image_path) as img:
    ...
```

本地版本改成同时接受字符串路径和已经打开的 PIL 图片对象：

```python
if isinstance(image_path, str):
    img = Image.open(image_path)
elif isinstance(image_path, Image.Image):
    img = image_path
else:
    raise ValueError(...)
```

影响：

- prompt rewrite / VLM 相关调用可以直接传 `PIL.Image.Image`，不一定要先落盘成图片路径。
- 当前 `test.sh` 设置 `REWRITE=false`，所以不会走这段 rewrite 图片编码逻辑。
- 如果未来打开 `REWRITE=true` 或 Web/VLM prompt cache，这个改动会提高图片输入形式的兼容性。

## 19. 和 `interactive_changes.md` 的对应关系

`interactive_changes.md` 里提到的 Python 改动基本都能在源码差异里对应上：

- 修改点1：`generate_custom_trajectory.py` 新增 `CameraState`，持久化相机变换。
- 修改点2：`generate.py` 新增 `incremental_poses_to_input(...)`，前面已单独记录。
- 修改点3：`worldplay_video_pipeline.py` 新增历史 latent / cond / pose / action 输入，并在 AR rollout 中编码历史 context。
- 修改点4：`worldplay_video_pipeline.py` 调用 `select_aligned_memory_frames(...)` 时使用历史和当前拼接后的 `combined_viewmats`，没有修改 `hyvideo/utils/retrieval_context.py` 本身。
- 修改点5：`generate.py` 新增交互循环、Web 控制台、动作和 prompt 记忆，前面已单独记录。
- 修改点6：新增 `hyvideo/display.py`。

因此，除 `generate.py` 之外，真正改变推理行为的是 `worldplay_video_pipeline.py`；`generate_custom_trajectory.py` 和 `display.py` 主要为交互式入口服务；`clients.py` 是 rewrite/VLM 输入兼容性增强。
