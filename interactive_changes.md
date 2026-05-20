# 交互式运行改造实施文档

---

## 概述

按照 `change.md` 中的说明，将系统从「一次性输入完整轨迹 → 生成整段视频」的批处理模式改造为交互式运行模式。目前已完成修改点 1-6，并额外加入 Web 控制台、蒸馏模型启动脚本、跨轮历史 latent 上下文、以及按非交互式 AR 逻辑处理续写的修复。

---

## 修改文件清单

| 文件 | 修改类型 | 修改点 |
|------|----------|--------|
| `hyvideo/generate_custom_trajectory.py` | 编辑 | 修改点1：相机变换矩阵持久化 |
| `hyvideo/generate.py` | 编辑 | 修改点2：增量式pose编码 + 修改点5：交互循环入口 + Web 控制台 + 累积 latent 保存 |
| `hyvideo/pipelines/worldplay_video_pipeline.py` | 编辑 | 修改点3：跨轮历史上下文进入 KV cache + 修改点4：跨轮 FOV 检索 |
| `hyvideo/display.py` | 新建 | 修改点6：视频帧实时显示 |
| `test_interact.sh` | 编辑 | Web 交互模式启动脚本，默认使用 AR 蒸馏模型 |

---

## 修改点1：相机变换矩阵持久化

**文件：** `hyvideo/generate_custom_trajectory.py`

### 改动内容

1. **新增 `CameraState` 类**，用于跨轮次保存相机状态：

```python
class CameraState:
    def __init__(self):
        self.T = np.eye(4)          # 当前变换矩阵
        self.intrinsic = [...]       # 内参矩阵
        self.last_c2w = None         # 上一轮末帧的c2w

    def reset(self): ...            # 重置到初始状态
    def save(self): ...             # 序列化状态为dict
    def restore(self, state): ...   # 从dict恢复状态
```

2. **修改 `generate_camera_trajectory_local()` 函数**：
   - 新增 `camera_state` 参数，默认使用全局 `_global_camera_state`
   - `T` 的初始化从函数内部的 `np.eye(4)` 改为从 `camera_state.T` 读取
   - 函数结束时将最终的 `T` 和 `last_c2w` 写回 `camera_state`
   - **向后兼容**：不传 `camera_state` 时行为与原版一致（使用全局状态）

### 关键逻辑

```
调用前: T = camera_state.T.copy()  # 从上轮结束位置开始
循环中: 对每个motion更新T
调用后: camera_state.T = T          # 保存本轮结束位置
        camera_state.last_c2w = T    # 供下游计算相对变换
```

---

## 修改点2：增量式pose编码

**文件：** `hyvideo/generate.py`

### 改动内容

1. **新增 `incremental_poses_to_input()` 函数**，用于交互模式下的增量pose编码：

```python
def incremental_poses_to_input(incremental_poses, prev_c2w=None, intrinsic=None):
    """
    Args:
        incremental_poses: list of 4x4 c2w矩阵
        prev_c2w: 上一轮末帧的c2w矩阵（None时退化为原始行为）
        intrinsic: 内参矩阵
    Returns:
        (w2c_list, intrinsic_list, action_one_label)
    """
```

2. **`relative_c2w` 计算的关键差异**：

| 场景 | 第一帧的 relative_c2w | 后续帧的 relative_c2w |
|------|----------------------|----------------------|
| 原始模式（prev_c2w=None） | `c2ws[0]`（自身） | `inv(c2ws[i-1]) @ c2ws[i]` |
| 交互模式（prev_c2w存在） | `inv(prev_c2w) @ c2ws[0]` | `inv(c2ws[i-1]) @ c2ws[i]` |

   这确保了增量段的第一帧相对于上一轮末帧计算变换，而非全局第一帧。

3. **新增 import**：`from hyvideo.generate_custom_trajectory import ..., CameraState`

---

## 修改点5：入口函数改为交互循环

**文件：** `hyvideo/generate.py`

### 改动内容

1. **新增 `generate_video_interactive()` 函数**，核心流程：

```
初始化Pipeline（一次性）
    ↓
创建 CameraState
    ↓
执行 initial_pose（可选）
    ↓
┌─ 交互循环 ─────────────────────────┐
│  1. Rank 0 从stdin读取pose命令      │
│  2. 广播到所有GPU进程               │
│  3. 解析motions → 生成增量poses     │
│  4. incremental_poses_to_input()    │
│  5. 调用pipe()生成本轮视频          │
│  6. 累积本轮 frames 与 latent 上下文 │
│  7. 显示/OpenCV 或 Web 预览          │
└─────────────────────────────────────┘
    ↓
退出时优先用累计 latent 整体 decode 并保存完整视频
```

2. **分布式输入广播**：多GPU下，只有 rank 0 从 stdin 读取输入，然后通过 `torch.distributed.broadcast_object_list` 广播给所有进程：

```python
if int(os.environ.get("RANK", "0")) == 0:
    pose_input = input("Enter pose command: ").strip()
else:
    pose_input = ""

if torch.distributed.is_initialized():
    obj_list = [pose_input]
    if get_parallel_state().sp_enabled:
        group_src_rank = torch.distributed.get_global_rank(
            get_parallel_state().sp_group, 0
        )
        torch.distributed.broadcast_object_list(
            obj_list, src=group_src_rank, group=get_parallel_state().sp_group
        )
    else:
        torch.distributed.broadcast_object_list(obj_list, src=0)
    pose_input = obj_list[0]
```

3. **交互模式下的特殊处理**：
   - SR超分关闭（`enable_sr=False`），减少延迟
   - Prompt重写关闭（`prompt_rewrite=False`），加速响应
   - 每轮使用不同的seed（`seed + round_count`）
   - 支持 `quit`/`q`/`exit` 退出，`save` 保存当前视频

4. **新增命令行参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--interactive` | false | 启用交互模式 |
| `--interactive_video_length` | 13 | 每轮生成帧数；latent 数需能被 4 整除 |
| `--initial_pose` | None | 初始pose命令 |
| `--display_window` | true | 启用实时显示窗口 |
| `--web` | false | 启用浏览器 Web 控制台 |
| `--web_host` | 0.0.0.0 | Web 控制台监听地址 |
| `--web_port` | 7860 | Web 控制台端口 |

5. **`main()` 函数修改**：根据 `--interactive` 参数选择调用 `generate_video()` 或 `generate_video_interactive()`

---

## 修改点6：视频帧实时显示

**文件：** `hyvideo/display.py`（新建）

### 改动内容

1. **`VideoDisplay` 类**：
   - `tensor_to_display_frame()`：将 `[C, H, W]` tensor 转为 `[H, W, 3]` uint8 BGR 格式
   - `draw_keyboard_overlay()`：在画面上绘制 WASD + 方向键叠加层
   - `display_frame()`：显示单帧并返回键盘输入

2. **全局便捷函数**：
   - `get_display(create=True)` — 获取/创建显示实例
   - `close_display()` — 关闭显示窗口
   - `display_frame(frame, actions)` — 显示单帧

3. **键盘输入解析**：`parse_keyboard_input(key)` 将 OpenCV 按键映射为动作字典

---

## test_interact.sh 修改

**文件：** `test_interact.sh`

### 改动内容

将默认运行模式从批处理改为 Web 交互模式，并使用 AR 蒸馏模型：

```bash
# 新增交互模式参数
INTERACTIVE_VIDEO_LENGTH=13  # 每轮13帧，对应4个latent
INITIAL_POSE='w-4'           # 初始向前移动4步
WEB_PORT=7860

torchrun --nproc_per_node=$N_INFERENCE_GPU hyvideo/generate.py  \
  --prompt "$PROMPT" \
  --image_path $IMAGE_PATH \
  --resolution $RESOLUTION \
  --aspect_ratio $ASPECT_RATIO \
  --seed $SEED \
  --rewrite false \
  --sr false \
  --output_path $OUTPUT_PATH \
  --model_path $MODEL_PATH \
  --action_ckpt $AR_DISTILL_ACTION_MODEL_PATH \
  --few_step true \
  --num_inference_steps 4 \
  --width $WIDTH \
  --height $HEIGHT \
  --model_type 'ar' \
  --interactive true \
  --interactive_video_length $INTERACTIVE_VIDEO_LENGTH \
  --initial_pose "$INITIAL_POSE" \
  --display_window false \
  --web true \
  --web_host 0.0.0.0 \
  --web_port $WEB_PORT \
  --use_vae_parallel false \
  --use_sageattn false \
  --use_fp8_gemm false \
  --transformer_resident_ar_rollout true \
  2>&1 | tee "$LOG_PATH"
```

原始批处理命令已注释保留，方便切换。

---

## 使用方式

### 启动交互模式

```bash
cd /data3/dulingyi/worldmodel/my_worldplay/HY-WorldPlay
bash test_interact.sh
```

当前 `test_interact.sh` 默认启用 Web 控制台：

```bash
--web true \
--web_host 0.0.0.0 \
--web_port 7860 \
--display_window false
```

启动后在浏览器打开：

```text
http://服务器IP:7860
```

浏览器中按 `W/A/S/D` 或方向键会发送一个 chunk 的动作请求；后端生成完成后保存该 chunk 的 mp4，并在页面中自动追加播放。

Web 控制台包含 `Prompt for next chunk` 输入框。每次按键或点击方向按钮时，前端会把当前输入框中的 prompt 与动作一起提交给后端：

```json
{
  "action": "w",
  "prompt": "当前 chunk 使用的 prompt"
}
```

后端会将该 prompt 用于本轮 `pipe(prompt=...)` 调用，因此可以在每个 chunk 前修改场景描述或风格描述。

### 交互命令

| 命令 | 含义 | 示例 |
|------|------|------|
| `w-N` | 向前移动N步 | `w-4` |
| `s-N` | 向后移动N步 | `s-2` |
| `a-N` | 向左移动N步 | `a-4` |
| `d-N` | 向右移动N步 | `d-4` |
| `left-N` | 左转N步 | `left-4` |
| `right-N` | 右转N步 | `right-4` |
| `up-N` | 抬头N步 | `up-2` |
| `down-N` | 低头N步 | `down-2` |
| `quit` / `q` | 退出 | `quit` |
| `save` | 保存当前视频 | `save` |

可组合命令，用逗号分隔：`w-4,left-4,d-2`

---

## 已实施的修改（第二、三阶段）

### 修改点3：跨轮历史上下文进入 KV Cache

**文件：** `hyvideo/pipelines/worldplay_video_pipeline.py`

**目标：** 每轮生成当前 chunk 时，能够让 transformer attention 看到前面轮次生成出的历史 latent，而不是只依赖原始参考图。

**要点：**
- `HunyuanVideoPipelineOutput` 新增返回 `latents`、`cond_latents`、`viewmats`、`Ks`、`action`
- `generate_video_interactive()` 在每轮结束后累积这些历史张量
- 下一轮调用 `pipe()` 时传入历史张量和 `start_latent_idx`
- AR rollout 在 denoise 当前 chunk 前，将选中的历史 latent 编码进 vision KV cache
- 当前实现会按需重建选中历史帧的 KV cache，避免直接长期持有全量 transformer KV 造成显存不可控
- 这不是直接保存上一轮完整 transformer KV cache，而是参考非交互式 AR 的方式，为当前 chunk 选择历史 latent context 后重新编码到 KV cache 中

### 修改点4：跨轮历史帧 FOV 检索

**文件：** `hyvideo/utils/retrieval_context.py`

**目标：** 交互模式下检索范围跨越所有已生成的帧。

**要点：**
- 调用侧将历史 `viewmats` 与当前 chunk 的 `viewmats` 合并成全局序列
- `current_frame_idx` 使用 `start_latent_idx + local_idx`，保证检索在跨轮全局坐标中进行
- FOV 检索返回的全局索引会被拆分为历史索引和当前轮本地索引
- 被选中的历史 latent / cond / pose / action 会一起进入当前轮的 context KV cache

---

## 后续修复：避免每轮从原图第一帧重新开始

**问题：** 交互模式每轮都会重新调用一次 `pipe()`。如果仍按普通 i2v 处理，当前 chunk 的第 0 个 latent 会再次携带 `image_path` 的参考图条件。这样即使相机和历史 latent 连续，画面也会被原始第一帧拉回，表现为「每轮从头开始」和 chunk 边界跳变。

**参考非交互式 AR 的行为：**
- 非交互式整段生成时，只有全局第 0 个 latent 的 `multitask_mask` 为 1
- 后续 AR chunk 的图像条件为 0，只通过历史 latent context 和 FOV 检索续写

**当前修复：**
- `worldplay_video_pipeline.py` 中，当 `model_type == "ar"` 且 `start_latent_idx > 0` 时，对当前 chunk 的 `cond_latents.zero_()`
- 这表示后续交互轮次不再把本地第 0 个 latent 当作全局第一帧
- `generate.py` 中 `save` / `quit` 最终保存时，优先用累计 `history_latents` 一次性 VAE decode 成完整视频
- 如果累计 latent decode 失败，才退回到旧的逐 chunk decoded frames 拼接方式

**注意：**
- Web 页面中的每个 chunk 预览仍是单独 mp4，边界处可能仍有轻微播放层面的停顿
- 最终保存的视频优先走累计 latent 整体 decode，连续性更接近非交互式输出

---

## 新增：动作序列与 Prompt 记忆

**目标：** 每次交互不仅生成视频 chunk，还记录本轮动作和本轮使用的 prompt，便于复现或后续分析。

### 前端行为

- Web 页面新增 `Prompt for next chunk` 文本框
- 页面初始化时会填入启动脚本中的基础 prompt
- 每次按下 `W/A/S/D` 或方向键时，提交当前 prompt 文本框内容
- 当焦点在 prompt 文本框中时，键盘输入不会触发移动动作，避免编辑 prompt 时误操作

### 后端行为

`generate_video_interactive()` 中维护 `action_prompt_memory` 列表，每轮成功生成后追加：

```json
{
  "round": 1,
  "command": "w-4",
  "prompt": "本轮使用的 prompt",
  "frames": 13,
  "latent_frames": 4,
  "seed": 1
}
```

### 保存文件

点击 Web 页面 `Save` 或输入 `save` 时，除保存当前累计视频外，还会保存：

```text
interactive_memory_<round_count>.json
```

退出时最终保存：

```text
interactive_final.mp4
interactive_memory_final.json
```

记忆 JSON 的顶层结构为：

```json
{
  "base_prompt": "启动时传入的原始 prompt",
  "entries": [
    {
      "round": 1,
      "command": "w-4",
      "prompt": "本轮使用的 prompt",
      "frames": 13,
      "latent_frames": 4,
      "seed": 1
    }
  ]
}
```

---

## 依赖关系图

```
修改点1（相机持久化）✅ ← 修改点2（增量pose编码）✅ 依赖它
       ↓
修改点3（跨轮历史KV上下文）✅
       ↓
修改点4（跨轮帧检索）✅

修改点5（入口循环）✅   ← 整合修改点1、2的调用
修改点6（实时显示）✅   ← 独立，已完成
```

---

## 向后兼容性

所有修改均保持向后兼容：
- `generate_camera_trajectory_local()` 不传 `camera_state` 时使用全局状态，行为与原版一致
- `pose_to_input()` 未做任何修改，原有批处理模式完全不受影响
- `generate_video()` 函数的批处理入口不变
- pipeline 新增的历史上下文参数均为默认 `None`，普通批处理不传入时保持原行为
- `test_interact.sh` 中原始批处理命令已注释保留
