# HY-WorldPlay interactive / official AR / WMFactory comparison

This note compares three code paths:

- `test.sh`: near-official non-interactive autoregressive generation.
- `test_change_prompt.sh`: local interactive prompt-changing script.
- `from_yuheng/WMFactory/WMBackend`: another project's WorldPlay backend wrapper.

The focus is on video quality-relevant mechanics: latent history, camera trajectory stitching, action encoding, image conditioning, random seeds, and context selection.

## 1. Entry Points

### Official-style non-interactive AR: `test.sh`

`test.sh` calls:

```bash
torchrun --nproc_per_node=$N_INFERENCE_GPU hyvideo/generate.py \
  --video_length $NUM_FRAMES \
  --pose "$POSE" \
  --model_type ar
```

In `generate.py`, this reaches `generate_video(args)`, which does:

```python
viewmats, Ks, action = pose_to_input(args.pose, (args.video_length - 1) // 4 + 1)
out = pipe(..., viewmats=viewmats, Ks=Ks, action=action, chunk_latent_frames=4)
```

The whole pose sequence, all latent slots, and all AR chunks are prepared in one pipeline call.

### Local interactive script: `test_change_prompt.sh`

`test_change_prompt.sh` calls:

```bash
torchrun --nproc_per_node=$N_INFERENCE_GPU hyvideo/generate.py \
  --interactive true \
  --interactive_video_length $INTERACTIVE_VIDEO_LENGTH \
  --web true
```

This reaches `generate_video_interactive(args)`. Each user action causes one new pipeline call:

```python
out = pipe(
    video_length=interactive_video_length,
    viewmats=w2c_list.unsqueeze(0),
    Ks=Ks.unsqueeze(0),
    action=action.unsqueeze(0),
    history_latents=history_latents,
    history_cond_latents=history_cond_latents,
    history_viewmats=history_viewmats,
    history_Ks=history_Ks,
    history_action=history_action,
    start_latent_idx=history_latents.shape[2] if history_latents is not None else 0,
)
```

The script currently uses:

```bash
INTERACTIVE_VIDEO_LENGTH=13
```

This means each interaction generates 4 latent frames.

### WMFactory backend

WMFactory uses a different WAN-based pipeline under:

```text
from_yuheng/WMFactory/WMBackend/services/worldplay/app.py
```

Each frontend step calls:

```python
step() -> _append_chunk_poses() -> _current_chunk_inputs() -> _run_chunk()
```

It uses `CHUNK_SIZE = 4`, also generating 4 latent frames per interactive step. The major difference is that it maintains `runner.pipe.ctx`, `session.latent_history_cpu`, prompt embeddings, decode state, and sometimes KV cache across steps.

## 2. Latent History

### `test.sh`

The official-style path does not need an external history handoff. All latents for the full trajectory are allocated in one pipeline call:

```python
latents = self.prepare_latents(..., latent_target_length, ...)
self.chunk_num = latent_frames // chunk_latent_frames
latents = self.ar_rollout(...)
```

During AR rollout, previous chunks are already inside the same `latents` tensor. Context selection works over one coherent timeline.

This is the cleanest path.

### `test_change_prompt.sh`

The interactive path manually accumulates history after each call:

```python
history_latents = torch.cat([history_latents, out.latents], dim=2)
history_cond_latents = torch.cat([history_cond_latents, out.cond_latents], dim=2)
history_viewmats = torch.cat([history_viewmats, out.viewmats], dim=1)
history_Ks = torch.cat([history_Ks, out.Ks], dim=1)
history_action = torch.cat([history_action, out.action], dim=1)
```

On the next interaction, the accumulated history is passed back into `pipe()`.

Inside `worldplay_video_pipeline.py`, `ar_rollout()` combines old and current view matrices:

```python
combined_viewmats = torch.cat([history_viewmats, viewmats.to(device)], dim=1)
```

Then it selects context frames from the global timeline and splits selected indices into:

```python
history_indices = [x for x in selected_frame_indices if x < start_latent_idx]
local_indices = [x - start_latent_idx for x in selected_frame_indices if start_latent_idx <= x < global_frame_idx]
```

This works, but it is a retrofit. Each interactive call creates a fresh local `latents` tensor and depends on manually supplied history for continuity.

### WMFactory

WMFactory stores generated latent chunks on CPU:

```python
new_latents = ctx["latents"][:, :, start:end].detach().to("cpu")
session.latent_history_cpu = torch.cat([session.latent_history_cpu, new_latents], dim=2)
```

When a rolling context is needed, it rebuilds a compact window:

```python
latent_history = session.latent_history_cpu[:, :, retained_abs].to(device=latent_device)
latents[:, :, :history_window] = latent_history
```

It also reconstructs `pipe.ctx` with prompt embeddings, timesteps, latents, viewmats, actions, KV cache slots, and decode metadata.

This is closer to a service/runtime design. It is more complex but has clearer state ownership than the local interactive HY script.

## 3. Context / Memory Frame Selection

### `test.sh`

Official AR memory selection happens inside one `ar_rollout()` over a complete trajectory.

For each AR chunk after the first:

```python
selected_frame_indices = select_aligned_memory_frames(
    viewmats[0],
    current_frame_idx,
    memory_frames=20,
    temporal_context_size=12,
    pred_latent_size=4,
)
```

The current and previous latents all belong to the same `latents` tensor.

### `test_change_prompt.sh`

Interactive HY calls the same selection logic, but with global history stitched from previous calls:

```python
selected_frame_indices = select_aligned_memory_frames(
    combined_viewmats[0],
    start_latent_idx + chunk_start_idx,
    memory_frames=20,
    temporal_context_size=12,
    pred_latent_size=4,
)
```

Important differences:

- `history_latents` came from earlier independent pipeline calls.
- Current chunk latents are newly sampled in this call.
- The selected context is encoded into KV cache before denoising the current chunk.
- The selection can include old frames that are geometrically similar but visually inconsistent because they were generated in separate calls.

This can produce boundary jitter, vertical wobble, or scene drift.

### WMFactory

WMFactory uses WAN's `select_mem_frames_wan()`:

```python
selected_abs = select_mem_frames_wan(
    all_viewmats.cpu().numpy(),
    current_frame_idx,
    memory_frames=16,
    temporal_context_size=12,
    pred_latent_size=CHUNK_SIZE,
)
```

This keeps 12 recent latent frames plus up to 4 retrieved memory frames. Compared with HY's 20-frame memory window, this is more conservative.

WMFactory also supports `selected_frame_indices_override = list(range(history_window))` when it rebuilds a rolling window. In that branch, the pipeline receives a compact local timeline and treats all retained history as the context.

## 4. Camera Trajectory Stitching

### `test.sh`

The official-style path is simplest:

```python
pose_to_input(args.pose, latent_num)
```

`pose_to_input()` converts the entire pose string into a full pose sequence first, then computes action labels from adjacent poses:

```python
relative_c2w[0] = c2ws[0]
relative_c2w[1:] = inv(c2ws[:-1]) @ c2ws[1:]
```

There is no cross-call stitching.

### `test_change_prompt.sh`

The local interactive path maintains a `CameraState`:

```python
prev_c2w = camera_state.last_c2w.copy() if camera_state.last_c2w is not None else None
next_camera_state.restore(camera_state.save())
poses = generate_camera_trajectory_local(motions, next_camera_state)
```

Then it passes only the current chunk's poses into `incremental_poses_to_input()`:

```python
if prev_c2w is not None:
    incremental_poses = poses[1:]
else:
    incremental_poses = poses
```

`incremental_poses_to_input()` has a special case for the first action label of each subsequent chunk:

```python
relative_c2w[0] = inv(prev_c2w) @ c2ws[0]
action_start_idx = 0 if prev_c2w is not None else 1
```

This special case is necessary. Without it, the first latent action of every interactive chunk becomes zero, which causes boundary inconsistency.

The remaining issue is structural: only the current chunk poses are encoded in the current call, and global continuity is reconstructed by passing `history_viewmats` separately.

### WMFactory

WMFactory stores a full `pose_history`:

```python
base_pose = session.pose_history[-1]
relative_poses = generate_camera_trajectory_local([motion.copy() for _ in range(CHUNK_SIZE)])
for rel_pose in relative_poses[1:]:
    session.pose_history.append(base_pose @ rel_pose)
```

For each step, it recomputes viewmats and actions from the full pose history:

```python
all_viewmats, all_Ks, all_action = self._pose_to_input(pose_json, len(session.pose_history))
```

This is closer to `test.sh`: action labels are computed from one full trajectory, not from only the current local chunk plus a special boundary case.

## 5. Image Conditioning

### `test.sh`

The official path calls the pipeline once with `reference_image=args.image_path`. The first latent gets the i2v condition; later AR chunks naturally continue inside the same call.

### `test_change_prompt.sh`

Interactive HY also passes the reference image each call through:

```python
extra_kwargs["reference_image"] = args.image_path
```

To avoid each new chunk being pulled back to the original image, the pipeline has this guard:

```python
if model_type == "ar" and start_latent_idx > 0:
    cond_latents.zero_()
```

This is necessary, but it is a patch around the fact that each interactive chunk invokes the full i2v pipeline again.

### WMFactory

WMFactory only uses the seed image in the first chunk:

```python
image_path=str(session.seed_path)
```

For later chunks:

```python
image_path=None
```

This is cleaner and closer to the conceptual AR process.

## 6. Random Seed / Noise Continuity

### `test.sh`

One generator seed initializes the full `latents` tensor for the whole sequence.

### `test_change_prompt.sh`

Current interactive HY uses:

```python
seed=args.seed + round_count if args.seed else None
```

Each interaction creates a new random field with a different seed. This is one of the biggest reasons the local interactive path can show chunk-boundary wobble or unstable motion.

### WMFactory

WMFactory also seeds step generation, but it owns the rolling context and stores/rebuilds `pipe.ctx`. Its state management is more explicit. In `_build_windowed_ctx()`, new future latents are sampled only after retained historical latents are copied into the front of the local window:

```python
latents[:, :, :history_window] = latent_history
```

## 7. Prompt Updates

### `test.sh`

No runtime prompt changes. Prompt is fixed for the entire sequence.

### `test_change_prompt.sh`

Each action request can pass a prompt:

```python
command_queue.put({"prompt": prompt, ...})
```

The current chunk is generated with that prompt. This means prompt embeddings can change abruptly at chunk boundaries. This is desired for editing, but it can reduce temporal consistency.

### WMFactory

WMFactory updates session prompt explicitly:

```python
prompt_changed = self._update_session_prompt(session, action)
```

When the prompt changes, it refreshes prompt embeddings and KV cache state:

```python
_refresh_prompt_context(session)
pipe.init_kv_cache()
pipe.ctx["kv_cache"] = pipe._kv_cache
```

This is more controlled than simply calling a fresh full pipeline with a new prompt.

## 8. Why `test_change_prompt.sh` Is Less Stable Than `test.sh`

The non-interactive official-style path is more stable because:

1. It allocates one full latent timeline.
2. It computes one full pose/action timeline.
3. It performs AR rollout inside one pipeline call.
4. Context selection reads from one consistent latent tensor.
5. The reference image condition is naturally applied only at the global beginning.
6. The seed initializes the whole video once.

The local interactive path is less stable because:

1. Every interaction is a new pipeline call.
2. Each chunk has newly sampled latents.
3. Prompt embeddings may change abruptly per chunk.
4. Camera/action continuity depends on manual boundary handling.
5. History is passed in as external tensors, not as native in-call AR state.
6. It uses a large memory window (`memory_frames=20`) that may retrieve visually inconsistent older latents.

## 9. How To Make `test_change_prompt.sh` Closer To Official Non-Interactive AR

Recommended changes, ordered by impact.

### 1. Keep a Full Pose History and Recompute Action Labels Globally

Adopt WMFactory's approach:

```python
pose_history = [np.eye(4)]
append new chunk poses to pose_history
all_viewmats, all_Ks, all_action = pose_to_input(full_pose_history)
curr_viewmats = all_viewmats[start_idx:end_idx]
curr_action = all_action[start_idx:end_idx]
```

This removes the need for special local boundary action logic and makes action labels closer to `test.sh`.

### 2. Stop Changing Seed Per Chunk

Change:

```python
seed=args.seed + round_count if args.seed else None
```

to either:

```python
seed=args.seed if args.seed else None
```

or expose a flag:

```bash
--interactive_fixed_seed true
```

This will not perfectly reproduce non-interactive AR, but it reduces chunk-to-chunk random discontinuity.

### 3. Increase Interactive Chunk Length

Current:

```bash
INTERACTIVE_VIDEO_LENGTH=13
```

This creates one boundary every 4 latents. Try:

```bash
INTERACTIVE_VIDEO_LENGTH=29  # 8 latents
INTERACTIVE_VIDEO_LENGTH=45  # 12 latents
```

Fewer boundaries usually means less jitter.

### 4. Match Memory Window To WMFactory

HY currently uses:

```python
memory_frames=20
temporal_context_size=12
```

WMFactory uses:

```python
memory_frames=16
temporal_context_size=12
```

For stability, try reducing HY to `memory_frames=16`, so the context becomes 12 recent + 4 retrieved memory frames.

### 5. Avoid Always Keeping Old Initial Frames

HY's `select_aligned_memory_frames()` initializes:

```python
memory_frames_indices = [0, 1, 2, 3]
```

This forces the first chunk into memory. It may help identity/scene anchoring, but it can also pull the camera back to an old view and cause wobble. For interactive control, consider changing this to:

```python
memory_frames_indices = []
```

This matches the WAN selector used by WMFactory more closely.

### 6. Make Later Chunks Explicitly `reference_image=None`

Instead of passing `reference_image=args.image_path` on every interactive call and zeroing `cond_latents` internally, make the call explicit:

```python
if history_latents is None:
    extra_kwargs["reference_image"] = args.image_path
else:
    extra_kwargs["reference_image"] = None
```

This is conceptually closer to WMFactory and easier to reason about.

### 7. Consider a Rolling `pipe.ctx` Design

The largest quality improvement would be to stop treating each interaction as an independent `pipe()` call with injected external history. Instead:

1. Initialize a context on the first call.
2. Keep latent history and KV cache in a runtime/session object.
3. Decode chunk-by-chunk.
4. When history exceeds a window, rebuild a compact context like WMFactory.

This is more engineering work, but it is the closest path to robust interactive quality.

## 10. Practical Short-Term Patch Set

If the goal is to quickly make `test_change_prompt.sh` closer to `test.sh` without rewriting the pipeline:

1. Change chunk length:

```bash
INTERACTIVE_VIDEO_LENGTH=29
```

2. Use fixed seed per chunk:

```python
seed=args.seed if args.seed else None
```

3. Reduce memory selection to 16:

```python
memory_frames=16
```

4. Remove forced first-chunk memory sink:

```python
memory_frames_indices = []
```

5. Use full pose history for `viewmats/K/action`.

These changes should reduce vertical jitter and chunk boundary instability while preserving prompt editing.

## 11. Bottom Line

`test.sh` is stable because it performs one coherent AR rollout over a full trajectory. `test_change_prompt.sh` is flexible because it allows prompt editing, but it pays for that by rebuilding generation state every chunk. WMFactory is closer to a real interactive runtime: it keeps session state, stores latent history, rebuilds compact context windows, and recomputes pose/action from full history.

To make `test_change_prompt.sh` behave more like official AR, prioritize full pose-history action encoding, fixed/random-continuous latent handling, smaller and more recent memory windows, and explicit first-image-only conditioning.
