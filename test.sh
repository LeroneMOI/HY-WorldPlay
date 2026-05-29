export PYTHONPATH=$(cd "$(dirname "$0")" && pwd):$PYTHONPATH

# export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=2,5,6,7

# 设置 vLLM 服务地址（在本机运行）
export T2V_REWRITE_BASE_URL="http://localhost:8000/v1"
export I2V_REWRITE_BASE_URL="http://localhost:8000/v1"

# 设置模型名称（使用你启动时的模型路径）
export T2V_REWRITE_MODEL_NAME="/data3/dulingyi/worldmodel/models/Qwen3.5-9B/Qwen/Qwen3.5-9B"
export I2V_REWRITE_MODEL_NAME="/data3/dulingyi/worldmodel/models/Qwen3.5-9B/Qwen/Qwen3.5-9B"

# PROMPT='A paved pathway leads towards a stone arch bridge spanning a calm body of water.  Lush green trees and foliage line the path and the far bank of the water. A traditional-style pavilion with a tiered, reddish-brown roof sits on the far shore. The water reflects the surrounding greenery and the sky.  The scene is bathed in soft, natural light, creating a tranquil and serene atmosphere. The pathway is composed of large, rectangular stones, and the bridge is constructed of light gray stone.  The overall composition emphasizes the peaceful and harmonious nature of the landscape.'
PROMPT=" "
# PROMPT="A sleek, neon-lit metallic walkway leads toward a high-tech charging station built from dark steel, glowing blue circuits, and glass panels. A metallic, robotic synthetic horse stands inside the illuminated bay. To the right, the path opens up to a sprawling futuristic cityscape with towering skyscrapers, holographic billboards, and flying vehicles under a deep purple night sky. In the center, a cyborg wanderer stands with their back to the viewer, wearing a tactical dark suit with glowing neon accents and carrying a semi-transparent hexagonal energy shield. The scene is bathed in the moody glow of neon lights, evoking a sense of gritty, futuristic exploration. The overall composition emphasizes the contrast between advanced technology and a neon-drenched cyberpunk metropolis"


SEED=1
ASPECT_RATIO=16:9
RESOLUTION=480p # Now we only provide the 480p model
RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_ROOT=../outputs/test/$RUN_TIMESTAMP
MODEL_PATH=../ckpts/HunyuanVideo-1.5                   # Path to pretrained hunyuanvideo-1.5 model
# AR_ACTION_MODEL_PATH=../ckpts/HY-WorldPlay/ar_model/diffusion_pytorch_model.safetensors         # Path to our HY-World 1.5 autoregressive checkpoints
AR_ACTION_MODEL_PATH=../ckpts/HY-WorldPlay/ar_rl_model/diffusion_pytorch_model.safetensors
BI_ACTION_MODEL_PATH=../ckpts/HY-WorldPlay/bidirectional_model/diffusion_pytorch_model.safetensors         # Path to our HY-World 1.5 bidirectional checkpoints
AR_DISTILL_ACTION_MODEL_PATH=../ckpts/HY-WorldPlay/ar_distilled_action_model/diffusion_pytorch_model.safetensors # Path to our HY-World 1.5 autoregressive distilled checkpoints
POSE='w-69,left-90'                   # Camera trajectory: pose string (e.g., 'w-31' means generating [1 + 31] latents) or JSON file path
# POSE='w-55'
NUM_FRAMES=637
WIDTH=832
HEIGHT=480

# Configuration for faster inference
# The maximum number recommended is 8.
N_INFERENCE_GPU=4 # Parallel inference GPU count.

# Configuration for better quality
REWRITE=false   # Enable prompt rewriting. Please ensure rewrite vLLM server is deployed and configured.
ENABLE_SR=false # Enable super resolution. When the NUM_FRAMES == 125, you can set it to true

# Multiple input images. Add or remove paths here as needed.
IMAGE_PATHS=(
#   /data3/dulingyi/worldmodel/my_worldplay/test_image/close_1.png
#   /data3/dulingyi/worldmodel/my_worldplay/test_image/close_2.png
#   /data3/dulingyi/worldmodel/my_worldplay/test_image/close_3.png
#   /data3/dulingyi/worldmodel/my_worldplay/test_image/close_4.png
#   /data3/dulingyi/worldmodel/my_worldplay/test_image/close_5.png
#   /data3/dulingyi/worldmodel/my_worldplay/test_image/close_6.png
#   /data3/dulingyi/worldmodel/my_worldplay/test_image/close_7.png
#   /data3/dulingyi/worldmodel/my_worldplay/test_image/close_8.png
#   /data3/dulingyi/worldmodel/my_worldplay/test_image/close_9.png
#   /data3/dulingyi/worldmodel/my_worldplay/test_image/close_10.png
  /data3/dulingyi/worldmodel/my_worldplay/test_image/img1.png
  /data3/dulingyi/worldmodel/my_worldplay/test_image/img2.png
  /data3/dulingyi/worldmodel/my_worldplay/test_image/img3.png
  /data3/dulingyi/worldmodel/my_worldplay/test_image/img4.png
  /data3/dulingyi/worldmodel/my_worldplay/test_image/img5.png
  /data3/dulingyi/worldmodel/my_worldplay/test_image/img6.png
  /data3/dulingyi/worldmodel/my_worldplay/test_image/img7.png
  /data3/dulingyi/worldmodel/my_worldplay/test_image/img8.png
  /data3/dulingyi/worldmodel/my_worldplay/test_image/img9.png
  /data3/dulingyi/worldmodel/my_worldplay/test_image/img10.png
  /data3/dulingyi/worldmodel/my_worldplay/test_image/img11.png
  /data3/dulingyi/worldmodel/my_worldplay/test_image/img12.png
  /data3/dulingyi/worldmodel/my_worldplay/test_image/img13.png
  /data3/dulingyi/worldmodel/my_worldplay/test_image/img14.png
  # /data3/dulingyi/worldmodel/my_worldplay/test_image/img15.png
  /data3/dulingyi/worldmodel/my_worldplay/test_image/img16.png
  /data3/dulingyi/worldmodel/my_worldplay/test_image/img17.png
  /data3/dulingyi/worldmodel/my_worldplay/test_image/img18.png
  /data3/dulingyi/worldmodel/my_worldplay/test_image/img19.png
)

set -o pipefail
FAILED_IMAGES=()

# inference with bidirectional model
# torchrun --nproc_per_node=$N_INFERENCE_GPU hyvideo/generate.py  \
#   --prompt "$PROMPT" \
#   --image_path $IMAGE_PATH \
#   --resolution $RESOLUTION \
#   --aspect_ratio $ASPECT_RATIO \
#   --video_length $NUM_FRAMES \
#   --seed $SEED \
#   --rewrite $REWRITE \
#   --sr $ENABLE_SR --save_pre_sr_video \
#   --pose "$POSE" \
#   --output_path $OUTPUT_PATH \
#   --model_path $MODEL_PATH \
#   --action_ckpt $BI_ACTION_MODEL_PATH \
#   --few_step false \
#   --model_type 'bi'

for IMAGE_PATH in "${IMAGE_PATHS[@]}"; do
  if [[ ! -f "$IMAGE_PATH" ]]; then
    echo "[WARN] Skip missing image: $IMAGE_PATH"
    FAILED_IMAGES+=("$IMAGE_PATH")
    continue
  fi

  IMAGE_PATH_TAG="${IMAGE_PATH#./}"
  IMAGE_PATH_TAG="${IMAGE_PATH_TAG//\//-}"
  IMAGE_PATH_TAG="${IMAGE_PATH_TAG%.*}"
  OUTPUT_PATH="${OUTPUT_ROOT}/test_${IMAGE_PATH_TAG}_${RUN_TIMESTAMP}"
  mkdir -p "$OUTPUT_PATH"
  LOG_PATH="$OUTPUT_PATH/terminal_$(date +%Y%m%d_%H%M%S).log"
  METADATA_PATH="$OUTPUT_PATH/run_metadata.json"

  echo "[INFO] Start inference: $IMAGE_PATH"
  echo "[INFO] Output path: $OUTPUT_PATH"

  python - "$METADATA_PATH" "$IMAGE_PATH" "$OUTPUT_PATH" "$LOG_PATH" "$PROMPT" "$POSE" "$MODEL_PATH" "$AR_ACTION_MODEL_PATH" "$BI_ACTION_MODEL_PATH" "$AR_DISTILL_ACTION_MODEL_PATH" "$SEED" "$ASPECT_RATIO" "$RESOLUTION" "$NUM_FRAMES" "$WIDTH" "$HEIGHT" "$N_INFERENCE_GPU" "$REWRITE" "$ENABLE_SR" <<'PY'
import json
import os
import sys
import time

(
    metadata_path,
    image_path,
    output_path,
    log_path,
    prompt,
    pose,
    model_path,
    ar_action_model_path,
    bi_action_model_path,
    ar_distill_action_model_path,
    seed,
    aspect_ratio,
    resolution,
    num_frames,
    width,
    height,
    n_inference_gpu,
    rewrite,
    enable_sr,
) = sys.argv[1:]
metadata_path = os.path.abspath(metadata_path)
data = {
    "status": "running",
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
    "server_paths": {
        "input_image": os.path.abspath(image_path),
        "output_dir": os.path.abspath(output_path),
        "terminal_log": os.path.abspath(log_path),
        "metadata_json": metadata_path,
    },
    "prompt": prompt,
    "actions": [item for item in pose.split(",") if item],
    "model": {
        "model_path": os.path.abspath(model_path),
        "ar_action_model_path": os.path.abspath(ar_action_model_path),
        "bi_action_model_path": os.path.abspath(bi_action_model_path),
        "ar_distill_action_model_path": os.path.abspath(ar_distill_action_model_path),
        "model_type": "ar",
    },
    "parameters": {
        "seed": int(seed),
        "aspect_ratio": aspect_ratio,
        "resolution": resolution,
        "pose": pose,
        "num_frames": int(num_frames),
        "width": int(width),
        "height": int(height),
        "n_inference_gpu": int(n_inference_gpu),
        "rewrite": rewrite.lower() == "true",
        "enable_sr": enable_sr.lower() == "true",
        "few_step": False,
        "use_vae_parallel": True,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "t2v_rewrite_base_url": os.environ.get("T2V_REWRITE_BASE_URL", ""),
        "i2v_rewrite_base_url": os.environ.get("I2V_REWRITE_BASE_URL", ""),
        "t2v_rewrite_model_name": os.environ.get("T2V_REWRITE_MODEL_NAME", ""),
        "i2v_rewrite_model_name": os.environ.get("I2V_REWRITE_MODEL_NAME", ""),
    },
}
with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
PY

  # inference with autoregressive model
  torchrun --master_port=29511 --nproc_per_node=$N_INFERENCE_GPU hyvideo/generate.py  \
    --prompt "$PROMPT" \
    --image_path "$IMAGE_PATH" \
    --resolution $RESOLUTION \
    --aspect_ratio $ASPECT_RATIO \
    --video_length $NUM_FRAMES \
    --seed $SEED \
    --rewrite $REWRITE \
    --sr $ENABLE_SR --save_pre_sr_video \
    --pose "$POSE" \
    --with-ui true \
    --output_path "$OUTPUT_PATH" \
    --model_path $MODEL_PATH \
    --action_ckpt $AR_ACTION_MODEL_PATH \
    --few_step false \
    --width $WIDTH \
    --height $HEIGHT \
    --model_type 'ar' \
    --use_vae_parallel false \
    --transformer_resident_ar_rollout false \
    --offloading \
    2>&1 | tee "$LOG_PATH"

  if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    echo "[ERROR] Inference failed: $IMAGE_PATH"
    python - "$METADATA_PATH" <<'PY'
import json
import sys
import time

metadata_path = sys.argv[1]
with open(metadata_path, "r", encoding="utf-8") as f:
    data = json.load(f)
data["status"] = "failed"
data["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())
with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
PY
    FAILED_IMAGES+=("$IMAGE_PATH")
  else
    echo "[INFO] Finished inference: $IMAGE_PATH"
    python - "$METADATA_PATH" "$OUTPUT_PATH" <<'PY'
import json
import os
import sys
import time

metadata_path = sys.argv[1]
output_dir = sys.argv[2]
with open(metadata_path, "r", encoding="utf-8") as f:
    data = json.load(f)
video_files = []
for root, _, files in os.walk(output_dir):
    for name in files:
        if name.lower().endswith((".mp4", ".mov", ".avi", ".webm")):
            video_files.append(os.path.abspath(os.path.join(root, name)))
data["status"] = "finished"
data["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())
data["server_paths"]["output_videos"] = sorted(video_files)
with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
PY
  fi
done

if (( ${#FAILED_IMAGES[@]} > 0 )); then
  echo "[ERROR] Failed images:"
  printf '  %s\n' "${FAILED_IMAGES[@]}"
  exit 1
fi

# inference with autoregressive distilled model
# torchrun --master_port=29511 --nproc_per_node=$N_INFERENCE_GPU hyvideo/generate.py \
#   --prompt "$PROMPT" \
#   --image_path $IMAGE_PATH \
#   --resolution $RESOLUTION \
#   --aspect_ratio $ASPECT_RATIO \
#   --video_length $NUM_FRAMES \
#   --seed $SEED \
#   --rewrite $REWRITE \
#   --sr $ENABLE_SR --save_pre_sr_video \
#   --pose "$POSE" \
#   --with-ui true \
#   --output_path $OUTPUT_PATH \
#   --model_path $MODEL_PATH \
#   --action_ckpt $AR_DISTILL_ACTION_MODEL_PATH \
#   --few_step true \
#   --num_inference_steps 4 \
#   --model_type 'ar' \
#   --use_vae_parallel false \
#   --use_sageattn false \
#   --use_fp8_gemm false \
#   --transformer_resident_ar_rollout true \
#   2>&1 | tee "$LOG_PATH"
