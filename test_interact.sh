export PYTHONPATH=$(cd "$(dirname "$0")" && pwd):$PYTHONPATH

# export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=2,5

# 设置 vLLM 服务地址（在本机运行）
export T2V_REWRITE_BASE_URL="http://localhost:8000/v1"
export I2V_REWRITE_BASE_URL="http://localhost:8000/v1"

# 设置模型名称（使用你启动时的模型路径）
export T2V_REWRITE_MODEL_NAME="/data3/dulingyi/worldmodel/models/Qwen3.5-9B/Qwen/Qwen3.5-9B"
export I2V_REWRITE_MODEL_NAME="/data3/dulingyi/worldmodel/models/Qwen3.5-9B/Qwen/Qwen3.5-9B"

# PROMPT='A paved pathway leads towards a stone arch bridge spanning a calm body of water.  Lush green trees and foliage line the path and the far bank of the water. A traditional-style pavilion with a tiered, reddish-brown roof sits on the far shore. The water reflects the surrounding greenery and the sky.  The scene is bathed in soft, natural light, creating a tranquil and serene atmosphere. The pathway is composed of large, rectangular stones, and the bridge is constructed of light gray stone.  The overall composition emphasizes the peaceful and harmonious nature of the landscape.'
PROMPT="A wide, paved pathway made of large, irregular rectangular stones leads toward a rustic stable built from rough-hewn stone and sturdy wooden beams. A majestic brown horse stands inside the open-air stall. To the right, the path opens up to a vast, rolling green meadow under a clear blue sky with soft, wispy clouds. In the center, a blonde hero stands with their back to the viewer, wearing a bright blue tunic and carrying a circular shield with a gold emblem. The scene is bathed in bright, midday sunlight, evoking a sense of adventure and peaceful exploration. The overall composition emphasizes the harmony between the rugged architecture and the expansive, natural landscape of a fantasy world."
# PROMPT="A sleek, neon-lit metallic walkway leads toward a high-tech charging station built from dark steel, glowing blue circuits, and glass panels. A metallic, robotic synthetic horse stands inside the illuminated bay. To the right, the path opens up to a sprawling futuristic cityscape with towering skyscrapers, holographic billboards, and flying vehicles under a deep purple night sky. In the center, a cyborg wanderer stands with their back to the viewer, wearing a tactical dark suit with glowing neon accents and carrying a semi-transparent hexagonal energy shield. The scene is bathed in the moody glow of neon lights, evoking a sense of gritty, futuristic exploration. The overall composition emphasizes the contrast between advanced technology and a neon-drenched cyberpunk metropolis"


IMAGE_PATH=./assets/img/1.png # Now we only provide the i2v model, so the path cannot be None
IMAGE_PATH_TAG="${IMAGE_PATH#./}"
IMAGE_PATH_TAG="${IMAGE_PATH_TAG//\//-}"
OUTPUT_PATH=../outputs/test_${IMAGE_PATH_TAG}_$(date +%Y%m%d_%H%M%S)

SEED=1
ASPECT_RATIO=16:9
RESOLUTION=480p # Now we only provide the 480p model
MODEL_PATH=../ckpts/HunyuanVideo-1.5                   # Path to pretrained hunyuanvideo-1.5 model
# AR_ACTION_MODEL_PATH=../ckpts/HY-WorldPlay/ar_model/diffusion_pytorch_model.safetensors         # Path to our HY-World 1.5 autoregressive checkpoints
AR_ACTION_MODEL_PATH=../ckpts/HY-WorldPlay/ar_rl_model/diffusion_pytorch_model.safetensors
BI_ACTION_MODEL_PATH=../ckpts/HY-WorldPlay/bidirectional_model/diffusion_pytorch_model.safetensors         # Path to our HY-World 1.5 bidirectional checkpoints
AR_DISTILL_ACTION_MODEL_PATH=../ckpts/HY-WorldPlay/ar_distilled_action_model/diffusion_pytorch_model.safetensors # Path to our HY-World 1.5 autoregressive distilled checkpoints
POSE='w-20,left-4,w-31'                   # Camera trajectory: pose string (e.g., 'w-31' means generating [1 + 31] latents) or JSON file path
# POSE='w-55'
NUM_FRAMES=221
WIDTH=832
HEIGHT=480

# Configuration for faster inference
# The maximum number recommended is 8.
N_INFERENCE_GPU=2 # Parallel inference GPU count.

# Configuration for better quality
REWRITE=true   # Enable prompt rewriting. Please ensure rewrite vLLM server is deployed and configured.
ENABLE_SR=false # Enable super resolution. When the NUM_FRAMES == 125, you can set it to true

mkdir -p "$OUTPUT_PATH"
LOG_PATH="$OUTPUT_PATH/terminal_$(date +%Y%m%d_%H%M%S).log"
set -o pipefail

# ============================================================
# Interactive mode: enter pose commands at runtime
# Each round generates a small number of frames (e.g., 13 frames = 4 latents)
# Commands: w(forward), s(backward), a(left), d(right),
#           up(pitch up), down(pitch down), left(yaw left), right(yaw right)
# Type 'quit' to exit, 'save' to save current video
# ============================================================
INTERACTIVE_VIDEO_LENGTH=13  # (4 latents - 1) * 4 + 1 = 13 frames per round
INITIAL_POSE='w-4'           # Initial pose to move camera forward before first round
WEB_PORT=7860

# inference with autoregressive model (interactive mode)
torchrun --nproc_per_node=$N_INFERENCE_GPU hyvideo/generate.py  \
  --prompt "$PROMPT" \
  --image_path "$IMAGE_PATH" \
  --resolution $RESOLUTION \
  --aspect_ratio $ASPECT_RATIO \
  --seed $SEED \
  --rewrite false \
  --sr false \
  --output_path $OUTPUT_PATH \
  --model_path $MODEL_PATH \
  --action_ckpt $AR_ACTION_MODEL_PATH \
  --few_step false \
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
  --vlm_prompt_cache true \
  --vlm_prompt_target_offset 1 \
  --use_vae_parallel false \
  --use_sageattn false \
  --use_fp8_gemm false \
  --transformer_resident_ar_rollout true \
  2>&1 | tee "$LOG_PATH"

# inference with autoregressive distilled model (interactive mode)
# torchrun --nproc_per_node=$N_INFERENCE_GPU hyvideo/generate.py  \
#   --prompt "$PROMPT" \
#   --image_path "$IMAGE_PATH" \
#   --resolution $RESOLUTION \
#   --aspect_ratio $ASPECT_RATIO \
#   --seed $SEED \
#   --rewrite false \
#   --sr false \
#   --output_path $OUTPUT_PATH \
#   --model_path $MODEL_PATH \
#   --action_ckpt $AR_DISTILL_ACTION_MODEL_PATH \
#   --few_step true \
#   --num_inference_steps 4 \
#   --width $WIDTH \
#   --height $HEIGHT \
#   --model_type 'ar' \
#   --interactive true \
#   --interactive_video_length $INTERACTIVE_VIDEO_LENGTH \
#   --initial_pose "$INITIAL_POSE" \
#   --display_window false \
#   --web true \
#   --web_host 0.0.0.0 \
#   --web_port $WEB_PORT \
#   --use_vae_parallel false \
#   --use_sageattn false \
#   --use_fp8_gemm false \
#   --transformer_resident_ar_rollout true \
#   2>&1 | tee "$LOG_PATH"

# ============================================================
# Original batch mode (commented out)
# ============================================================

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

# inference with autoregressive model (batch mode)
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
#   --with-ui true \
#   --output_path $OUTPUT_PATH \
#   --model_path $MODEL_PATH \
#   --action_ckpt $AR_ACTION_MODEL_PATH \
#   --few_step false \
#   --width $WIDTH \
#   --height $HEIGHT \
#   --model_type 'ar' \
#   2>&1 | tee "$LOG_PATH"

# inference with autoregressive distilled model
# torchrun --nproc_per_node=$N_INFERENCE_GPU hyvideo/generate.py \
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
