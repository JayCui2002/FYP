# Start

## 1. Local backend with Docker

```powershell
cd C:\path\to\fyp_2026

$env:STREAM_PROMPT="your prompt here"
$env:STREAM_STEP="2"
$env:STREAM_HEIGHT="384"
$env:STREAM_WIDTH="384"
$env:STREAM_CKPT_ROOT="C:\path\to\ckpts"
$env:STREAM_WAN_MODELS_ROOT="C:\path\to\wan_models"

# super-resolution
# $env:STREAM_USE_SR="1"

# start backend
docker compose -f docker-compose.backend.yml up --build
```

## 2. Local frontend

```powershell
# keep input/scene.cfg as:
# stream_v2_autostart=false
# stream_v2_host=127.0.0.1
# stream_v2_port=8765

cd C:\path\to\fyp_2026
.\fyp.exe
```

## 3. Remote backend on Linux

```bash
# edit input/scene.cfg on Windows first:
# stream_v2_autostart=false
# stream_v2_host=<remote-host-or-ip>
# stream_v2_port=8765

cd /path/to/StreamDiffusionV2
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate <env-name>
python streamdiffusionv2_bridge.py \
  --host 0.0.0.0 \
  --port 8765 \
  --real-v2 \
  --assets-root /path/to/StreamDiffusionV2 \
  --config-path input/streamv2v_config.yaml \
  --checkpoint-folder /path/to/ckpts/wan_causal_dmd_v2v \
  --height 384 \
  --width 384 \
  --step 2 \
  --noise-scale 0.85 \
  --prompt "your prompt here"
```

## 4. Stop Docker backend

```powershell
cd C:\path\to\fyp_2026
docker compose -f docker-compose.backend.yml down
```
