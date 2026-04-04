import argparse
import importlib
import os
import select
import socket
import struct
import sys
import time
import urllib.request
from types import SimpleNamespace
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

REQUEST_HEADER = struct.Struct("<4sIIII")
RESPONSE_HEADER = struct.Struct("<4sIIIII")
REQUEST_MAGIC = b"FRM0"
RESPONSE_MAGIC = b"OUT0"
RESPONSE_FLAG_STYLIZED_READY = 1
RESPONSE_FLAG_REUSE_LATEST = 2
DEFAULT_SR_MODEL = "realesrgan-x2plus"
DEFAULT_SR_X2_URL = "https://huggingface.co/2kpr/Real-ESRGAN/resolve/main/RealESRGAN_x2plus.pth"


def log(message: str):
    return


def log_ready(seq: int, reuse_latest: bool):
    print(f"[bridge] ready seq={seq} reuse={1 if reuse_latest else 0}", flush=True)


def setup_local_streamdiffusionv2_path():
    repo_root = Path(__file__).resolve().parent
    local_pkg_root = repo_root / "third_party" / "streamdiffusionv2"
    if local_pkg_root.exists():
        local_pkg_root_str = str(local_pkg_root)
        if local_pkg_root_str in sys.path:
            sys.path.remove(local_pkg_root_str)
        sys.path.insert(0, local_pkg_root_str)


def ensure_superres_compat():
    if "torchvision.transforms.functional_tensor" not in sys.modules:
        try:
            compat_module = importlib.import_module("torchvision.transforms._functional_tensor")
            sys.modules["torchvision.transforms.functional_tensor"] = compat_module
        except ModuleNotFoundError:
            pass


def resolve_superres_model_path(args):
    if args.sr_model_path:
        return args.sr_model_path.strip()

    if args.assets_root:
        model_dir = Path(args.assets_root).resolve() / "sr_models"
    else:
        model_dir = Path(__file__).resolve().parent / "input" / "sr_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "RealESRGAN_x2plus.pth"
    if not model_path.exists():
        log(f"Downloading 2x super resolution weights to: {model_path}")
        urllib.request.urlretrieve(DEFAULT_SR_X2_URL, model_path)
    return str(model_path)


def build_superres(args, torch_module, runtime_device):
    if not args.sr:
        return None
    if int(args.sr_scale) != 2:
        raise RuntimeError(f"only 2x super resolution is supported right now, got: {args.sr_scale}")
    if args.sr_model != DEFAULT_SR_MODEL:
        raise RuntimeError(f"unsupported super resolution model: {args.sr_model}")

    log("Loading 2x super resolution backend.")
    ensure_superres_compat()
    RRDBNet = importlib.import_module("basicsr.archs.rrdbnet_arch").RRDBNet
    RealESRGANer = importlib.import_module("realesrgan").RealESRGANer

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    model_path = resolve_superres_model_path(args)
    upsampler = RealESRGANer(
        scale=2,
        model_path=model_path,
        model=model,
        tile=max(0, int(args.sr_tile)),
        tile_pad=10,
        pre_pad=0,
        half=(runtime_device.type == "cuda"),
        device=runtime_device,
    )
    log("2x super resolution backend is ready.")
    return {
        "backend": DEFAULT_SR_MODEL,
        "upsampler": upsampler,
        "scale": 2,
        "enabled": True,
        "output_count": 0,
    }


def maybe_apply_superres_rgb_array(engine, rgb_array):
    if rgb_array is None:
        return None
    if not isinstance(engine, dict):
        return rgb_array
    superres = engine.get("superres")
    if not superres or not superres.get("enabled"):
        return rgb_array

    rgb = np.ascontiguousarray(rgb_array[:, :, :3])
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    output_bgr, _ = superres["upsampler"].enhance(bgr, outscale=superres["scale"])
    output_rgb = np.ascontiguousarray(output_bgr[:, :, ::-1])
    superres["output_count"] += 1
    return output_rgb


def build_engine(model_id: str, prompt: str, negative_prompt: str, device: str, args):
    del model_id
    del negative_prompt
    if args.assets_root:
        os.environ["STREAMDIFFUSIONV2_ROOT"] = str(Path(args.assets_root).resolve())
    setup_local_streamdiffusionv2_path()
    if args.real_v2:
        config_path = Path(args.config_path)
        if not config_path.exists():
            raise RuntimeError(f"real-v2 config not found: {config_path}")
        checkpoint_model = Path(args.checkpoint_folder) / "model.pt"
        if not checkpoint_model.exists():
            raise RuntimeError(f"real-v2 checkpoint not found: {checkpoint_model}")
        torch = importlib.import_module("torch")
        merge_cli_config = importlib.import_module("streamv2v.inference_common").merge_cli_config
        inference_module_name = "streamv2v.inference"
        inference_module = importlib.import_module(inference_module_name)
        SingleGPUInferencePipeline = inference_module.SingleGPUInferencePipeline
        set_seed = importlib.import_module("models.util").set_seed

        config_overlay = SimpleNamespace(
            step=args.step,
            height=args.height,
            width=args.width,
            fps=int(max(1.0, args.fps)),
            model_type=args.model_type,
            num_frames=args.num_frames,
            t2v=False,
            profile=False,
            target_fps=None,
            fixed_noise_scale=False,
            use_taehv=args.use_taehv,
        )
        if not args.input and getattr(args, "use_taehv", False):
            log("Ignoring --use-taehv for socket realtime mode and using Wan VAE decode.")
            args.use_taehv = False
            config_overlay.use_taehv = False
        config = merge_cli_config(str(config_path), config_overlay)
        set_seed(args.seed)
        if device:
            runtime_device = torch.device(device)
        else:
            runtime_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        manager = SingleGPUInferencePipeline(config, runtime_device)
        manager.load_model(str(Path(args.checkpoint_folder)))
        superres = build_superres(args, torch, runtime_device)
        if args.stream_wo_batch:
            log("Ignoring --stream-wo-batch for socket realtime mode and using inference_stream.")
        chunk_size = manager.base_chunk_size * manager.pipeline.num_frame_per_block
        priming_frames = 1 + chunk_size
        return {
            "backend": "streamv2v_real",
            "torch": torch,
            "manager": manager,
            "compute_noise_scale_and_step": inference_module.compute_noise_scale_and_step,
            "device": runtime_device,
            "prompt": prompt,
            "noise_scale": float(args.noise_scale),
            "init_noise_scale": float(args.noise_scale),
            "base_chunk_size": int(manager.base_chunk_size),
            "chunk_size": int(chunk_size),
            "priming_frames": int(priming_frames),
            "frame_buffer": [],
            "initialized": False,
            "current_start": 0,
            "current_end": 0,
            "last_input_tensor": None,
            "last_output": None,
            "num_steps": int(len(manager.pipeline.denoising_step_list)),
            "height": int(args.height),
            "width": int(args.width),
            "t_refresh": int(getattr(manager, "t_refresh", 50)),
            "superres": superres,
            "received_requests": 0,
            "sent_responses": 0,
            "sent_ready_responses": 0,
            "sent_reuse_responses": 0,
            "sent_intermediate_responses": 0,
            "produced_outputs": 0,
            "pending_boundary_requests": 0,
            "debug_log_interval": 16,
            "cached_ready_frame": None,
            "cached_ready_size": None,
        }

    inference_module_name = "streamv2v.inference"
    module = importlib.import_module(inference_module_name)
    fallback_engine = {"backend": "streamv2v", "module": module}
    if args.sr:
        torch = importlib.import_module("torch")
        runtime_device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fallback_engine["superres"] = build_superres(args, torch, runtime_device)
    return fallback_engine


def _frames_to_tensor(frame_list, torch, device):
    arr = np.stack(frame_list, axis=0).astype(np.float32)
    arr = arr / 127.5 - 1.0
    tensor = torch.from_numpy(arr).permute(3, 0, 1, 2).unsqueeze(0).contiguous()
    return tensor.to(device=device, dtype=torch.bfloat16)


def _decode_last_frame_to_image(video_np):
    if video_np is None:
        return None
    if video_np.ndim != 4 or video_np.shape[-1] != 3:
        return None
    frame = video_np[-1]
    frame_u8 = np.clip(frame * 255.0, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(frame_u8, mode="RGB")


def output_to_rgb_array(output):
    if isinstance(output, Image.Image):
        return np.array(output.convert("RGB"), dtype=np.uint8)
    if isinstance(output, np.ndarray):
        arr = output
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        return arr
    if isinstance(output, (list, tuple)) and output:
        return output_to_rgb_array(output[0])
    return None


def recv_exact(sock, size: int):
    chunks = bytearray()
    while len(chunks) < size:
        block = sock.recv(size - len(chunks))
        if not block:
            return None
        chunks.extend(block)
    return bytes(chunks)


def _read_request(conn):
    header = recv_exact(conn, REQUEST_HEADER.size)
    if header is None:
        return None
    magic, width, height, payload_size, seq = REQUEST_HEADER.unpack(header)
    if magic != REQUEST_MAGIC:
        raise ConnectionError("invalid request header")
    payload = recv_exact(conn, payload_size)
    if payload is None:
        return None
    frame = np.frombuffer(payload, dtype=np.uint8)
    if frame.size != width * height * 3:
        raise ConnectionError("invalid frame payload size")
    image = Image.fromarray(frame.reshape((height, width, 3)), mode="RGB")
    return {
        "seq": seq,
        "image": image,
        "rgb": np.array(image, dtype=np.uint8),
        "width": width,
        "height": height,
    }


def _max_batch_requests(engine):
    if not engine["initialized"]:
        return int(engine["priming_frames"])
    return int(engine["chunk_size"])


def _drain_request_batch(conn, max_requests):
    requests = []
    first = _read_request(conn)
    if first is None:
        return requests
    requests.append(first)

    while len(requests) < max_requests:
        ready, _, _ = select.select([conn], [], [], 0.0)
        if not ready:
            break
        req = _read_request(conn)
        if req is None:
            break
        requests.append(req)

    return requests


def _consume_frame_batch(engine, count):
    batch = engine["frame_buffer"][:count]
    del engine["frame_buffer"][:count]
    return batch


def _initialize_stream_engine(engine, prompt):
    manager = engine["manager"]
    inp = _frames_to_tensor(_consume_frame_batch(engine, engine["priming_frames"]), engine["torch"], engine["device"])
    manager.pipeline.vae.model.first_encode = True
    manager.pipeline.vae.model.first_decode = True
    manager.pipeline.kv_cache1 = None
    manager.pipeline.crossattn_cache = None
    manager.pipeline.block_x = None
    manager.pipeline.hidden_states = None
    manager.processed = 0
    current_start = 0
    current_end = manager.pipeline.frame_seq_length * (1 + engine["chunk_size"] // engine["base_chunk_size"])
    noisy_latents = manager._encode_noisy_latents(inp, float(engine["noise_scale"]))
    with engine["torch"].inference_mode():
        denoised_pred = manager.prepare_pipeline(
            text_prompts=[prompt or engine["prompt"]],
            noise=noisy_latents,
            current_start=current_start,
            current_end=current_end,
        )
        video_np = manager._decode_video_array(denoised_pred, last_frame_only=False)
    out_img = _decode_last_frame_to_image(video_np)
    engine["initialized"] = True
    engine["current_start"] = current_end
    engine["current_end"] = current_end + (engine["chunk_size"] // engine["base_chunk_size"]) * manager.pipeline.frame_seq_length
    engine["last_input_tensor"] = inp[:, :, [-1]].clone()
    if out_img is not None:
        engine["last_output"] = out_img
    engine["produced_outputs"] += 1
    return out_img


def _step_stream_engine(engine):
    manager = engine["manager"]
    inp = _frames_to_tensor(_consume_frame_batch(engine, engine["chunk_size"]), engine["torch"], engine["device"])
    noise_scale, current_step = engine["compute_noise_scale_and_step"](
        engine["torch"].cat([engine["last_input_tensor"], inp], dim=2),
        engine["chunk_size"] + 1,
        engine["chunk_size"],
        float(engine["noise_scale"]),
        float(engine["init_noise_scale"]),
    )
    if engine["current_start"] // manager.pipeline.frame_seq_length >= engine["t_refresh"]:
        engine["current_start"] = manager.pipeline.kv_cache_length - manager.pipeline.frame_seq_length
        engine["current_end"] = engine["current_start"] + (engine["chunk_size"] // engine["base_chunk_size"]) * manager.pipeline.frame_seq_length
    noisy_latents = manager._encode_noisy_latents(inp, noise_scale)
    with engine["torch"].inference_mode():
        denoised_pred = manager.pipeline.inference_stream(
            noise=noisy_latents[:, -1].unsqueeze(1),
            current_start=engine["current_start"],
            current_end=engine["current_end"],
            current_step=current_step,
        )
    manager.processed += 1
    out_img = None
    if manager.processed >= engine["num_steps"]:
        video_np = manager._decode_video_array(denoised_pred, last_frame_only=True)
        out_img = _decode_last_frame_to_image(video_np)
    engine["current_start"] = engine["current_end"]
    engine["current_end"] += (engine["chunk_size"] // engine["base_chunk_size"]) * manager.pipeline.frame_seq_length
    engine["last_input_tensor"] = inp[:, :, [-1]].clone()
    engine["noise_scale"] = noise_scale
    if out_img is not None:
        engine["last_output"] = out_img
        engine["produced_outputs"] += 1
    return out_img


def _process_ready_outputs(engine, prompt):
    outputs = []
    if not engine["initialized"] and len(engine["frame_buffer"]) >= engine["priming_frames"]:
        outputs.append(_initialize_stream_engine(engine, prompt))
    while engine["initialized"] and len(engine["frame_buffer"]) >= engine["chunk_size"]:
        outputs.append(_step_stream_engine(engine))
    return outputs


def _cache_ready_frame(engine, output):
    out_frame = output_to_rgb_array(output)
    if out_frame is None:
        return None
    out_frame = maybe_apply_superres_rgb_array(engine, out_frame)
    out_frame = np.ascontiguousarray(out_frame[:, :, :3])
    engine["cached_ready_frame"] = out_frame
    engine["cached_ready_size"] = (out_frame.shape[1], out_frame.shape[0])
    return out_frame


def _refresh_pending(engine):
    if not engine.get("initialized"):
        return False
    frame_seq_length = int(engine["manager"].pipeline.frame_seq_length)
    return int(engine["current_start"]) // frame_seq_length >= int(engine["t_refresh"])


def _send_response(conn, engine, request, output=None, stylized_ready=False, reuse_latest=False):
    response = b""
    out_w, out_h = request["width"], request["height"]
    if stylized_ready and not reuse_latest and output is not None:
        out_frame = _cache_ready_frame(engine, output)
        if out_frame is None:
            stylized_ready = False
        else:
            out_h, out_w = out_frame.shape[:2]
            response = out_frame.tobytes()
    elif stylized_ready and reuse_latest:
        cached_size = engine.get("cached_ready_size")
        if cached_size is not None:
            out_w, out_h = cached_size
        else:
            stylized_ready = False
            reuse_latest = False
    flags = 0
    if stylized_ready:
        flags |= RESPONSE_FLAG_STYLIZED_READY
    if reuse_latest:
        flags |= RESPONSE_FLAG_REUSE_LATEST
    conn.sendall(RESPONSE_HEADER.pack(RESPONSE_MAGIC, out_w, out_h, len(response), flags, request["seq"]))
    if response:
        conn.sendall(response)
    engine["sent_responses"] += 1
    if stylized_ready:
        engine["sent_ready_responses"] += 1
        log_ready(request["seq"], reuse_latest)
    if reuse_latest:
        engine["sent_reuse_responses"] += 1


def _send_intermediate_response(conn, engine, request):
    engine["sent_intermediate_responses"] += 1
    if _refresh_pending(engine):
        _send_response(conn, engine, request, stylized_ready=False, reuse_latest=False)
        return
    if engine.get("cached_ready_frame") is not None:
        _send_response(conn, engine, request, stylized_ready=True, reuse_latest=True)
        return
    _send_response(conn, engine, request, stylized_ready=False, reuse_latest=False)


def serve_socket(engine, args):
    log(f"Starting socket server on {args.host}:{args.port}.")
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)
    log("Waiting for OpenGL client connection.")
    conn, _ = server.accept()
    conn.settimeout(10.0)
    log("OpenGL client connected.")
    boundary_requests = []
    with conn:
        while True:
            try:
                requests = _drain_request_batch(conn, _max_batch_requests(engine))
                if not requests:
                    break
            except (OSError, ConnectionError):
                break

            engine["received_requests"] += len(requests)
            classified_initialized = engine["initialized"]
            fill = len(engine["frame_buffer"])
            try:
                for request in requests:
                    resized = request["image"].convert("RGB").resize((engine["width"], engine["height"]), Image.Resampling.BILINEAR)
                    engine["frame_buffer"].append(np.array(resized, dtype=np.uint8))

                    if not classified_initialized:
                        fill += 1
                        if fill == engine["priming_frames"]:
                            boundary_requests.append(request)
                            engine["pending_boundary_requests"] = len(boundary_requests)
                            classified_initialized = True
                            fill = 0
                        else:
                            _send_intermediate_response(conn, engine, request)
                    else:
                        fill += 1
                        if fill == engine["chunk_size"]:
                            boundary_requests.append(request)
                            engine["pending_boundary_requests"] = len(boundary_requests)
                            fill = 0
                        else:
                            _send_intermediate_response(conn, engine, request)

                outputs = _process_ready_outputs(engine, args.prompt)
                for output in outputs:
                    if not boundary_requests:
                        break
                    request = boundary_requests.pop(0)
                    engine["pending_boundary_requests"] = len(boundary_requests)
                    if output is not None:
                        _send_response(conn, engine, request, output, stylized_ready=True, reuse_latest=False)
                    else:
                        _send_response(conn, engine, request, stylized_ready=False, reuse_latest=False)
            except (OSError, ConnectionError):
                break
    server.close()


def run_engine(engine, image: Image.Image, prompt: str, negative_prompt: str):
    if isinstance(engine, dict) and engine.get("backend") == "streamv2v_real":
        resized = image.convert("RGB").resize((engine["width"], engine["height"]), Image.Resampling.BILINEAR)
        frame = np.array(resized, dtype=np.uint8)
        engine["frame_buffer"].append(frame)

        if not engine["initialized"]:
            if len(engine["frame_buffer"]) < engine["priming_frames"]:
                return engine["last_output"], False
            out_img = _initialize_stream_engine(engine, prompt)
            return out_img if out_img is not None else engine["last_output"], out_img is not None

        if len(engine["frame_buffer"]) < engine["chunk_size"]:
            return engine["last_output"], False

        out_img = _step_stream_engine(engine)
        return out_img if out_img is not None else engine["last_output"], out_img is not None

    if isinstance(engine, dict) and engine.get("backend") == "streamv2v":
        base = np.array(image.convert("RGB"), dtype=np.float32)
        quant = np.floor(base / 32.0) * 32.0
        edges = np.array(image.filter(ImageFilter.FIND_EDGES).convert("RGB"), dtype=np.float32)
        stylized = np.clip(0.85 * quant + 0.35 * (255.0 - edges), 0.0, 255.0).astype(np.uint8)
        return Image.fromarray(stylized, mode="RGB"), True
    arr = np.array(image)
    candidates = ("infer", "generate", "run", "step", "__call__")
    for name in candidates:
        fn = getattr(engine, name, None)
        if callable(fn):
            for payload in (image, arr):
                try:
                    return fn(payload, prompt=prompt, negative_prompt=negative_prompt), True
                except TypeError:
                    try:
                        return fn(payload, prompt=prompt), True
                    except TypeError:
                        try:
                            return fn(payload), True
                        except Exception:
                            pass
                except Exception:
                    pass
    raise RuntimeError("no callable inference method on streamdiffusionv2 engine")


def save_output(output, output_path: Path):
    if output_path is None:
        return
    if isinstance(output, Image.Image):
        output.save(output_path)
        return
    if isinstance(output, np.ndarray):
        arr = output
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        Image.fromarray(arr).save(output_path)
        return
    if isinstance(output, (list, tuple)) and output:
        first = output[0]
        if isinstance(first, Image.Image):
            first.save(output_path)
            return
        if isinstance(first, np.ndarray):
            arr = first
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            Image.fromarray(arr).save(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--model-id", default="")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--ready-flag", default="")
    parser.add_argument("--real-v2", action="store_true")
    parser.add_argument("--stream-wo-batch", action="store_true")
    parser.add_argument("--config-path", default="")
    parser.add_argument("--checkpoint-folder", default="")
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--noise-scale", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--model-type", default="T2V-1.3B")
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--use-taehv", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--assets-root", default="")
    parser.add_argument("--sr", action="store_true")
    parser.add_argument("--sr-scale", type=int, default=2)
    parser.add_argument("--sr-model", default=DEFAULT_SR_MODEL)
    parser.add_argument("--sr-model-path", default="")
    parser.add_argument("--sr-tile", type=int, default=0)
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else None
    output_path = Path(args.output) if args.output else None
    ready_flag_path = Path(args.ready_flag) if args.ready_flag else None
    interval = 1.0 / args.fps if args.fps > 0 else 0.1

    if args.real_v2:
        if not args.config_path or not args.checkpoint_folder:
            raise RuntimeError("--real-v2 requires --config-path and --checkpoint-folder")
    engine = build_engine(args.model_id, args.prompt, args.negative_prompt, args.device, args)
    if ready_flag_path is not None:
        ready_flag_path.parent.mkdir(parents=True, exist_ok=True)
        ready_flag_path.write_text("ready\n", encoding="utf-8")
    if not args.input:
        serve_socket(engine, args)
        return
    last_mtime = -1.0

    while True:
        try:
            if input_path is not None and input_path.exists():
                mtime = input_path.stat().st_mtime
                if mtime > last_mtime:
                    img = Image.open(input_path).convert("RGB")
                    out, _ = run_engine(engine, img, args.prompt, args.negative_prompt)
                    out_array = output_to_rgb_array(out)
                    if out_array is not None:
                        out = maybe_apply_superres_rgb_array(engine, out_array)
                    save_output(out, output_path)
                    last_mtime = mtime
                    if args.once:
                        break
            time.sleep(interval)
        except KeyboardInterrupt:
            break
        except Exception:
            time.sleep(interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception:
        pass
