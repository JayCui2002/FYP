import argparse
import importlib
import io
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
RESPONSE_HEADER = struct.Struct("<4sIIIIIIII")
REQUEST_MAGIC = b"FRM0"
RESPONSE_MAGIC = b"OUT0"
RESPONSE_FLAG_STYLIZED_READY = 1
RESPONSE_FLAG_REUSE_LATEST = 2
DEFAULT_SR_MODEL = "realesrgan-x2plus"
DEFAULT_SR_X2_URL = "https://huggingface.co/2kpr/Real-ESRGAN/resolve/main/RealESRGAN_x2plus.pth"
DEFAULT_METRICS_OUTPUT = "stream_metrics.txt"

# Print ready line
def log_ready(seq: int, reuse_latest: bool):
    print(f"[bridge] ready seq={seq} reuse={1 if reuse_latest else 0}", flush=True)


# Save stream stats
def write_metrics(engine):
    if not isinstance(engine, dict):
        return
    path = Path(__file__).resolve().parent / DEFAULT_METRICS_OUTPUT
    first_request_time = engine.get("first_request_time")
    first_output_time = engine.get("first_output_time")
    last_output_time = engine.get("last_output_time")
    metrics_output_count = int(engine.get("metrics_output_count", 0))
    if (
        first_request_time is not None
        and first_output_time is not None
        and first_output_time >= first_request_time
    ):
        first_frame_time_ms = (first_output_time - first_request_time) * 1000.0
    else:
        first_frame_time_ms = 0.0
    if (
        first_output_time is not None
        and last_output_time is not None
        and last_output_time > first_output_time
        and metrics_output_count > 1
    ):
        average_fps = (metrics_output_count - 1) / (last_output_time - first_output_time)
    else:
        average_fps = 0.0
    first_ready_time = engine.get("first_ready_time")
    last_ready_time = engine.get("last_ready_time")
    ready_response_count = int(engine.get("ready_response_count", 0))
    if (
        first_ready_time is not None
        and last_ready_time is not None
        and last_ready_time > first_ready_time
        and ready_response_count > 1
    ):
        display_fps = (ready_response_count - 1) / (last_ready_time - first_ready_time)
    else:
        display_fps = 0.0
    latency_count = int(engine.get("latency_count", 0))
    if latency_count > 0:
        average_latency_ms = float(engine.get("latency_sum_ms", 0.0)) / latency_count
    else:
        average_latency_ms = 0.0
    text = (
        f"first_frame_time_ms={first_frame_time_ms:.4f}\n"
        f"average_output_fps={average_fps:.4f}\n"
        f"display_fps={display_fps:.4f}\n"
        f"average_latency_ms={average_latency_ms:.4f}\n"
        f"output_frames={metrics_output_count}\n"
        f"ready_frames={ready_response_count}\n"
        f"latency_samples={latency_count}\n"
    )
    try:
        path.write_text(text, encoding="utf-8")
    except Exception:
        pass


# Extend module path
def setup_local_streamdiffusionv2_path():
    repo_root = Path(__file__).resolve().parent
    local_pkg_root = repo_root / "third_party" / "streamdiffusionv2"
    if local_pkg_root.exists():
        local_pkg_root_str = str(local_pkg_root)
        if local_pkg_root_str in sys.path:
            sys.path.remove(local_pkg_root_str)
        sys.path.insert(0, local_pkg_root_str)


# Patch sr imports
def ensure_superres_compat():
    if "torchvision.transforms.functional_tensor" not in sys.modules:
        try:
            compat_module = importlib.import_module("torchvision.transforms._functional_tensor")
            sys.modules["torchvision.transforms.functional_tensor"] = compat_module
        except ModuleNotFoundError:
            pass


# Resolve sr path
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
        urllib.request.urlretrieve(DEFAULT_SR_X2_URL, model_path)
    return str(model_path)


# Build sr backend
def build_superres(args, torch_module, runtime_device):
    if not args.sr:
        return None
    if int(args.sr_scale) != 2:
        raise RuntimeError()
    if args.sr_model != DEFAULT_SR_MODEL:
        raise RuntimeError()

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
    return {
        "upsampler": upsampler,
        "scale": 2,
        "enabled": True,
    }


# Apply sr frame
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
    return output_rgb


# Build inference engine
def build_engine(model_id: str, prompt: str, negative_prompt: str, device: str, args):
    del model_id
    del negative_prompt
    if args.assets_root:
        os.environ["STREAMDIFFUSIONV2_ROOT"] = str(Path(args.assets_root).resolve())
    setup_local_streamdiffusionv2_path()
    if args.real_v2:
        config_path = Path(args.config_path)
        if not config_path.exists():
            raise RuntimeError()
        checkpoint_model = Path(args.checkpoint_folder) / "model.pt"
        if not checkpoint_model.exists():
            raise RuntimeError()
        torch = importlib.import_module("torch")
        merge_cli_config = importlib.import_module("streamv2v.inference_common").merge_cli_config
        inference_module_name = "streamv2v.inference_wo_batch" if args.stream_wo_batch else "streamv2v.inference"
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
        base_chunk_size = manager.base_chunk_size * manager.pipeline.num_frame_per_block
        chunk_size = 4 * manager.pipeline.num_frame_per_block if args.stream_wo_batch else base_chunk_size
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
            "stream_wo_batch": bool(args.stream_wo_batch),
            "chunk_size": int(chunk_size),
            "priming_frames": int(priming_frames),
            "frame_buffer": [],
            "initialized": False,
            "session": None,
            "last_output": None,
            "num_steps": int(len(manager.pipeline.denoising_step_list)),
            "height": int(args.height),
            "width": int(args.width),
            "t_refresh": int(getattr(manager, "t_refresh", 50)),
            "superres": superres,
            "cached_stylized_frame": None,
            "cached_stylized_size": None,
            "cached_superres_frame": None,
            "cached_superres_size": None,
            "first_request_time": None,
            "first_output_time": None,
            "last_output_time": None,
            "first_ready_time": None,
            "last_ready_time": None,
            "ready_response_count": 0,
            "latency_sum_ms": 0.0,
            "latency_count": 0,
            "metrics_output_count": 0,
        }

    inference_module_name = "streamv2v.inference"
    module = importlib.import_module(inference_module_name)
    fallback_engine = {"backend": "streamv2v", "module": module}
    if args.sr:
        torch = importlib.import_module("torch")
        runtime_device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fallback_engine["superres"] = build_superres(args, torch, runtime_device)
    return fallback_engine


# Pack input batch
def make_input_batch(frames, torch, device, target_height=None, target_width=None):
    arr = np.stack(frames, axis=0)
    tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
    tensor = tensor.to(device=device, dtype=torch.float32)
    if (
        target_height is not None
        and target_width is not None
        and (tensor.shape[2] != target_height or tensor.shape[3] != target_width)
    ):
        tensor = torch.nn.functional.interpolate(
            tensor,
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        )
    tensor = tensor / 127.5 - 1.0
    return tensor.permute(1, 0, 2, 3).unsqueeze(0).contiguous().to(dtype=torch.bfloat16)


# Decode last frame
def decode_last_image(video_np):
    if video_np is None:
        return None
    if video_np.ndim != 4 or video_np.shape[-1] != 3:
        return None
    frame = video_np[-1]
    return np.clip(frame * 255.0, 0.0, 255.0).astype(np.uint8)


# Normalize output frame
def to_rgb_frame(output):
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
        return to_rgb_frame(output[0])
    return None


# Read exact bytes
def recv_exact(sock, size: int):
    chunks = bytearray()
    while len(chunks) < size:
        block = sock.recv(size - len(chunks))
        if not block:
            return None
        chunks.extend(block)
    return bytes(chunks)


# Decode request frame
def _read_request(conn):
    header = recv_exact(conn, REQUEST_HEADER.size)
    if header is None:
        return None
    magic, width, height, payload_size, seq = REQUEST_HEADER.unpack(header)
    if magic != REQUEST_MAGIC:
        raise ConnectionError()
    payload = recv_exact(conn, payload_size)
    if payload is None:
        return None
    if payload_size == width * height * 3:
        frame = np.frombuffer(payload, dtype=np.uint8)
        rgb = frame.reshape((height, width, 3)).copy()
    else:
        try:
            with Image.open(io.BytesIO(payload)) as image:
                image = image.convert("RGB")
                if image.size != (width, height):
                    image = image.resize((width, height), Image.BILINEAR)
                rgb = np.array(image, dtype=np.uint8)
        except Exception as exc:
            raise ConnectionError() from exc
    return {
        "seq": seq,
        "rgb": rgb,
        "width": width,
        "height": height,
        "received_time": time.perf_counter(),
    }


# Drain request batch
def _drain_request_batch(conn, limit):
    requests = []
    first = _read_request(conn)
    if first is None:
        return requests
    requests.append(first)

    while len(requests) < limit:
        ready, _, _ = select.select([conn], [], [], 0.0)
        if not ready:
            break
        req = _read_request(conn)
        if req is None:
            break
        requests.append(req)

    return requests


# Prime stream session
def start_real_stream(engine, prompt):
    manager = engine["manager"]
    frames = engine["frame_buffer"][:engine["priming_frames"]]
    del engine["frame_buffer"][:engine["priming_frames"]]
    inp = make_input_batch(frames, engine["torch"], engine["device"], engine["height"], engine["width"])
    with engine["torch"].inference_mode():
        session, initial_video = manager.start_stream_session(
            prompt or engine["prompt"],
            inp,
            float(engine["noise_scale"]),
        )
    out_img = decode_last_image(initial_video)
    engine["initialized"] = True
    engine["session"] = session
    engine["noise_scale"] = float(session.noise_scale)
    if out_img is not None:
        engine["last_output"] = out_img
    return out_img


# Advance stream step
def step_real_stream(engine):
    manager = engine["manager"]
    frames = engine["frame_buffer"][:engine["chunk_size"]]
    del engine["frame_buffer"][:engine["chunk_size"]]
    inp = make_input_batch(frames, engine["torch"], engine["device"], engine["height"], engine["width"])
    session = engine.get("session")
    if session is None:
        return None
    noise_scale, current_step = engine["compute_noise_scale_and_step"](
        engine["torch"].cat([session.last_image, inp], dim=2),
        engine["chunk_size"] + 1,
        engine["chunk_size"],
        float(session.noise_scale),
        float(session.init_noise_scale),
    )
    out_img = None
    if engine.get("stream_wo_batch"):
        if session.current_start // manager.pipeline.frame_seq_length >= engine["t_refresh"]:
            with engine["torch"].inference_mode():
                refreshed_session, refresh_video = manager._refresh_stream_session(session, inp)
            session = refreshed_session
            out_img = decode_last_image(refresh_video[[-1]] if refresh_video is not None else None)
        else:
            noisy_latents = manager._encode_noisy_latents(inp, noise_scale)
            with engine["torch"].inference_mode():
                denoised_pred = manager.pipeline.inference_wo_batch(
                    noise=noisy_latents,
                    current_start=session.current_start,
                    current_end=session.current_end,
                    current_step=current_step,
                )
            session.processed += 1
            manager.processed = session.processed
            video_np = manager._decode_video_array(denoised_pred, last_frame_only=True)
            out_img = decode_last_image(video_np)
            session.current_start = session.current_end
            session.current_end += (engine["chunk_size"] // 4) * manager.pipeline.frame_seq_length
            history_window = max(getattr(manager, "refresh_history_frames", 0), session.chunk_size + 1)
            history_images = getattr(session, "history_images", None)
            if history_images is None:
                history_images = session.last_image
            session.history_images = engine["torch"].cat([history_images, inp], dim=2)
            if session.history_images.shape[2] > history_window:
                session.history_images = session.history_images[:, :, -history_window:]
            session.last_image = inp[:, :, [-1]].clone()
            session.noise_scale = noise_scale
    elif session.current_start // manager.pipeline.frame_seq_length >= engine["t_refresh"]:
        with engine["torch"].inference_mode():
            refreshed_session, _ = manager._refresh_stream_session(session, inp)
        session = refreshed_session
        out_img = None
    else:
        noisy_latents = manager._encode_noisy_latents(inp, noise_scale)
        with engine["torch"].inference_mode():
            denoised_pred = manager.pipeline.inference_stream(
                noise=noisy_latents[:, -1].unsqueeze(1),
                current_start=session.current_start,
                current_end=session.current_end,
                current_step=current_step,
            )
        session.processed += 1
        manager.processed = session.processed
        if session.processed >= engine["num_steps"]:
            video_np = manager._decode_video_array(denoised_pred, last_frame_only=True)
            out_img = decode_last_image(video_np)
        session.current_start = session.current_end
        session.current_end += (session.chunk_size // manager.base_chunk_size) * manager.pipeline.frame_seq_length
        session.last_image = inp[:, :, [-1]].clone()
        session.noise_scale = noise_scale
        history_window = max(getattr(manager, "refresh_history_frames", 0), session.chunk_size + 1)
        history_images = getattr(session, "history_images", None)
        if history_images is None:
            history_images = session.last_image
        session.history_images = engine["torch"].cat([history_images, inp], dim=2)
        if session.history_images.shape[2] > history_window:
            session.history_images = session.history_images[:, :, -history_window:]
    engine["session"] = session
    engine["noise_scale"] = float(session.noise_scale)
    if out_img is not None:
        engine["last_output"] = out_img
    return out_img


# Collect ready frames
def _process_ready_outputs(engine, prompt):
    outputs = []
    if not engine["initialized"] and len(engine["frame_buffer"]) >= engine["priming_frames"]:
        outputs.append(start_real_stream(engine, prompt))
    while engine["initialized"] and len(engine["frame_buffer"]) >= engine["chunk_size"]:
        outputs.append(step_real_stream(engine))
    return outputs


# Cache output frames
def _cache_ready_frames(engine, output):
    stylized_frame = to_rgb_frame(output)
    if stylized_frame is None:
        return None, None
    stylized_frame = np.ascontiguousarray(stylized_frame[:, :, :3])
    engine["cached_stylized_frame"] = stylized_frame
    engine["cached_stylized_size"] = (stylized_frame.shape[1], stylized_frame.shape[0])

    superres_frame = None
    superres = engine.get("superres")
    if superres and superres.get("enabled"):
        superres_frame = maybe_apply_superres_rgb_array(engine, stylized_frame)
        if superres_frame is not None:
            superres_frame = np.ascontiguousarray(superres_frame[:, :, :3])
            engine["cached_superres_frame"] = superres_frame
            engine["cached_superres_size"] = (superres_frame.shape[1], superres_frame.shape[0])
    else:
        engine["cached_superres_frame"] = None
        engine["cached_superres_size"] = None

    return stylized_frame, superres_frame


# Send output packet
def _send_response(conn, engine, request, output=None, stylized_ready=False, reuse_latest=False):
    stylized_response = b""
    superres_response = b""
    stylized_w, stylized_h = request["width"], request["height"]
    superres_w, superres_h = 0, 0
    if stylized_ready and not reuse_latest and output is not None:
        stylized_frame, superres_frame = _cache_ready_frames(engine, output)
        if stylized_frame is None:
            stylized_ready = False
        else:
            stylized_h, stylized_w = stylized_frame.shape[:2]
            stylized_response = stylized_frame.tobytes()
            if superres_frame is not None:
                superres_h, superres_w = superres_frame.shape[:2]
                superres_response = superres_frame.tobytes()
    elif stylized_ready and reuse_latest:
        cached_stylized_size = engine.get("cached_stylized_size")
        cached_superres_size = engine.get("cached_superres_size")
        if cached_stylized_size is not None:
            stylized_w, stylized_h = cached_stylized_size
            if cached_superres_size is not None:
                superres_w, superres_h = cached_superres_size
        else:
            stylized_ready = False
            reuse_latest = False
    flags = 0
    if stylized_ready:
        flags |= RESPONSE_FLAG_STYLIZED_READY
    if reuse_latest:
        flags |= RESPONSE_FLAG_REUSE_LATEST
    conn.sendall(RESPONSE_HEADER.pack(
        RESPONSE_MAGIC,
        stylized_w,
        stylized_h,
        len(stylized_response),
        superres_w,
        superres_h,
        len(superres_response),
        flags,
        request["seq"],
    ))
    if stylized_response:
        conn.sendall(stylized_response)
    if superres_response:
        conn.sendall(superres_response)
    if stylized_ready:
        ready_time = time.perf_counter()
        if engine.get("first_ready_time") is None:
            engine["first_ready_time"] = ready_time
        engine["last_ready_time"] = ready_time
        engine["ready_response_count"] = int(engine.get("ready_response_count", 0)) + 1
        log_ready(request["seq"], reuse_latest)


# Send reuse packet
def _send_intermediate_response(conn, engine, request):
    if engine.get("cached_stylized_frame") is not None:
        _send_response(conn, engine, request, stylized_ready=True, reuse_latest=True)
        return
    _send_response(conn, engine, request, stylized_ready=False, reuse_latest=False)


# Run socket loop
def serve_socket(engine, args):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)
    try:
        conn, _ = server.accept()
        conn.settimeout(10.0)
        boundary_requests = []
        with conn:
            while True:
                try:
                    limit = int(engine["priming_frames"] if not engine["initialized"] else engine["chunk_size"])
                    requests = _drain_request_batch(conn, limit)
                    if not requests:
                        break
                except (OSError, ConnectionError):
                    break

                classified_initialized = engine["initialized"]
                fill = len(engine["frame_buffer"])
                try:
                    for request in requests:
                        if engine.get("first_request_time") is None:
                            engine["first_request_time"] = request["received_time"]
                        engine["frame_buffer"].append(request["rgb"])

                        if not classified_initialized:
                            fill += 1
                            if fill == engine["priming_frames"]:
                                boundary_requests.append(request)
                                classified_initialized = True
                                fill = 0
                            else:
                                _send_intermediate_response(conn, engine, request)
                        else:
                            fill += 1
                            if fill == engine["chunk_size"]:
                                boundary_requests.append(request)
                                fill = 0
                            else:
                                _send_intermediate_response(conn, engine, request)

                    outputs = _process_ready_outputs(engine, args.prompt)
                    for output in outputs:
                        if not boundary_requests:
                            break
                        request = boundary_requests.pop(0)
                        if output is not None:
                            ready_time = time.perf_counter()
                            if engine.get("first_output_time") is None:
                                engine["first_output_time"] = ready_time
                            engine["last_output_time"] = ready_time
                            engine["latency_sum_ms"] += (ready_time - request["received_time"]) * 1000.0
                            engine["latency_count"] += 1
                            engine["metrics_output_count"] += 1
                            _send_response(conn, engine, request, output, stylized_ready=True, reuse_latest=False)
                        else:
                            if engine.get("cached_stylized_frame") is not None:
                                _send_response(conn, engine, request, stylized_ready=True, reuse_latest=True)
                            else:
                                _send_response(conn, engine, request, stylized_ready=False, reuse_latest=False)
                except (OSError, ConnectionError):
                    break
    finally:
        server.close()
        write_metrics(engine)


# Run file engine
def run_engine(engine, image: Image.Image, prompt: str, negative_prompt: str):
    if isinstance(engine, dict) and engine.get("backend") == "streamv2v_real":
        frame = np.array(image.convert("RGB"), dtype=np.uint8)
        engine["frame_buffer"].append(frame)

        if not engine["initialized"]:
            if len(engine["frame_buffer"]) < engine["priming_frames"]:
                return engine["last_output"], False
            out_img = start_real_stream(engine, prompt)
            return out_img if out_img is not None else engine["last_output"], out_img is not None

        if len(engine["frame_buffer"]) < engine["chunk_size"]:
            return engine["last_output"], False

        out_img = step_real_stream(engine)
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
    raise RuntimeError()


# Parse cli args
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--model-id", default="")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--device", default="cuda")
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
    ready_flag_path = Path(args.ready_flag) if args.ready_flag else None
    interval = 1.0 / args.fps if args.fps > 0 else 0.1

    if args.real_v2:
        if not args.config_path or not args.checkpoint_folder:
            raise RuntimeError()
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
                    out_array = to_rgb_frame(out)
                    if out_array is not None:
                        out = maybe_apply_superres_rgb_array(engine, out_array)
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
