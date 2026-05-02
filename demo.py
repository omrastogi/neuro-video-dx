"""
Multi-Person GVHMR Inference Script (with Scene Detection)
============================================================
Handles videos with scene cuts + multiple people.

Pipeline:
  0. Detect scene cuts (PySceneDetect)
  1. For each scene clip:
     a. YOLO detection + tracking  → all person tracks
     b. Visual Odometry            → camera poses (shared, once per scene)
     c. For each person:
        - ViTPose → 2D keypoints
        - ViT feature extraction → image features
        - GVHMR predict → SMPLX params + global motion

Usage:
  cd /path/to/GVHMR
  python demo.py --video data/testing.mp4               # moving cam + scene detect
  python demo.py --video data/testing.mp4 -s             # static cam
  python demo.py --video data/testing.mp4 --scene_threshold 30
  python demo.py --video data/testing.mp4 --no_scene_detect
  python demo.py --video data/testing.mp4 --min_track_len 30

Outputs:
  outputs/demo/<video>/scene_000/person_<id>/hmr4d_results.pt
  (If 1 scene: outputs/demo/<video>/person_<id>/hmr4d_results.pt)
"""

import argparse
import sys
import torch
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# ── GVHMR imports (same as tools/demo/demo.py) ──────────────────────────────

import hydra
from hydra import initialize_config_module, compose

from hmr4d.configs import register_store_gvhmr
from hmr4d.utils.pylogger import Log
from hmr4d.utils.video_io_utils import get_video_lwh, read_video_np, save_video, get_writer, get_video_reader
from hmr4d.utils.preproc import Tracker, Extractor, VitPoseExtractor, SimpleVO
from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy, estimate_K, create_camera_sensor
from hmr4d.utils.geo_transform import compute_cam_angvel
from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL
from hmr4d.utils.net_utils import detach_to_cpu


# ═════════════════════════════════════════════════════════════════════════════
# Scene Detection
# ═════════════════════════════════════════════════════════════════════════════

def detect_scenes(video_path, threshold=27.0, min_scene_len=30):
    """Detect scene cuts. Returns list of (start_frame, end_frame) tuples."""
    try:
        from scenedetect import detect, ContentDetector
    except ImportError:
        Log.info("[WARN] scenedetect not installed (pip install scenedetect[opencv])")
        Log.info("[WARN] Treating entire video as one scene.")
        L = get_video_lwh(str(video_path))[0]
        return [(0, L)]

    scene_list = detect(str(video_path), ContentDetector(threshold=threshold))

    if len(scene_list) == 0:
        L = get_video_lwh(str(video_path))[0]
        return [(0, L)]

    segments = []
    for start_tc, end_tc in scene_list:
        sf = start_tc.get_frames()
        ef = end_tc.get_frames()
        if ef - sf >= min_scene_len:
            segments.append((sf, ef))
        elif len(segments) > 0:
            segments[-1] = (segments[-1][0], ef)
        else:
            segments.append((sf, ef))

    return segments


def write_scene_clip(all_frames, start_f, end_f, out_path, fps, W, H):
    """Write a subset of frames to a video file."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))
    for f in all_frames[start_f:end_f]:
        writer.write(f)
    writer.release()


# ═════════════════════════════════════════════════════════════════════════════
# Multi-Person Tracking (wraps YOLO directly for all tracks)
# ═════════════════════════════════════════════════════════════════════════════

def get_all_tracks(video_path, min_track_len=15):
    """
    Run YOLO tracking and return ALL person tracks.
    The original Tracker.get_one_track() picks only the longest track.
    We use the same YOLO model but keep every track.

    Returns:
        dict: {track_id: {"frame_ids": [...], "bbx_xyxys": np.array}}
        int:  total frame count
    """
    from ultralytics import YOLO

    yolo = YOLO("inputs/checkpoints/yolo/yolov8x.pt")

    id_to_frame_ids = defaultdict(list)
    id_to_bbx_xyxys = defaultdict(list)

    L = get_video_lwh(str(video_path))[0]
    reader = get_video_reader(str(video_path))

    for frame_idx, frame in tqdm(enumerate(reader), total=L, desc="YoloV8 Tracking"):
        results = yolo.track(
            frame, persist=True, classes=[0], verbose=False, conf=0.3
        )
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                tid = int(boxes.id[i].item())
                xyxy = boxes.xyxy[i].cpu().numpy()
                id_to_frame_ids[tid].append(frame_idx)
                id_to_bbx_xyxys[tid].append(xyxy)

    reader.close()

    valid = {}
    for tid in id_to_frame_ids:
        if len(id_to_frame_ids[tid]) >= min_track_len:
            valid[tid] = {
                "frame_ids": id_to_frame_ids[tid],
                "bbx_xyxys": np.array(id_to_bbx_xyxys[tid]),
            }

    return valid, L


def build_full_bbx(track_data, total_frames):
    """
    Build a full-length (L, 4) bbx_xyxy tensor from sparse detections,
    with linear interpolation for gaps and edge replication outside the span.
    """
    frame_ids = track_data["frame_ids"]
    bbx_np = track_data["bbx_xyxys"]

    bbx_xyxy = torch.zeros(total_frames, 4)
    frame_mask = torch.zeros(total_frames, dtype=torch.bool)

    for fi, box in zip(frame_ids, bbx_np):
        bbx_xyxy[fi] = torch.tensor(box, dtype=torch.float32)
        frame_mask[fi] = True

    start_f, end_f = min(frame_ids), max(frame_ids)

    # Linear interpolation within track span
    for fi in range(start_f, end_f + 1):
        if not frame_mask[fi]:
            prev_fi = max(f for f in frame_ids if f < fi)
            next_fi = min(f for f in frame_ids if f > fi)
            alpha = (fi - prev_fi) / (next_fi - prev_fi)
            bbx_xyxy[fi] = (1 - alpha) * bbx_xyxy[prev_fi] + alpha * bbx_xyxy[next_fi]
            frame_mask[fi] = True

    # Edge replication
    bbx_xyxy[:start_f] = bbx_xyxy[start_f]
    if end_f + 1 < total_frames:
        bbx_xyxy[end_f + 1:] = bbx_xyxy[end_f]

    return bbx_xyxy, frame_mask


# ═════════════════════════════════════════════════════════════════════════════
# Per-Scene Multi-Person Pipeline
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_scene(
    video_path, scene_out_dir, static_cam, f_mm, min_track_len,
    vitpose_extractor, feature_extractor, model,
):
    """Run multi-person GVHMR on a single scene clip (no scene cuts inside)."""
    scene_out_dir = Path(scene_out_dir)
    scene_out_dir.mkdir(parents=True, exist_ok=True)

    video_path = str(video_path)
    L, W, H = get_video_lwh(video_path)
    Log.info(f"  Scene clip: {Path(video_path).name} ({L} frames, {W}x{H})")

    if L < min_track_len:
        Log.info(f"  [SKIP] Scene too short ({L} < {min_track_len})")
        return {}

    # ── 1. Multi-person tracking ─────────────────────────────────────────
    Log.info(f"  [1] YOLO multi-person tracking...")
    valid_tracks, _ = get_all_tracks(video_path, min_track_len=min_track_len)
    Log.info(f"  [1] {len(valid_tracks)} valid tracks (>= {min_track_len} frames)")
    if len(valid_tracks) == 0:
        return {}

    # ── 2. Visual Odometry (shared) ──────────────────────────────────────
    R_w2c = None
    if not static_cam:
        Log.info(f"  [2] Visual Odometry...")
        slam_path = scene_out_dir / "slam_results.pt"
        if not slam_path.exists():
            try:
                simple_vo = SimpleVO(video_path, scale=0.5, step=8, method="sift", f_mm=f_mm)
                vo_results = simple_vo.compute()
                torch.save(vo_results, slam_path)
            except Exception as e:
                Log.info(f"  [WARN] VO failed ({e}), using identity rotation.")
                vo_results = np.tile(np.eye(4, dtype=np.float32), (L, 1, 1))
                torch.save(vo_results, slam_path)
        else:
            vo_results = torch.load(slam_path)
        R_w2c = torch.from_numpy(vo_results[:, :3, :3])
    else:
        Log.info(f"  [2] Skipping VO (static cam)")

    if R_w2c is None:
        R_w2c = torch.eye(3).repeat(L, 1, 1)

    cam_angvel = compute_cam_angvel(R_w2c)

    if f_mm is not None:
        K_fullimg = create_camera_sensor(W, H, f_mm)[2].repeat(L, 1, 1)
    else:
        K_fullimg = estimate_K(W, H).repeat(L, 1, 1)

    # ── 3. Per-person inference ──────────────────────────────────────────
    person_results = {}
    for pidx, (tid, track_data) in enumerate(valid_tracks.items()):
        n_det = len(track_data["frame_ids"])
        Log.info(f"  [3] Person {pidx+1}/{len(valid_tracks)} "
                 f"(track_id={tid}, {n_det} detections)")

        person_dir = scene_out_dir / f"person_{tid}"
        person_dir.mkdir(parents=True, exist_ok=True)

        # Build full-length bounding boxes
        bbx_xyxy, frame_mask = build_full_bbx(track_data, L)
        bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()

        torch.save(
            {"bbx_xyxy": bbx_xyxy, "bbx_xys": bbx_xys, "frame_mask": frame_mask},
            person_dir / "bbx.pt",
        )

        # ViTPose
        Log.info(f"      ViTPose...")
        vitpose = vitpose_extractor.extract(video_path, bbx_xys)
        torch.save(vitpose, person_dir / "vitpose.pt")

        # ViT features
        Log.info(f"      ViT features...")
        vit_features = feature_extractor.extract_video_features(video_path, bbx_xys)
        torch.save(vit_features, person_dir / "vit_features.pt")

        # GVHMR predict — same data dict as original demo.py's load_data_dict()
        Log.info(f"      GVHMR predict...")
        data = {
            "length": torch.tensor(L),
            "bbx_xys": bbx_xys,
            "kp2d": vitpose,
            "K_fullimg": K_fullimg,
            "cam_angvel": cam_angvel,
            "f_imgseq": vit_features,
        }
        pred = model.predict(data, static_cam=static_cam)
        pred = detach_to_cpu(pred)

        torch.save(pred, person_dir / "hmr4d_results.pt")
        Log.info(f"      -> {person_dir / 'hmr4d_results.pt'}")
        person_results[tid] = pred

    return person_results


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Multi-person GVHMR + scene detection")
    p.add_argument("--video", required=True, type=str)
    p.add_argument("-s", "--static_cam", action="store_true")
    p.add_argument("--output_root", default="outputs/demo")
    p.add_argument("--min_track_len", type=int, default=15)
    p.add_argument("--f_mm", type=int, default=None)
    p.add_argument("--scene_threshold", type=float, default=27.0)
    p.add_argument("--min_scene_len", type=int, default=30)
    p.add_argument("--no_scene_detect", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    video_path = Path(args.video).resolve()
    assert video_path.exists(), f"Video not found: {video_path}"
    video_name = video_path.stem
    out_root = Path(args.output_root) / video_name
    out_root.mkdir(parents=True, exist_ok=True)

    L, W, H = get_video_lwh(str(video_path))
    Log.info(f"[Input] {video_path}")
    Log.info(f"[Input] (L, W, H) = ({L}, {W}, {H})")

    # ── Step 0: Scene Detection ──────────────────────────────────────────
    if args.no_scene_detect:
        Log.info(f"[0/4] Scene detection skipped")
        scenes = [(0, L)]
    else:
        Log.info(f"[0/4] Detecting scene cuts (threshold={args.scene_threshold})...")
        scenes = detect_scenes(video_path, args.scene_threshold, args.min_scene_len)

    Log.info(f"[0/4] {len(scenes)} scene(s):")
    for i, (sf, ef) in enumerate(scenes):
        Log.info(f"       Scene {i}: frames {sf}-{ef} ({ef-sf} frames)")

    # ── Step 1: Read all frames ──────────────────────────────────────────
    Log.info(f"[1/4] Reading video frames...")
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    cap.release()
    Log.info(f"       {len(all_frames)} frames at {fps:.1f} fps")

    # ── Step 2: Load GVHMR model via Hydra (same as original demo.py) ────
    Log.info(f"[2/4] Loading models...")

    with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
        register_store_gvhmr()
        overrides = [
            f"video_name={video_name}",
            f"static_cam={args.static_cam}",
            f"verbose={args.verbose}",
        ]
        if args.f_mm is not None:
            overrides.append(f"f_mm={args.f_mm}")
        cfg = compose(config_name="demo", overrides=overrides)

    # Instantiate model — exactly like original demo.py line 305
    model: DemoPL = hydra.utils.instantiate(cfg.model, _recursive_=False)
    model.load_pretrained_model(cfg.ckpt_path)
    model = model.eval().cuda()
    Log.info(f"       GVHMR on {torch.cuda.get_device_name()}")

    vitpose_extractor = VitPoseExtractor()
    feature_extractor = Extractor()
    Log.info(f"       ViTPose + ViT extractor loaded")

    # ── Step 3: Process each scene ───────────────────────────────────────
    all_results = {}
    single_scene = (len(scenes) == 1)

    for scene_idx, (sf, ef) in enumerate(scenes):
        n_frames = ef - sf
        if single_scene:
            scene_out_dir = out_root
            scene_label = "all"
        else:
            scene_label = f"scene_{scene_idx:03d}"
            scene_out_dir = out_root / scene_label

        Log.info(f"\n{'='*60}")
        Log.info(f"[3/4] Scene {scene_idx+1}/{len(scenes)}: "
                 f"frames {sf}-{ef} ({n_frames} frames)")
        Log.info(f"{'='*60}")

        # Write scene clip to disk
        scene_clip_path = scene_out_dir / f"clip_{sf}_{ef}.mp4"
        scene_out_dir.mkdir(parents=True, exist_ok=True)
        if not scene_clip_path.exists():
            write_scene_clip(all_frames, sf, ef, scene_clip_path, fps, W, H)
            Log.info(f"  Wrote clip: {scene_clip_path}")

        try:
            scene_results = run_scene(
                video_path=scene_clip_path,
                scene_out_dir=scene_out_dir,
                static_cam=args.static_cam,
                f_mm=args.f_mm,
                min_track_len=args.min_track_len,
                vitpose_extractor=vitpose_extractor,
                feature_extractor=feature_extractor,
                model=model,
            )
            all_results[scene_label] = scene_results
        except Exception as e:
            Log.info(f"  [ERROR] Scene {scene_idx} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    # ── Summary ──────────────────────────────────────────────────────────
    Log.info(f"\n{'='*60}")
    Log.info(f"[4/4] Done!")
    Log.info(f"{'='*60}")
    Log.info(f"Output: {out_root}")

    total_persons = 0
    for d in sorted(out_root.rglob("person_*")):
        if d.is_dir():
            results_file = d / "hmr4d_results.pt"
            status = "OK" if results_file.exists() else "MISSING"
            rel = d.relative_to(out_root)
            Log.info(f"  {rel}/hmr4d_results.pt  [{status}]")
            if results_file.exists():
                total_persons += 1

    Log.info(f"Total: {total_persons} person(s) across {len(scenes)} scene(s)")


if __name__ == "__main__":
    main()