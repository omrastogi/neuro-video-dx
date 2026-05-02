# neuro-video-dx

Multi-person human motion recovery from video using [GVHMR](https://zju3dv.github.io/gvhmr). Handles scene cuts, moving cameras, and multiple people per video.

## Setup

**1. Clone and create environment**

```bash
git clone https://github.com/omrastogi/neuro-video-dx.git
cd neuro-video-dx

conda create -y -n gvhmr python=3.10
conda activate gvhmr
pip install -r requirements.txt
pip install -e .
```

**2. Download checkpoints**

All weights are pulled from HuggingFace — no manual sign-ups required:

```bash
bash dnwd_ckpt.sh
```

This downloads into `inputs/checkpoints/`:

```
inputs/checkpoints/
├── gvhmr/gvhmr_siga24_release.ckpt
├── hmr2/epoch=10-step=25000.ckpt
├── vitpose/vitpose-h-multi-coco.pth
├── dpvo/dpvo.pth
├── yolo/yolov8x.pt
└── body_models/
    ├── smpl/SMPL_NEUTRAL.pkl
    └── smplx/SMPLX_NEUTRAL.npz
```

**3. (Optional) Install scenedetect**

For automatic scene-cut detection:

```bash
pip install scenedetect[opencv]
```

## Running inference

Place your video in the `data/` folder, then run `demo.py`.

**Basic usage (moving camera):**
```bash
python demo.py --video data/your_video.mp4
```

**Static camera (skip visual odometry):**
```bash
python demo.py --video data/your_video.mp4 -s
```

**Disable scene detection:**
```bash
python demo.py --video data/your_video.mp4 --no_scene_detect
```

**All options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--video` | required | Path to input video |
| `-s / --static_cam` | off | Skip visual odometry for static cameras |
| `--output_root` | `outputs/demo` | Root directory for results |
| `--min_track_len` | 15 | Minimum frames for a person track to be kept |
| `--f_mm` | auto | Camera focal length in mm (full-frame equivalent) |
| `--scene_threshold` | 27.0 | Sensitivity for scene cut detection |
| `--min_scene_len` | 30 | Minimum frames per scene segment |
| `--no_scene_detect` | off | Treat the whole video as one scene |

## Output structure

Results are saved as `.pt` files per person, per scene:

```
outputs/demo/<video_name>/
├── person_<id>/
│   ├── hmr4d_results.pt   # SMPLX params + global motion
│   ├── bbx.pt
│   ├── vitpose.pt
│   └── vit_features.pt
└── (multi-scene videos)
    └── scene_000/person_<id>/...
```
