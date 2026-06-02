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

## Clinical gait analysis with Qwen3-VL

`qwen/test_inference.py` runs Qwen3-VL-8B on the processed videos and produces a structured clinical diagnosis (MS / PD / Stroke) using a three-stage reasoning prompt.

### Setup

```bash
pip install transformers accelerate qwen-vl-utils pandas
```

The model (`Qwen/Qwen3-VL-8B-Instruct`) is downloaded automatically from HuggingFace on first run.

### Data layout

Videos must be organised under `data/final/` with sample IDs in the format `{condition}__{scene}__{person}__{type}`:

```
data/final/
├── ms__scene001__p01__clip.mp4
├── ms__scene001__p01__mesh.mp4
├── pd__scene002__p02__keypoints.mp4
└── ...
```

Supported `--video_type` values: `clip`, `mesh`, `keypoints`, `mesh_multiview`.

### Running inference

**Quick test (3 samples, raw clip):**
```bash
python -m qwen.test_inference
```

**Specify video type and sample count:**
```bash
python -m qwen.test_inference --video_type mesh --samples 10
```

**Enable chain-of-thought thinking tokens:**
```bash
python -m qwen.test_inference --video_type keypoints --thinking --samples 5
```

**Save per-sample JSON + summary CSV:**
```bash
python -m qwen.test_inference --video_type clip --samples 20 --save
```

**All options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `Qwen/Qwen3-VL-8B-Instruct` | HuggingFace model ID |
| `--video_type` | `clip` | One of `clip`, `mesh`, `keypoints`, `mesh_multiview` |
| `--samples` | `3` | Number of samples to run |
| `--thinking` | off | Enable thinking tokens (increases max output to 4096) |
| `--fps` | auto | Override video FPS fed to the model |
| `--max_frames` | `64` | Maximum frames sampled per video |
| `--save` | off | Write per-sample JSON and summary CSV to `results/` |

### Output

Each run prints a per-sample breakdown (stages 1–3) followed by a summary table:

```
======================================================================
SUMMARY TABLE
======================================================================
         sample_id true_label predicted  correct confidence  prob_MS  prob_PD  prob_Stroke  tally_MS  tally_PD  tally_Stroke
ms__s1__p1__clip          MS        MS     True       high     0.82     0.10         0.08         5         1             1
pd__s2__p1__clip          PD        PD     True     medium     0.05     0.75         0.20         1         4             2

Accuracy: 2/2 correct
```

When `--save` is used, results are written to:

```
results/
├── {sample_id}__{video_type}.json   # full response + parsed JSON per sample
└── summary__{video_type}.csv        # aggregated table for all samples
```

---

## Output structure (GVHMR)

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
