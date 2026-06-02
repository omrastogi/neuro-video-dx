import re
import random
from pathlib import Path

import av

LABEL_MAP = {"ms": 0, "pd": 1, "stroke": 2}
VALID_VIDEO_TYPES = {"clip", "mesh", "keypoints", "mesh_multiview"}


def _video_duration(video_path: str) -> float:
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        if stream.duration and stream.time_base:
            return float(stream.duration * stream.time_base)
        return float(container.duration / av.time_base)


def _select_fps(duration: float, max_frames: int, max_fps: float = 2.0, min_fps: float = 0.5) -> float:
    return max(min_fps, min(max_fps, max_frames / duration))


def _parse_sample_id(name: str) -> dict:
    """Parse 'ms_02__scene_single__person_1' into condition/scene/person."""
    parts = name.split("__")
    condition = re.match(r"([a-z]+)_\d+", parts[0]).group(1)
    scene = parts[1] if len(parts) > 1 else ""
    person = parts[2] if len(parts) > 2 else ""
    return {"condition": condition, "scene": scene, "person": person}


class NeuroDxVideoDataset:
    def __init__(
        self,
        data_root: str,
        video_type: str = "clip",
        split: str = "all",
        val_size: float = 0.2,
        seed: int = 42,
    ):
        assert video_type in VALID_VIDEO_TYPES, f"video_type must be one of {VALID_VIDEO_TYPES}"
        assert split in ("train", "val", "all"), "split must be 'train', 'val', or 'all'"

        self.data_root = Path(data_root)
        self.video_type = video_type
        self.filename = f"{video_type}.mp4"

        all_samples = self._collect_samples()
        self.samples = self._split(all_samples, split, val_size, seed)

    def _collect_samples(self) -> list[dict]:
        samples = []
        for d in sorted(self.data_root.iterdir()):
            if not d.is_dir():
                continue
            video_path = d / self.filename
            if not video_path.exists():
                continue
            parsed = _parse_sample_id(d.name)
            condition = parsed["condition"]
            if condition not in LABEL_MAP:
                continue
            samples.append({
                "video_path": str(video_path),
                "label": LABEL_MAP[condition],
                "label_name": condition,
                "sample_id": d.name,
                **parsed,
            })
        return samples

    def _split(self, samples: list[dict], split: str, val_size: float, seed: int) -> list[dict]:
        if split == "all":
            return samples

        rng = random.Random(seed)
        by_condition: dict[str, list] = {}
        for s in samples:
            by_condition.setdefault(s["condition"], []).append(s)

        train, val = [], []
        for condition_samples in by_condition.values():
            shuffled = condition_samples[:]
            rng.shuffle(shuffled)
            n_val = max(1, round(len(shuffled) * val_size))
            val.extend(shuffled[:n_val])
            train.extend(shuffled[n_val:])

        return train if split == "train" else val

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


def build_qwen_messages(
    sample: dict,
    prompt: str,
    fps: float | None = None,
    max_frames: int = 64,
) -> list[dict]:
    """Build a Qwen3-VL messages list from a dataset sample."""
    video_path = sample["video_path"]

    if fps is None:
        try:
            duration = _video_duration(video_path)
            fps = _select_fps(duration, max_frames)
        except Exception:
            fps = 1.0

    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "fps": fps,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
