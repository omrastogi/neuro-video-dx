# render_person.py — run from GVHMR root
# Usage: python render_person.py outputs/demo/testing/scene_001/person_1

import sys, torch
import numpy as np
from pathlib import Path
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.net_utils import to_cuda
from hmr4d.utils.vis.renderer import Renderer
from hmr4d.utils.video_io_utils import get_video_lwh, get_writer, get_video_reader
from tqdm import tqdm

person_dir = Path(sys.argv[1])
scene_dir = person_dir.parent

# Find the scene clip video
clip = list(scene_dir.glob("clip_*.mp4"))[0]
pred = torch.load(person_dir / "hmr4d_results.pt")

L, W, H = get_video_lwh(str(clip))
K = pred["K_fullimg"][0]

# SMPL mesh from predictions
smplx = make_smplx("supermotion").cuda()
smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt").cuda()
faces = make_smplx("smpl").faces

out = smplx(**to_cuda(pred["smpl_params_incam"]))
verts = torch.stack([smplx2smpl @ v for v in out.vertices])

# Output filename
out_name = f"{scene_dir.name}_{person_dir.name}_render.mp4"
out_path = person_dir / out_name

# Render side-by-side: original (left) | mesh overlay (right)
renderer = Renderer(W, H, device="cuda", faces=faces, K=K)
reader = get_video_reader(str(clip))
writer = get_writer(str(out_path), fps=30, crf=23)

for i, img in tqdm(enumerate(reader), total=L, desc=f"Rendering {out_name}"):
    overlay = renderer.render_mesh(verts[i].cuda(), img.copy(), [0.8, 0.8, 0.8])
    combined = np.concatenate([img, overlay], axis=1)  # side by side
    writer.write_frame(combined)

writer.close()
reader.close()
print(f"Saved: {out_path}")