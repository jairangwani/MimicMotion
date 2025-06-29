# inference.py  – freeze lips, eyes, nose, eyebrows (shape); move with head
import argparse, copy, json, logging
from datetime import datetime
from pathlib import Path

import numpy as np, torch
from omegaconf import OmegaConf
from torchvision.datasets.folder import pil_loader
from torchvision.transforms.functional import pil_to_tensor, resize, to_pil_image

from mimicmotion.utils.geglu_patch import patch_geglu_inplace; patch_geglu_inplace()
from mimicmotion.dwpose.preprocess import (
    NpEncoder, draw_pose, get_image_pose, get_video_pose
)
from mimicmotion.utils.loader import create_pipeline
from mimicmotion.utils.utils import save_to_mp4

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── regions whose shape we freeze (68-point mesh indices) ------------
FREEZE_SLICES = {
    "brow_R": slice(17, 22),   # right eyebrow
    "brow_L": slice(22, 27),   # left eyebrow
    "nose":   slice(27, 36),   # ridge + tip
    "eye_L":  slice(36, 42),
    "eye_R":  slice(42, 48),
    "lips":   slice(48, 68),
}

# ── other helpers ----------------------------------------------------
SHOULDERS, HIPS = [2, 5], [8, 11]
ANCHORS = SHOULDERS + HIPS
ST = lambda arr, S, T: arr * S + T
unique_pts = lambda a: np.unique(a, axis=0).shape[0]
hand_valid = lambda h: h.size >= 42 and unique_pts(h) > 2
face_valid = lambda f: f.size >= 8 and unique_pts(f) > 3
has_face = lambda k: "faces" in k and len(k["faces"]) and face_valid(np.array(k["faces"][0]))

def torso_similarity(base_cand, src_cand, vis):
    ok = np.array([(i in ANCHORS) and vis[i] for i in range(18)])
    bp, sp = base_cand[ok], src_cand[ok]
    if bp.shape[0] < 4:
        raise ValueError
    bs, bh = bp[:2].mean(0), bp[2:].mean(0)
    ss, sh = sp[:2].mean(0), sp[2:].mean(0)
    hb, hs = np.linalg.norm(bs - bh), max(np.linalg.norm(ss - sh), 1e-6)
    wb, ws = np.linalg.norm(bp[0] - bp[1]), max(np.linalg.norm(sp[0] - sp[1]), 1e-6)
    S = (hb / hs + wb / ws) / 2
    T = bs - S * ss
    return float(S), T

def bbox_similarity(base_pts, src_pts):
    if base_pts.shape[0] < 2 or src_pts.shape[0] < 2:
        return 1.0, np.zeros(2)
    bx, by = np.ptp(base_pts[:, 0]), np.ptp(base_pts[:, 1])
    sx, sy = max(np.ptp(src_pts[:, 0]), 1e-6), max(np.ptp(src_pts[:, 1]), 1e-6)
    S = (bx / sx + by / sy) / 2
    T = base_pts.mean(0) - S * src_pts.mean(0)
    return float(S), T

# ── preprocessing ----------------------------------------------------
def preprocess(task, out_dir):
    # letterbox still image to 1024×576
    img = pil_loader(task.character_image_path)
    t = pil_to_tensor(img)
    new_w = int(t.shape[2] * (576 / t.shape[1]))
    canvas = torch.zeros((3, 576, 1024), dtype=t.dtype)
    canvas[:, :, (1024 - new_w)//2:(1024 - new_w)//2 + new_w] = resize(
        t, [576, new_w], antialias=True
    )
    img_hwc = canvas.permute(1, 2, 0).numpy()
    h, w, _ = img_hwc.shape

    # dirs
    png_dir = None
    if getattr(task, "save_pose_images", False):
        png_dir = (
            Path(task.pose_images_dir)
            if Path(task.pose_images_dir).is_absolute()
            else Path(out_dir) / task.pose_images_dir
        )
        png_dir.mkdir(parents=True, exist_ok=True)
    pose_dir = Path(task.dw_pose_dir) / f"{Path(task.motion_video_path).stem}_final_poses"
    pose_dir.mkdir(parents=True, exist_ok=True)

    # base pose
    _, base_kps = get_image_pose(img_hwc)
    json.dump(base_kps, (pose_dir / "frame_base.json").open("w"), cls=NpEncoder)
    if png_dir:
        to_pil_image(torch.from_numpy(draw_pose(base_kps, h, w).astype(np.uint8))).save(
            png_dir / "base_pose.png"
        )

    # build cache if needed
    cache_missing = not task.use_cached_pose or not (pose_dir / "frame_000000.json").exists()
    if cache_missing:
        motion = get_video_pose(
            task.motion_video_path, img_hwc, task.sample_stride, task.num_frames
        )
        logger.info(f"Motion frames: {len(motion)}")
        vis = base_kps["bodies"]["subset"][0] >= 0
        try:
            S, T = torso_similarity(
                base_kps["bodies"]["candidate"], motion[0]["bodies"]["candidate"], vis
            )
            logger.info(f"Torso similarity  S={S:.3f}  T={T}")
        except ValueError:
            bp = base_kps["bodies"]["candidate"][vis]
            sp = motion[0]["bodies"]["candidate"][vis]
            S, T = bbox_similarity(bp, sp)
            logger.info(f"BBox fallback     S={S:.3f}  T={T}")

        # freeze shapes
        freeze_shapes = {}
        base_face = np.array(base_kps.get("faces", [[]])[0])
        if base_face.size and base_face.shape[0] >= 68:
            for name, sl in FREEZE_SLICES.items():
                pts = base_face[sl]
                freeze_shapes[name] = pts - pts.mean(0)

        seq = []
        for mf in motion:
            nk = copy.deepcopy(mf)
            nk["bodies"]["candidate"] = ST(mf["bodies"]["candidate"], S, T)
            nk["hands"] = [
                ST(h, S, T) if hand_valid(h) else np.empty((0, 2), dtype=h.dtype)
                for h in mf.get("hands", [])
            ]

            if has_face(mf):
                cur_face = ST(np.array(mf["faces"][0]), S, T)
                if cur_face.shape[0] >= 68:
                    for name, sl in FREEZE_SLICES.items():
                        if name in freeze_shapes:
                            cur_center = cur_face[sl].mean(0)
                            cur_face[sl] = cur_center + freeze_shapes[name]
                nk["faces"] = [cur_face]
            else:
                nk.pop("faces", None)
            seq.append(nk)

        for i, k in enumerate(seq):
            json.dump(k, (pose_dir / f"frame_{i:06d}.json").open("w"), cls=NpEncoder)

    # load & draw
    kps = [
        json.load(p.open())
        for p in sorted(p for p in pose_dir.glob("frame_*.json") if p.name != "frame_base.json")
    ]
    for k in kps:
        if not isinstance(k["bodies"]["score"], np.ndarray):
            k["bodies"]["score"] = np.array(k["bodies"]["score"])
    imgs = [draw_pose(k, h, w) for k in kps]
    if png_dir:
        for i, im in enumerate(imgs):
            to_pil_image(torch.from_numpy(im.astype(np.uint8))).save(
                png_dir / f"pose_{i:06d}.png"
            )

    pose_px = torch.from_numpy(np.stack(imgs)).to(dtype=torch.float16) / 127.5 - 1
    img_px = canvas.unsqueeze(0).to(dtype=torch.float16) / 127.5 - 1
    return pose_px, img_px

# ── pipeline & main stay unchanged ----------------------------------
@torch.no_grad()
def run_pipeline(pipe, img_px, pose_px, task):
    ref = ((img_px[0] + 1) * 127.5).clamp(0, 255).to(torch.uint8).cpu()
    raw = pipe(
        [to_pil_image(ref)],
        image_pose=pose_px,
        num_frames=pose_px.size(0),
        tile_size=task.num_frames,
        tile_overlap=task.frames_overlap,
        height=pose_px.shape[-2],
        width=pose_px.shape[-1],
        fps=task.fps,
        noise_aug_strength=task.noise_aug_strength,
        num_inference_steps=task.num_inference_steps,
        generator=torch.Generator(device=img_px.device).manual_seed(task.seed),
        min_guidance_scale=task.guidance_scale,
        max_guidance_scale=task.guidance_scale,
        decode_chunk_size=8,
        output_type="pt",
        device=device,
    ).frames.cpu()
    vid = (raw * 255).to(torch.uint8)
    if vid.ndim == 5:
        vid = vid[0]
    return vid[1:]  # drop ref

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inference_config", default="configs/test.yaml")
    ap.add_argument("--output_dir", default=r"G:\My Drive\LIVA\Bucket\Temp\test_ground")
    ap.add_argument("--no_use_float16", action="store_true")
    args = ap.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if not args.no_use_float16:
        torch.set_default_dtype(torch.float16)

    cfg = OmegaConf.load(args.inference_config)
    pipe = create_pipeline(cfg, device)
    for i, task in enumerate(cfg.test_case):
        logger.info(f"--- Task {i+1}/{len(cfg.test_case)} ---")
        pose, img = preprocess(task, args.output_dir)
        vid = run_pipeline(pipe, img.to(device), pose.to(device), task)
        out = Path(args.output_dir) / (
            f"{Path(task.character_image_path).stem}_on_"
            f"{Path(task.motion_video_path).stem}_{datetime.now():%Y%m%d_%H%M%S}.mp4"
        )
        save_to_mp4(vid, str(out), fps=task.fps)
        logger.info(f"Saved ⇒ {out}")

if __name__ == "__main__":
    main()
