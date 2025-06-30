# ===================================================================
# inference.py – zoom-match + head/face freeze  + optional freeze_all
# ===================================================================
import argparse, copy, json, logging
from datetime import datetime
from pathlib import Path
import numpy as np, torch
from omegaconf import OmegaConf
from torchvision.datasets.folder import pil_loader
from torchvision.transforms.functional import pil_to_tensor, resize, to_pil_image

# ─────────────────── MimicMotion internals ────────────────────
from mimicmotion.utils.geglu_patch import patch_geglu_inplace; patch_geglu_inplace()
from mimicmotion.dwpose.preprocess import (
    NpEncoder, draw_pose, get_image_pose, get_video_pose
)
from mimicmotion.utils.loader      import create_pipeline
from mimicmotion.utils.utils       import save_to_mp4
# ──────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------- constants & utilities -------------------
# body-18 indices: 0-nose 1-neck 2-R-shoulder 5-L-shoulder 8-R-hip 11-L-hip …
SHOULDERS, HIPS = [2, 5], [8, 11]
ANCHORS         = SHOULDERS + HIPS
HEAD_IDX        = [0, 1, 14, 15, 16, 17]          # nose + eyes + ears
ST              = lambda arr, S, T: arr * S + T   # scale/translate helper

unique_pts = lambda a: np.unique(a, axis=0).shape[0]
hand_valid = lambda h: h.size >= 42 and unique_pts(h) > 2
face_valid = lambda f: f.size >= 8  and unique_pts(f) > 3
has_face   = lambda k: "faces" in k and len(k["faces"]) and face_valid(np.array(k["faces"][0]))

def torso_similarity(base_cand, src_cand, vis_mask):
    ok = np.array([(i in ANCHORS) and vis_mask[i] for i in range(18)])
    bp, sp = base_cand[ok], src_cand[ok]
    if bp.shape[0] < 4:                       # need both shoulders & hips
        raise ValueError
    cb_b, ch_b = bp[:2].mean(0), bp[2:].mean(0)
    cb_s, ch_s = sp[:2].mean(0), sp[2:].mean(0)
    Hb, Hs = np.linalg.norm(cb_b - ch_b), max(np.linalg.norm(cb_s - ch_s), 1e-6)
    Wb, Ws = np.linalg.norm(bp[0] - bp[1]), max(np.linalg.norm(sp[0] - sp[1]), 1e-6)
    S = (Hb / Hs + Wb / Ws) / 2
    T = cb_b - S * cb_s
    return float(S), T

def bbox_similarity(base_pts, src_pts):
    if len(base_pts) < 2 or len(src_pts) < 2:
        return 1.0, np.zeros(2, float)
    bx, by = np.ptp(base_pts[:, 0]), np.ptp(base_pts[:, 1])
    sx = max(np.ptp(src_pts[:, 0]), 1e-6)
    sy = max(np.ptp(src_pts[:, 1]), 1e-6)
    S = (bx / sx + by / sy) / 2
    T = base_pts.mean(0) - S * src_pts.mean(0)
    return float(S), T

# ------------------- preprocessing ---------------------------
def preprocess(task, out_dir: str, freeze_all: bool):
    # 1. letter-box portrait to 1024×576
    img = pil_loader(task.character_image_path)
    t   = pil_to_tensor(img)
    new_w = int(t.shape[2] * (576 / t.shape[1]))
    canvas = torch.zeros((3, 576, 1024), dtype=t.dtype)
    canvas[:, :, (1024 - new_w)//2:(1024 - new_w)//2 + new_w] = resize(
        t, [576, new_w], antialias=True)
    img_hwc = canvas.permute(1, 2, 0).numpy();  h, w, _ = img_hwc.shape

    # 2. dirs
    png_dir  = None
    if getattr(task, "save_pose_images", False):
        png_dir = (Path(task.pose_images_dir) if Path(task.pose_images_dir).is_absolute()
                   else Path(out_dir) / task.pose_images_dir)
        png_dir.mkdir(parents=True, exist_ok=True)

    pose_dir = Path(task.dw_pose_dir) / f"{Path(task.motion_video_path).stem}_final_poses"
    pose_dir.mkdir(parents=True, exist_ok=True)

    # 3. base pose (from portrait)
    _, base_kps = get_image_pose(img_hwc)
    json.dump(base_kps, (pose_dir / "frame_base.json").open("w"), cls=NpEncoder)
    if png_dir:
        to_pil_image(torch.from_numpy(draw_pose(base_kps, h, w).astype(np.uint8)))\
            .save(png_dir / "base_pose.png")

    # 4. build cache if needed
    if (not task.use_cached_pose) or not (pose_dir / "frame_000000.json").exists():
        motion = get_video_pose(task.motion_video_path,
                                img_hwc,
                                task.sample_stride,
                                task.num_frames)
        logger.info(f"Motion frames: {len(motion)}")

        vis = base_kps["bodies"]["subset"][0] >= 0
        try:
            S, T = torso_similarity(base_kps["bodies"]["candidate"],
                                    motion[0]["bodies"]["candidate"], vis)
            logger.info(f"Torso similarity  S={S:.3f}  T={T}")
        except ValueError:
            bp = base_kps["bodies"]["candidate"][vis]
            sp = motion[0]["bodies"]["candidate"][vis]
            S, T = bbox_similarity(bp, sp)
            logger.info(f"BBox fallback     S={S:.3f}  T={T}")

        seq = []

        # ---------- hybrid reference frame ----------
        ref = copy.deepcopy(motion[0])
        # body (scaled motion)
        ref["bodies"]["candidate"] = ST(ref["bodies"]["candidate"], S, T)
        ref["hands"] = [ST(h, S, T) if hand_valid(h)
                        else np.empty((0, 2), dtype=h.dtype)
                        for h in ref.get("hands", [])]

        # head joints from PORTRAIT (already at correct coords)
        ref["bodies"]["candidate"][HEAD_IDX] = base_kps["bodies"]["candidate"][HEAD_IDX]
        ref["bodies"]["score"][0, HEAD_IDX]  = base_kps["bodies"]["score"][0, HEAD_IDX]

        # face mesh from PORTRAIT
        if has_face(base_kps):
            ref["faces"]       = [np.array(base_kps["faces"][0])]
            ref["faces_score"] = [np.array(base_kps["faces_score"][0])]
        else:
            ref.pop("faces", None)

        seq.append(ref)

        # ---------- rest of sequence ----------
        if freeze_all:
            logger.info("⚙️  freeze_all=True → duplicating hybrid frame to every step")
            seq.extend(copy.deepcopy(ref) for _ in range(task.num_frames - 1))
        else:
            head_static = ref["bodies"]["candidate"][HEAD_IDX].copy()
            head_score  = ref["bodies"]["score"][0, HEAD_IDX].copy()
            face_static = np.array(ref["faces"][0])       if "faces" in ref else None
            face_score  = np.array(ref["faces_score"][0]) if "faces" in ref else None

            for mf in motion[1:]:
                k = copy.deepcopy(mf)
                k["bodies"]["candidate"] = ST(k["bodies"]["candidate"], S, T)
                k["hands"] = [ST(h, S, T) if hand_valid(h)
                              else np.empty((0, 2), dtype=h.dtype)
                              for h in k.get("hands", [])]

                # freeze head & face
                k["bodies"]["candidate"][HEAD_IDX] = head_static
                k["bodies"]["score"][0, HEAD_IDX]  = head_score
                if face_static is not None:
                    k["faces"], k["faces_score"] = [face_static.copy()], [face_score.copy()]
                else:
                    k.pop("faces", None)
                seq.append(k)

        # cache to disk
        for i, k in enumerate(seq):
            json.dump(k, (pose_dir / f"frame_{i:06d}.json").open("w"), cls=NpEncoder)

    # 5. preview → tensors
    kps_files = sorted(p for p in pose_dir.glob("frame_[0-9]*.json"))
    kps = [json.load(p.open()) for p in kps_files]
    for k in kps:                              # scores may reload as list
        if not isinstance(k["bodies"]["score"], np.ndarray):
            k["bodies"]["score"] = np.array(k["bodies"]["score"])

    imgs = [draw_pose(k, h, w) for k in kps]
    if png_dir:
        for i, im in enumerate(imgs):
            to_pil_image(torch.from_numpy(im.astype(np.uint8)))\
                .save(png_dir / f"pose_{i:06d}.png")

    pose_px = torch.from_numpy(np.stack(imgs)).to(torch.float16) / 127.5 - 1
    img_px  = canvas.unsqueeze(0).to(torch.float16) / 127.5 - 1
    return pose_px, img_px

# ------------------- generation ------------------------------
@torch.no_grad()
def run_pipeline(pipe, img_px, pose_px, task):
    ref_img = ((img_px[0] + 1) * 127.5).clamp(0, 255).to(torch.uint8).cpu()
    raw = pipe([to_pil_image(ref_img)],
               image_pose          = pose_px,
               num_frames          = pose_px.size(0),
               tile_size           = task.num_frames,
               tile_overlap        = task.frames_overlap,
               height              = pose_px.shape[-2],
               width               = pose_px.shape[-1],
               fps                 = task.fps,
               noise_aug_strength  = task.noise_aug_strength,
               num_inference_steps = task.num_inference_steps,
               generator           = torch.Generator(device=img_px.device).manual_seed(task.seed),
               min_guidance_scale  = task.guidance_scale,
               max_guidance_scale  = task.guidance_scale,
               decode_chunk_size   = 8,
               output_type         = "pt",
               device              = device).frames.cpu()

    vid = (raw * 255).to(torch.uint8)
    vid = vid[0] if vid.ndim == 5 else vid          # [F,C,H,W]
    return vid[1:]                                  # drop reference

# ------------------- main ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inference_config", default="configs/test.yaml")
    ap.add_argument("--output_dir",        default=r"G:\My Drive\LIVA\Bucket\Temp\test_ground")
    ap.add_argument("--no_use_float16",    action="store_true")
    ap.add_argument("--freeze_all",        action="store_true",
                    help="Duplicate the hybrid frame to every step (debug)")
    args = ap.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if not args.no_use_float16:
        torch.set_default_dtype(torch.float16)

    cfg  = OmegaConf.load(args.inference_config)
    pipe = create_pipeline(cfg, device)

    for i, task in enumerate(cfg.test_case):
        logger.info(f"--- Task {i+1}/{len(cfg.test_case)} ---")
        pose, img = preprocess(task, args.output_dir, freeze_all=args.freeze_all)
        vid       = run_pipeline(pipe, img.to(device), pose.to(device), task)

        out = Path(args.output_dir) / (
            f"{Path(task.character_image_path).stem}_on_"
            f"{Path(task.motion_video_path).stem}_{datetime.now():%Y%m%d_%H%M%S}.mp4")
        save_to_mp4(vid, str(out), fps=task.fps)
        logger.info(f"Saved ⇒ {out}")

if __name__ == "__main__":
    main()
