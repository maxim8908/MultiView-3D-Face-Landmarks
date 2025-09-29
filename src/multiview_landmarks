import os
import numpy as np
import torch
from pytorch3d.renderer import FoVPerspectiveCameras

from .renderer_texture_2d import MeshRenderer2D
from .landmark_detector_2d import LandmarkDetector2D
from .rasterizer_landmark_projector import RasterizerLandmarkProjector

class MultiViewLandmarkZBufferConsensus:
    """
    Render multiple views, detect 2D landmarks, project them back to 3D via rasterizer fragments,
    and perform cross-view depth ranking + centroid selection for robust consensus.
    """

    def __init__(self, mesh_path, texture_path, view_configs, image_size=(1024, 1024), device="cpu"):
        self.mesh_path = mesh_path
        self.texture_path = texture_path
        self.view_configs = view_configs
        self.image_size = image_size
        self.device = device

        self.landmark_sets_3d = {}   # {view: [list of 68 torch(3,) or None]}
        self.landmark_sets_idx = {}  # {view: [list of 68 face_idx or None]}
        self.cameras = {}            # {view: FoVPerspectiveCameras}
        self.landmarks_2d_dict = {}  # {view: (68,2) np.float32}
        self.per_lm_crossview = None
        self.consensus = None

    # --------- Prepare ----------
    def prepare_views(self, dist=55.0, fov=30.0):
        for cfg in self.view_configs:
            name, elev, azim = cfg["name"], cfg["elev"], cfg["azim"]
            renderer = MeshRenderer2D(self.mesh_path, self.texture_path, device=self.device, image_size=self.image_size, fov=fov)
            renderer.load_mesh()
            renderer.set_camera(dist=dist, elev=elev, azim=azim)
            img = renderer.render()                                  # (H,W,3) float in [0,1]
            img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)

            # Detect 2D landmarks
            detector = LandmarkDetector2D(img_u8, device=str(self.device))
            lms2d = detector.detect()
            self.landmarks_2d_dict[name] = lms2d if lms2d is not None else np.full((68,2), np.nan, dtype=np.float32)

            # If detection failed, keep placeholders
            if lms2d is None:
                self.landmark_sets_3d[name] = [None] * 68
                self.landmark_sets_idx[name] = [None] * 68
                self.cameras[name] = renderer.get_camera()
                continue

            # Project 2D → 3D via rasterizer fragments
            rasterizer = renderer.get_rasterizer()
            mesh = renderer.get_mesh()
            projector = RasterizerLandmarkProjector(mesh, rasterizer, self.image_size, device=self.device)
            coords_3d, indices_3d = projector.project_landmarks(lms2d)

            self.landmark_sets_3d[name] = coords_3d
            self.landmark_sets_idx[name] = indices_3d
            self.cameras[name] = renderer.get_camera()

    # --------- Cache ----------
    def save_prepared(self, cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
        for view in list(self.cameras.keys()):
            coords_list = self.landmark_sets_3d.get(view, None)
            if coords_list is None:
                continue

            coords_np = []
            for p in coords_list:
                if p is None:
                    coords_np.append([np.nan, np.nan, np.nan])
                else:
                    coords_np.append(p.detach().cpu().numpy())
            coords_np = np.asarray(coords_np, dtype=np.float32)

            idx_list = self.landmark_sets_idx.get(view, None)
            if idx_list is None:
                idx_np = np.full((coords_np.shape[0],), -1, dtype=np.int64)
            else:
                idx_np = np.array([(-1 if v is None else int(v)) for v in idx_list], dtype=np.int64)

            cam = self.cameras[view]
            R = cam.R.detach().cpu().numpy()
            T = cam.T.detach().cpu().numpy()

            lms2d = self.landmarks_2d_dict.get(view, np.full((68, 2), np.nan, dtype=np.float32)).astype(np.float32)

            out_path = os.path.join(cache_dir, f"{view}.npz")
            np.savez_compressed(out_path, coords=coords_np, idx=idx_np, R=R, T=T, landmarks_2d=lms2d)
            print(f"[cache] saved {out_path}")

    def load_prepared(self, cache_dir):
        self.landmark_sets_3d.clear()
        self.landmark_sets_idx.clear()
        self.cameras.clear()
        self.landmarks_2d_dict.clear()

        for cfg in self.view_configs:
            name = cfg["name"]
            path = os.path.join(cache_dir, f"{name}.npz")
            if not os.path.exists(path):
                print(f"[cache] missing {path}; skip this view")
                continue

            data = np.load(path)
            coords_np = data["coords"]            # (68,3)
            idx_np = data["idx"]                  # (68,)
            R_np = data["R"]
            T_np = data["T"]
            lms2d = data["landmarks_2d"]

            coords_list = []
            for row in coords_np:
                if np.isnan(row).any():
                    coords_list.append(None)
                else:
                    coords_list.append(torch.from_numpy(row).to(self.device).float())

            idx_list = [None if int(v) < 0 else int(v) for v in idx_np.tolist()]
            R = torch.from_numpy(R_np).to(self.device).float()
            T = torch.from_numpy(T_np).to(self.device).float()
            cam = FoVPerspectiveCameras(device=self.device, R=R, T=T)

            self.landmark_sets_3d[name] = coords_list
            self.landmark_sets_idx[name] = idx_list
            self.cameras[name] = cam
            self.landmarks_2d_dict[name] = lms2d
            print(f"[cache] loaded {path}")

    # --------- Depth-based ranking ----------
    def _depth_in_camera(self, cam, coord3d):
        # Transform world→view and return absolute z in camera space
        z = cam.get_world_to_view_transform().transform_points(coord3d.unsqueeze(0))[0, 2].item()
        return abs(float(z))

    def rank_views_by_depth_sum(self):
        results = {}
        for i in range(68):
            cands = []
            for view_name, coords_list in self.landmark_sets_3d.items():
                coord = coords_list[i]
                idx = self.landmark_sets_idx.get(view_name, [None]*68)[i]
                if coord is None or idx is None:
                    continue
                cands.append({"from_view": view_name, "index": int(idx), "coord": coord})
            k = len(cands)
            if k == 0:
                results[i] = {"views": [], "candidates": [], "depth_table": np.zeros((0,0), np.float32), "summary": []}
                continue

            views = [c["from_view"] for c in cands]
            depth_mat = np.zeros((k, k), dtype=np.float32)
            for r, ref_v in enumerate(views):
                ref_cam = self.cameras[ref_v]
                for c_idx, row in enumerate(cands):
                    depth_mat[r, c_idx] = self._depth_in_camera(ref_cam, row["coord"])

            row_sums = depth_mat.sum(axis=1)
            summary = [{"ref_view": views[r], "total_sum": float(row_sums[r])} for r in range(k)]
            summary.sort(key=lambda x: x["total_sum"])
            for rank, s in enumerate(summary):
                s["rank"] = rank

            results[i] = {"views": views, "candidates": cands, "depth_table": depth_mat, "summary": summary}

        self.per_lm_crossview = results
        return results

    # --------- Centroid consensus ----------
    def centroid_selection(self, reject_pct=0.4, min_keep=1):
        if self.per_lm_crossview is None:
            self.rank_views_by_depth_sum()

        results = {}
        for i in range(68):
            entry = self.per_lm_crossview.get(i, None)
            if entry is None or len(entry["views"]) == 0:
                results[i] = {"selected": None, "centroid": None, "inliers": [], "rejected": [], "k_all": 0, "k_kept": 0}
                continue

            cands = entry["candidates"]
            summary = entry["summary"]
            k = len(cands)
            keep_n = max(min_keep, int(round(k * (1.0 - float(reject_pct)))))
            keep_n = max(1, min(keep_n, k))
            kept_ref_views = set([row["ref_view"] for row in summary[:keep_n]])

            inliers, rejected = [], []
            ref_info = {row["ref_view"]: (row["rank"], row["total_sum"]) for row in summary}
            for c in cands:
                v = c["from_view"]
                coord = c["coord"]
                idx = c["index"]
                d_self = self._depth_in_camera(self.cameras[v], coord)
                row = {
                    "view": v, "index": idx, "coord": coord,
                    "depth_self": float(d_self),
                    "rank_crossview": ref_info[v][0],
                    "total_sum_view": float(ref_info[v][1]),
                }
                (inliers if v in kept_ref_views else rejected).append(row)

            if len(inliers) == 0 and len(rejected) > 0:
                best = sorted(rejected, key=lambda x: (x["rank_crossview"], x["depth_self"]))[0]
                inliers = [best]
                rejected = [r for r in rejected if r is not best]

            if len(inliers) == 0:
                results[i] = {"selected": None, "centroid": None, "inliers": [], "rejected": rejected, "k_all": k, "k_kept": 0}
                continue

            coords = torch.stack([r["coord"] for r in inliers], dim=0)
            centroid = coords.mean(dim=0)          # (3,)
            dists = torch.linalg.norm(coords - centroid[None, :], dim=1)
            min_j = int(torch.argmin(dists).item())

            for j, row in enumerate(inliers):
                row["dist_to_centroid"] = float(dists[j].item())
            selected = dict(inliers[min_j])
            results[i] = {
                "selected": selected,
                "centroid": centroid,
                "inliers": inliers,
                "rejected": rejected,
                "k_all": k,
                "k_kept": len(inliers),
            }

        self.consensus = results
        return results

    # --------- Outputs ----------
    def get_consensus_arrays(self):
        if self.consensus is None:
            raise RuntimeError("Call centroid_selection() first.")
        out_coords, out_indices, out_views = [None]*68, [None]*68, [None]*68
        for i in range(68):
            sel = self.consensus[i]["selected"]
            if sel is not None:
                out_coords[i], out_indices[i], out_views[i] = sel["coord"], sel["index"], sel["view"]
        return out_coords, out_indices, out_views

    def print_ranking(self, i, max_rows=None):
        if self.per_lm_crossview is None:
            self.rank_views_by_depth_sum()
        entry = self.per_lm_crossview.get(i, None)
        if entry is None or len(entry["views"]) == 0:
            print(f"LM {i}: (no candidates)")
            return
        views = entry["views"]
        depth_mat = entry["depth_table"]
        summary = entry["summary"]
        view2idx = {row["ref_view"]: row["rank"] for row in summary}

        print(f"LM {i}: cross-view depth sums (rows = ref view, smaller is better)")
        header = "ref\\cand | " + " | ".join([f"{v[:10]:>10}" for v in views])
        print(header)
        print("-" * len(header))
        rows_to_show = range(len(views)) if max_rows is None else range(min(max_rows, len(views)))
        for r in rows_to_show:
            row_str = f"{views[r][:10]:>8} | " + " | ".join([f"{depth_mat[r, c]:10.5f}" for c in range(len(views))])
            print(row_str)
        print("\nTotals (smaller = better):")
        for s in summary:
            v = s["ref_view"]
            print(f"  rank={s['rank']:02d}  view={v:<8}  total_sum={s['total_sum']:.6f}  row={view2idx[v]}")
