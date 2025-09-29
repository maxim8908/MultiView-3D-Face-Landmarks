import torch

class RasterizerLandmarkProjector:
    """
    Project 2D pixel landmarks to 3D points on the mesh surface using
    the rasterized fragments (face index + barycentric). We rasterize
    once and then index into the fragments at landmark pixel coords.

    Assumes:
      - rasterizer is MeshRasterizer with the same image_size as used for 2D detection
      - mesh is textured Meshes with verts/faces on device
      - landmarks_2d are in pixel coords (x,y) with origin at top-left
    """

    def __init__(self, mesh, rasterizer, image_size, device="cpu"):
        self.mesh = mesh
        self.rasterizer = rasterizer
        self.device = torch.device(device)
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = (int(image_size[0]), int(image_size[1]))

        # Rasterize once, keep fragments for lookup
        with torch.no_grad():
            fragments = self.rasterizer(self.mesh)
        self.pix_to_face = fragments.pix_to_face[0]     # (H,W,K)
        self.bary_coords = fragments.bary_coords[0]     # (H,W,K,3)
        self.K = self.pix_to_face.shape[-1]

        # Packed verts/faces for interpolation
        self.verts = self.mesh.verts_packed()           # (V,3)
        self.faces = self.mesh.faces_packed()           # (F,3)

    def _coord_to_hw_index(self, x, y):
        # Clamp to image bounds
        H, W = self.image_size[0], self.image_size[1]
        xi = max(0, min(int(round(x)), W - 1))
        yi = max(0, min(int(round(y)), H - 1))
        return yi, xi

    def _face_point(self, face_idx, bary):
        """
        Interpolate 3D point from a face and barycentric coords.
        """
        vids = self.faces[face_idx]          # (3,)
        v = self.verts[vids]                 # (3,3)
        p = (v * bary.unsqueeze(1)).sum(dim=0)  # (3,)
        return p

    def project_landmarks(self, landmarks_2d):
        """
        landmarks_2d: (68,2) numpy or torch, in pixel coords.
        Returns:
            coords_3d: list of 68 items (torch(3,) or None)
            indices_3d: list of 68 items (int face vertex index OR packed vertex id OR None)
                        Here we return the 'winning face index' for traceability.
        """
        if landmarks_2d is None:
            return [None]*68, [None]*68

        if not torch.is_tensor(landmarks_2d):
            lm = torch.from_numpy(landmarks_2d).float()
        else:
            lm = landmarks_2d.float()
        coords_out, idx_out = [], []

        H, W = self.image_size
        for i in range(lm.shape[0]):
            x, y = lm[i, 0].item(), lm[i, 1].item()
            yi, xi = self._coord_to_hw_index(x, y)

            # Use top-1 face (K=0). If -1: background → no hit.
            face_idx = int(self.pix_to_face[yi, xi, 0].item())
            if face_idx < 0:
                coords_out.append(None)
                idx_out.append(None)
                continue

            bary = self.bary_coords[yi, xi, 0]          # (3,)
            p3d = self._face_point(face_idx, bary)      # (3,)
            coords_out.append(p3d.detach())
            idx_out.append(face_idx)

        return coords_out, idx_out
