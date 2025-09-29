import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer, MeshRasterizer,
    SoftPhongShader, PointLights,
    look_at_view_transform, TexturesUV
)

def _to_image_size_tuple(image_size):
    if isinstance(image_size, int):
        return (image_size, image_size)
    assert isinstance(image_size, (tuple, list)) and len(image_size) == 2
    return tuple(int(x) for x in image_size)

class MeshRenderer2D:
    """
    Minimal textured mesh renderer wrapper around PyTorch3D for consistent camera/rasterizer access.
    """

    def __init__(self, mesh_path, texture_path, device=None, image_size=512, fov=30.0):
        self.device = torch.device(device) if device is not None else (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.mesh_path = mesh_path
        self.texture_path = texture_path
        self.image_size = _to_image_size_tuple(image_size)
        self.fov = float(fov)

        self.mesh = None
        self.cameras = None
        self.renderer = None
        self.raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        self.lights = PointLights(device=self.device, location=[[0.0, 0.0, 30.0]])

    # ---------- Mesh ----------
    def load_mesh(self):
        verts, faces, aux = load_obj(self.mesh_path, load_textures=True)
        if aux.verts_uvs is None or faces.textures_idx is None:
            raise ValueError("OBJ must have UVs and texture indices for TexturesUV.")
        verts_uvs = aux.verts_uvs[None, ...]                 # (1, Nv, 2)
        faces_uvs = faces.textures_idx[None, ...]            # (1, F, 3)
        faces_idx = faces.verts_idx                          # (F, 3)

        tex_img = Image.open(self.texture_path).convert("RGB")
        tex_img = torch.from_numpy(np.asarray(tex_img)).float() / 255.0  # (H,W,3)
        tex_img = tex_img[None, ...].to(self.device)                     # (1,H,W,3)

        textures = TexturesUV(maps=tex_img,
                              faces_uvs=faces_uvs.to(self.device),
                              verts_uvs=verts_uvs.to(self.device))
        self.mesh = Meshes(
            verts=[verts.to(self.device)],
            faces=[faces_idx.to(self.device)],
            textures=textures
        )

    # ---------- Camera / Renderer ----------
    def set_camera(self, dist=5.0, elev=0.0, azim=0.0):
        R, T = look_at_view_transform(dist=float(dist), elev=float(elev), azim=float(azim))
        self.cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T, fov=self.fov)
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=self.cameras, raster_settings=self.raster_settings),
            shader=SoftPhongShader(device=self.device, cameras=self.cameras, lights=self.lights),
        )

    def get_camera(self):
        return self.cameras

    def get_rasterizer(self):
        return self.renderer.rasterizer

    def get_mesh(self):
        return self.mesh

    # ---------- Render ----------
    def render(self):
        if self.mesh is None or self.renderer is None:
            raise RuntimeError("Call load_mesh() and set_camera() before render().")
        images = self.renderer(self.mesh)   # (1,H,W,4)
        return images[0, ..., :3].detach().cpu().numpy()

    def render_torch(self):
        if self.mesh is None or self.renderer is None:
            raise RuntimeError("Call load_mesh() and set_camera() before render().")
        images = self.renderer(self.mesh)   # (1,H,W,4)
        return images[0, ..., :3]

    # ---------- Utils ----------
    def show(self, image):
        plt.imshow(image)
        plt.axis("off")
        plt.show()

    def save(self, image, output_dir, elev, azim):
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(self.mesh_path))[0]
        fname = f"{base}_texture_rendered_e{int(elev)}_a{int(azim)}.png"
        out = os.path.join(output_dir, fname)
        img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(img_uint8).save(out)
        print(f"[renderer] saved {out}")
