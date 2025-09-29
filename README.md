# MultiView-3D-Face-Landmarks

Robust multi-view 3D face landmark consensus using **PyTorch3D**.  
This pipeline renders a mesh from multiple camera views, detects **2D 68-point landmarks** (face-alignment), projects them back to **3D** via rasterizer fragments, then ranks/filters candidates across views by **depth** and chooses a final per-landmark **centroid** candidate.

## Features
- Textured rendering with PyTorch3D
- 2D landmark detection (face-alignment)
- 2D→3D projection using rasterizer fragments (face + barycentric)
- Cross-view depth table & ranking
- Centroid consensus with configurable rejection
- NPZ caching of per-view results

## Requirements

You need Python ≥ 3.9 and the following packages:

| Package | Purpose |
|--------|--------|
| **torch** | Core deep learning framework |
| **pytorch3d** | 3D rendering, rasterization, and geometry ops |
| **numpy** | Array and numerical operations |
| **matplotlib** | Visualization and plotting |
| **face-alignment** | 2D landmark detection |
| **Pillow** | Image processing (texture loading & saving) |
| **imageio** | Image I/O utilities |
| **scikit-learn** *(optional)* | PCA / evaluation helpers (if needed) |

## Install
```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
