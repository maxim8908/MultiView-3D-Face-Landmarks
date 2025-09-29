import numpy as np
from src.multiview_landmarks import MultiViewLandmarkZBufferConsensus

mesh_path = r'\\path\\to\\geo_scan.obj'
texture_path = r'\\path\\to\\geo_scan.jpg'
cache_dir = r'\\path\\to\\cache_npz'

view_configs = [
    {"name": "0", "elev": 0, "azim": 0},
    {"name": "15", "elev": 0, "azim": 15},
    {"name": "30", "elev": 0, "azim": 30},
    {"name": "45", "elev": 0, "azim": 45},
    {"name": "60", "elev": 0, "azim": 60},
    {"name": "300", "elev": 0, "azim": 300},
    {"name": "315", "elev": 0, "azim": 315},
    {"name": "330", "elev": 0, "azim": 330},
    {"name": "345", "elev": 0, "azim": 345},
]

mvc = MultiViewLandmarkZBufferConsensus(mesh_path, texture_path, view_configs, image_size=(1024,1024), device="cuda:0")
mvc.prepare_views(dist=55.0, fov=30.0)
mvc.save_prepared(cache_dir)

# Or load later:
# mvc.load_prepared(cache_dir)

mvc.rank_views_by_depth_sum()
mvc.centroid_selection(reject_pct=0.6, min_keep=1)
coords, indices, views = mvc.get_consensus_arrays()
print(indices)
