# Usage: python evaluate_stereo_edges.py --restore_ckpt 
#        ../pretrained_models/Selective-IGEV/middlebury/middlebury_finetune.pth 
#        --edges_txt_dir /gpfs/data/bkimia/Datasets/Middlebury_Stereo/scenes2021/data/ 
#        --output_dir /gpfs/data/bkimia/cchien3/Learning_Based_Stereo_Matching/Selective-Stereo/ 
#        --scene_name chess1

from __future__ import print_function, division
import sys
sys.path.append('core')

import argparse
import time
import logging
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from core.igev_stereo import IGEVStereo, autocast
import core.stereo_datasets as datasets
from core.utils.utils import InputPadder

try:
    # Prefer lightweight colormap usage without pyplot
    from matplotlib import cm
except Exception:
    cm = None

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _scene_from_image_path(imageL_file: str, dataset_type: str = 'middlebury') -> str:
    """Extract scene name from image path."""
    p = Path(imageL_file)
    parts = p.parts
    
    if dataset_type == 'eth3d':
        # ETH3D paths: .../stereo_pairs/<scene_name>/im0.png
        return p.parent.name
    else:
        # Middlebury-2021 style paths
        return p.parents[2].name if "ambient" in parts else p.parent.name


def _load_edges_from_txt(txt_path: str):
    """
    Load edge coordinates from a .txt file.
    Format: one coordinate pair per line as "y x" (space-separated) or "y,x" (comma-separated).
    Supports subpixel float coordinates.
    
    Returns:
        numpy array of shape (N, 2) with [y, x] coordinates, or None if file doesn't exist
    """
    if not os.path.exists(txt_path):
        return None
    
    coords = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):  # Skip empty lines and comments
                continue
            # Try space-separated first, then comma-separated
            if ',' in line:
                parts = line.split(',')
            else:
                parts = line.split()
            if len(parts) >= 2:
                try:
                    y = float(parts[0].strip())
                    x = float(parts[1].strip())
                    orient = float(parts[2].strip())
                    coords.append([y, x, orient])
                    # coords.append([y, x])
                except ValueError:
                    logging.warning(f"Skipping invalid line in {txt_path}: {line}")
    
    if len(coords) == 0:
        return None
    return np.asarray(coords, dtype=np.float64)


def _find_edges_txt_for_image(edges_txt_dir: str, imageL_file: str, dataset_type: str = 'middlebury'):
    """
    Find the corresponding .txt file for an image.
    Tries multiple naming conventions:
    1. <edges_txt_dir>/<scene>/edges.txt
    2. <edges_txt_dir>/<scene>.txt
    3. <edges_txt_dir>/<image_basename>.txt
    """
    scene = _scene_from_image_path(imageL_file, dataset_type=dataset_type)
    image_basename = Path(imageL_file).stem  # e.g., "im0"
    
    # Try scene-based paths
    candidates = [
        Path(edges_txt_dir) / scene / "edges.txt",
        Path(edges_txt_dir) / f"{scene}.txt",
        Path(edges_txt_dir) / f"{image_basename}.txt",
    ]
    
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    
    return None


def _load_edges_mask_for_scene(mask_dir: str, scene: str, size_hw: tuple, mask_filename: str):
    """Load a per-scene edges mask image and return boolean mask with given HxW."""
    # Expect mask at <mask_dir>/<scene>/<mask_filename>
    candidate = Path(mask_dir) / scene / mask_filename
    if not candidate.exists():
        return None
    mask_img = Image.open(str(candidate)).convert('L')
    if mask_img.size[::-1] != size_hw:
        mask_img = mask_img.resize((size_hw[1], size_hw[0]), resample=Image.NEAREST)
    mask_np = np.array(mask_img)
    # Non-zero is edge
    return mask_np > 0


def _make_edges_mask_from_coords(coords: np.ndarray, size_hw: tuple):
    """Build boolean mask from array of [y, x] coords (supports subpixel float coords)."""
    h, w = size_hw
    mask = np.zeros((h, w), dtype=bool)
    if coords is None or len(coords) == 0:
        return mask
    # Clip to valid image bounds (accounting for subpixel locations)
    ys = np.clip(coords[:, 0], 0, h - 1)
    xs = np.clip(coords[:, 1], 0, w - 1)
    # Round to nearest pixel for mask visualization
    mask[np.round(ys).astype(int), np.round(xs).astype(int)] = True
    return mask


def _sample_epe_at_subpixel(epe_map: np.ndarray, coords: np.ndarray):
    """
    Sample EPE values at subpixel coordinates using bilinear interpolation.
    
    Args:
        epe_map: HxW array of EPE values
        coords: Nx2 array of [y, x] coordinates (float, can be subpixel)
    
    Returns:
        N-length array of interpolated EPE values
    """
    if coords is None or len(coords) == 0:
        return np.array([], dtype=np.float32)
    
    h, w = epe_map.shape
    ys = np.clip(coords[:, 0], 0, h - 1)
    xs = np.clip(coords[:, 1], 0, w - 1)
    
    # Get integer floor coordinates
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    
    # Fractional parts for interpolation
    dy = ys - y0
    dx = xs - x0
    
    # Bilinear interpolation weights
    w00 = (1 - dx) * (1 - dy)
    w01 = (1 - dx) * dy
    w10 = dx * (1 - dy)
    w11 = dx * dy
    
    # Sample values at four corners
    v00 = epe_map[y0, x0]
    v01 = epe_map[y1, x0]
    v10 = epe_map[y0, x1]
    v11 = epe_map[y1, x1]
    
    # Interpolate
    values = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11
    return values.astype(np.float32)

def _sample_disparity_at_subpixel_blinear_interpolation(disp_map: np.ndarray, coords: np.ndarray):
    """
    Sample disparity values at subpixel coordinates using bilinear interpolation.
    
    Args:
        disp_map: HxW array of disparity values
        coords: Nx2 array of [y, x] coordinates (float, can be subpixel)
    
    Returns:
        N-length array of interpolated disparity values
    """
    if coords is None or len(coords) == 0:
        return np.array([], dtype=np.float32)
    
    h, w = disp_map.shape
    ys = np.clip(coords[:, 0], 0, h - 1)
    xs = np.clip(coords[:, 1], 0, w - 1)
    
    # Get integer floor coordinates
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    
    # Fractional parts for interpolation
    dy = ys - y0
    dx = xs - x0
    
    # Bilinear interpolation weights
    w00 = (1 - dx) * (1 - dy)
    w01 = (1 - dx) * dy
    w10 = dx * (1 - dy)
    w11 = dx * dy
    
    # Sample values at four corners
    v00 = disp_map[y0, x0]
    v01 = disp_map[y1, x0]
    v10 = disp_map[y0, x1]
    v11 = disp_map[y1, x1]
    
    # Interpolate
    values = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11
    return values.astype(np.float32)

def _sample_disparity_at_subpixel(disp_map: np.ndarray, coords: np.ndarray):
    """
    Sample disparity values at subpixel coordinates by rounding to nearest integer pixel.
    
    Args:
        disp_map: HxW array of disparity values
        coords: Nx2 array of [y, x] coordinates (float, can be subpixel)
    
    Returns:
        N-length array of disparity values at rounded coordinates
    """
    if coords is None or len(coords) == 0:
        return np.array([], dtype=np.float32)
    
    h, w = disp_map.shape
    
    # Round coordinates to nearest integer and clip to valid bounds
    y_round = np.round(np.clip(coords[:, 0], 0, h - 1)).astype(int)
    x_round = np.round(np.clip(coords[:, 1], 0, w - 1)).astype(int)
    
    # Sample disparity at rounded integer coordinates
    values = disp_map[y_round, x_round]
    return values.astype(np.float32)


def _render_heatmap(epe_map: np.ndarray, edges_mask: np.ndarray, use_percentile=True, percentile=95):
    """
    Return color heatmap (HxWx3 uint8) for EPE values on edges with enhanced visibility.
    Uses percentile-based normalization to highlight hotspots. High EPE = bright colors.
    """
    h, w = epe_map.shape
    values = np.zeros((h, w), dtype=np.float32)
    values[edges_mask] = epe_map[edges_mask]
    
    # Get valid (non-zero) values for normalization
    valid_values = values[edges_mask]
    
    if len(valid_values) == 0 or valid_values.max() <= 0:
        # Return dark background if no valid values
        return np.zeros((h, w, 3), dtype=np.uint8)
    
    # Normalize using percentile to highlight hotspots
    # Use percentile as vmax to prevent extreme outliers from washing out the rest
    if use_percentile and len(valid_values) > 0:
        vmax = np.percentile(valid_values, percentile)
        vmin = valid_values.min()
        # Don't clip - allow values above percentile to saturate (become brightest)
        # This ensures high EPE values are always bright
        if vmax > vmin:
            # Normalize: values at or above vmax become 1.0 (brightest)
            norm = np.clip((values - vmin) / (vmax - vmin + 1e-8), 0, 1)
        else:
            # All values are the same
            norm = np.ones_like(values) if valid_values.max() > 0 else np.zeros_like(values)
    else:
        # Linear normalization - high values become bright
        vmax = valid_values.max()
        vmin = valid_values.min()
        if vmax > vmin:
            norm = (values - vmin) / (vmax - vmin + 1e-8)
        else:
            norm = np.ones_like(values) if valid_values.max() > 0 else np.zeros_like(values)
    
    # Use linear mapping (no gamma) to preserve color distinction across EPE range
    # This ensures different EPE values map to different colors
    # norm: 0.0 = lowest EPE (blue), 1.0 = highest EPE (red)
    norm = np.clip(norm, 0, 1)
    
    if cm is None:
        # Enhanced grayscale fallback with better contrast
        gray = (norm * 255.0).astype(np.uint8)
        return np.stack([gray, gray, gray], axis=-1)
    
    # Create custom colormap: blue -> cyan -> green -> yellow -> orange -> red
    # This gives: low EPE (0.0) = blue, high EPE (1.0) = red
    # LinearSegmentedColormap maps: 0.0 -> first color, 1.0 -> last color
    try:
        from matplotlib.colors import LinearSegmentedColormap
        # Define colors from low EPE (blue) to high EPE (red)
        colors = [
            (0.0, 0.0, 1.0),    # Blue (lowest EPE, norm=0.0)
            (0.0, 1.0, 1.0),    # Cyan
            (0.0, 1.0, 0.0),    # Green
            (1.0, 1.0, 0.0),    # Yellow
            (1.0, 0.5, 0.0),    # Orange
            (1.0, 0.0, 0.0),    # Red (highest EPE, norm=1.0)
        ]
        n_bins = 256
        cmap = LinearSegmentedColormap.from_list('blue_to_red', colors, N=n_bins)
    except:
        # Fallback: use 'coolwarm' which goes from blue to red (we'll reverse it)
        # Actually, we want red (high) to blue (low), so we use coolwarm as-is
        # since coolwarm goes blue->red, and we want the opposite
        try:
            cmap = cm.get_cmap('coolwarm')
        except:
            cmap = cm.get_cmap('RdBu_r')
    
    rgba = cmap(norm)  # HxWx4, float [0,1]
    rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
    
    # Make background black (where edges_mask is False) for better contrast
    background = np.zeros((h, w, 3), dtype=np.uint8)
    rgb = np.where(edges_mask[..., np.newaxis], rgb, background)
    
    return rgb


def _overlay_on_image(image_rgb: np.ndarray, heatmap_rgb: np.ndarray, alpha: float = 0.6):
    """Overlay heatmap on top of image."""
    image_f = image_rgb.astype(np.float32)
    heat_f = heatmap_rgb.astype(np.float32)
    overlay = (1 - alpha) * image_f + alpha * heat_f
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay


@torch.no_grad()
def validate_middlebury_edges(model, iters=32, mixed_prec=False, resolution='F',
                              edges_txt_dir=None, edges_mask_dir=None, mask_filename='edges.png',
                              output_dir=None, scene_name=None):
    """Validation on Middlebury with EPE computed only at provided edge locations."""
    assert output_dir is not None, "--output_dir is required"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "epe_values"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "epe_distributions"), exist_ok=True)

    model.eval()
    aug_params = {}
    val_dataset = datasets.Middlebury(aug_params, split='scenes2021', resolution=resolution)
    
    logging.info(f"Loaded Middlebury dataset with {len(val_dataset)} images")

    out_list, epe_list = [], []
    processed_count = 0
    skipped_count = 0
    for val_id in range(len(val_dataset)):
        (imageL_file, _, _), image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        
        # Filter by scene name if specified
        if scene_name is not None:
            current_scene = _scene_from_image_path(imageL_file, dataset_type='middlebury')
            if current_scene != scene_name:
                skipped_count += 1
                continue
        
        print(f"[{val_id+1}/{len(val_dataset)}] Processing image: {imageL_file}")

        processed_count += 1
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow_pr = padder.unpad(flow_pr.float()).cpu().squeeze(0)
        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)

        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()  # HxW
        epe_np = epe.numpy()
        H, W = epe_np.shape

        # Middlebury 2021 non-occlusion mask
        occ_mask_img = Image.open(imageL_file.replace('im0.png', 'mask0nocc.png')).convert('L')
        occ_mask = np.ascontiguousarray(occ_mask_img, dtype=np.float32)
        valid = (np.asarray(valid_gt, dtype=np.float32) >= 0.5) & (occ_mask == 255)

        # Build edges mask and get coordinates
        edges_mask = np.zeros((H, W), dtype=bool)
        edge_coords = None
        if edges_txt_dir is not None:
            txt_path = _find_edges_txt_for_image(edges_txt_dir, imageL_file, dataset_type='middlebury')
            if txt_path is not None:
                edge_coords = _load_edges_from_txt(txt_path)
                if edge_coords is not None:
                    # Create mask for visualization (rounded to nearest pixel)
                    edges_mask = _make_edges_mask_from_coords(edge_coords, (H, W))
        elif edges_mask_dir is not None:
            scene = _scene_from_image_path(imageL_file)
            mask_img = _load_edges_mask_for_scene(edges_mask_dir, scene, (H, W), mask_filename)
            if mask_img is not None:
                edges_mask = mask_img.astype(bool)

        # Compute EPE at edge locations and save valid subpixel edges with GT disparities
        if edge_coords is not None and len(edge_coords) > 0:
            # Get ground-truth disparity (flow_gt is [1, H, W], extract first channel)
            gt_disp_np = flow_gt[0].cpu().numpy() if isinstance(flow_gt, torch.Tensor) else flow_gt[0]
            
            # Use subpixel sampling for coordinate-based edges
            epe_at_edges = _sample_epe_at_subpixel(epe_np, edge_coords)
            gt_disp_at_edges = _sample_disparity_at_subpixel(gt_disp_np, edge_coords)
            
            # Check validity at subpixel locations (using nearest pixel for validity check)
            y_round = np.round(np.clip(edge_coords[:, 0], 0, H - 1)).astype(int)
            x_round = np.round(np.clip(edge_coords[:, 1], 0, W - 1)).astype(int)
            valid_at_edges = valid[y_round, x_round]
            
            # Filter out NaN and inf values from EPE
            finite_mask_epe = np.isfinite(epe_at_edges)
            # Filter out NaN and inf values from GT disparity
            finite_mask_disp = np.isfinite(gt_disp_at_edges)
            # Combined validity: valid mask AND finite EPE AND finite GT disparity
            valid_mask = valid_at_edges & finite_mask_epe & finite_mask_disp
            
            # Get valid edges with their original subpixel coordinates and GT disparities
            valid_edge_coords = edge_coords[valid_mask]
            valid_gt_disparities = gt_disp_at_edges[valid_mask]
            
            # Save valid subpixel edges with GT disparities
            current_scene = _scene_from_image_path(imageL_file, dataset_type='middlebury')
            base_name = Path(imageL_file).stem  # im0
            valid_edges_path = os.path.join(output_dir, "epe_values", f"{current_scene}_{base_name}_valid_edges_gt_disp.txt")
            with open(valid_edges_path, 'w') as f:
                f.write(f"# Valid subpixel edge coordinates with ground-truth disparities for {current_scene} {base_name}\n")
                f.write(f"# Total valid edges: {len(valid_edge_coords)}\n")
                if len(valid_gt_disparities) > 0:
                    f.write(f"# GT Disparity stats - Min: {valid_gt_disparities.min():.6f}, Max: {valid_gt_disparities.max():.6f}, Mean: {valid_gt_disparities.mean():.6f}\n")
                f.write("# Format: y x orient gt_disparity (one per line, space-separated)\n")
                f.write("# Coordinates are subpixel (float values)\n")
                for coord, gt_disp in zip(valid_edge_coords, valid_gt_disparities):
                    f.write(f"{coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f} {gt_disp:.6f}\n")
            
            logging.info(f"Saved {len(valid_edge_coords)} valid subpixel edges with GT disparities")
            
            # Continue with EPE computation
            epe_valid = epe_at_edges[valid_mask]
            if len(epe_valid) > 0:
                out_valid = (epe_valid > 2.0)
                image_out = out_valid.mean().item()
                image_epe = epe_valid.mean().item()
            else:
                image_out = 0.0
                image_epe = 0.0
        else:
            # Use mask-based approach for mask-based edges
            val = (valid & edges_mask).reshape(-1)
            epe_flat = epe.reshape(-1)
            epe_valid_flat = epe_flat[val]
            # Convert to numpy and filter out NaN and inf values
            epe_valid_np = epe_valid_flat.numpy() if isinstance(epe_valid_flat, torch.Tensor) else epe_valid_flat
            finite_mask = np.isfinite(epe_valid_np)
            epe_valid_np = epe_valid_np[finite_mask]
            if len(epe_valid_np) > 0:
                out = (epe_valid_np > 2.0)
                image_out = out.mean().item()
                image_epe = epe_valid_np.mean().item()
            else:
                image_out = 0.0
                image_epe = 0.0

        edge_count = len(edge_coords) if edge_coords is not None and len(edge_coords) > 0 else int(edges_mask.sum())
        current_scene = _scene_from_image_path(imageL_file, dataset_type='middlebury')
        total_info = f"{processed_count}" if scene_name else f"{val_id+1}/{len(val_dataset)}"
        logging.info(f"[{total_info}] {Path(imageL_file).name} scene={current_scene} edges_count={edge_count} EPE {round(image_epe,4)} D1 {round(image_out,4)}")
        epe_list.append(image_epe)
        out_list.append(image_out)

        # Save EPE values to text files with coordinates
        base = Path(imageL_file).stem  # im0
        scene = _scene_from_image_path(imageL_file, dataset_type='middlebury')
        
        # Get edge pixels with coordinates (valid pixels only)
        edge_mask_valid = edges_mask & valid
        edge_indices = np.where(edge_mask_valid)
        edge_y_coords = edge_indices[0]  # row coordinates
        edge_x_coords = edge_indices[1]  # column coordinates
        edge_epe_values = epe_np[edge_mask_valid]
        
        # Get non-edge pixels with coordinates (valid pixels only)
        non_edge_mask = ~edges_mask
        non_edge_mask_valid = non_edge_mask & valid
        non_edge_indices = np.where(non_edge_mask_valid)
        non_edge_y_coords = non_edge_indices[0]  # row coordinates
        non_edge_x_coords = non_edge_indices[1]  # column coordinates
        non_edge_epe_values = epe_np[non_edge_mask_valid]
        
        # Save edge EPE values with coordinates
        edge_epe_path = os.path.join(output_dir, "epe_values", f"{scene}_{base}_edge_epe.txt")
        with open(edge_epe_path, 'w') as f:
            f.write(f"# EPE values at edge locations for {scene} {base}\n")
            f.write(f"# Total edge pixels (valid): {len(edge_epe_values)}\n")
            f.write(f"# Min: {edge_epe_values.min():.6f}, Max: {edge_epe_values.max():.6f}, Mean: {edge_epe_values.mean():.6f}\n")
            f.write("# Format: y x epe_value (one per line, space-separated)\n")
            for y, x, epe_val in zip(edge_y_coords, edge_x_coords, edge_epe_values):
                f.write(f"{y} {x} {epe_val:.6f}\n")
        
        # Save non-edge EPE values with coordinates
        non_edge_epe_path = os.path.join(output_dir, "epe_values", f"{scene}_{base}_non_edge_epe.txt")
        with open(non_edge_epe_path, 'w') as f:
            f.write(f"# EPE values at non-edge locations for {scene} {base}\n")
            f.write(f"# Total non-edge pixels (valid): {len(non_edge_epe_values)}\n")
            f.write(f"# Min: {non_edge_epe_values.min():.6f}, Max: {non_edge_epe_values.max():.6f}, Mean: {non_edge_epe_values.mean():.6f}\n")
            f.write("# Format: y x epe_value (one per line, space-separated)\n")
            for y, x, epe_val in zip(non_edge_y_coords, non_edge_x_coords, non_edge_epe_values):
                f.write(f"{y} {x} {epe_val:.6f}\n")
        
        logging.info(f"Saved EPE values: {len(edge_epe_values)} edge pixels, {len(non_edge_epe_values)} non-edge pixels")
        
        # Visualize EPE probability distributions
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # Left plot: Histogram comparison
            ax1 = axes[0]
            if len(edge_epe_values) > 0 and len(non_edge_epe_values) > 0:
                # Determine appropriate bins
                all_epe = np.concatenate([edge_epe_values, non_edge_epe_values])
                bins = np.linspace(0, min(np.percentile(all_epe, 99), all_epe.max()), 100)
                
                ax1.hist(edge_epe_values, bins=bins, alpha=0.7, label='Edge', color='red', 
                        density=True, edgecolor='black', linewidth=0.5)
                ax1.hist(non_edge_epe_values, bins=bins, alpha=0.7, label='Non-edge', color='blue', 
                        density=True, edgecolor='black', linewidth=0.5)
                ax1.set_xlabel('EPE (pixels)', fontsize=12, fontweight='bold')
                ax1.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
                ax1.set_title(f'EPE Distribution Comparison - {scene}\nEdge vs Non-edge', 
                            fontsize=13, fontweight='bold', pad=10)
                ax1.legend(fontsize=11, loc='upper right')
                ax1.tick_params(labelsize=18)  # Make axis numbers larger
                ax1.grid(True, alpha=0.3)
                ax1.set_xlim(0, min(np.percentile(all_epe, 99), all_epe.max()))
            
            # Right plot: Cumulative Distribution Function (CDF)
            ax2 = axes[1]
            if len(edge_epe_values) > 0 and len(non_edge_epe_values) > 0:
                sorted_edge = np.sort(edge_epe_values)
                sorted_non_edge = np.sort(non_edge_epe_values)
                p_edge = np.arange(1, len(sorted_edge) + 1) / len(sorted_edge)
                p_non_edge = np.arange(1, len(sorted_non_edge) + 1) / len(sorted_non_edge)
                
                ax2.plot(sorted_edge, p_edge, label='Edge', color='red', linewidth=2)
                ax2.plot(sorted_non_edge, p_non_edge, label='Non-edge', color='blue', linewidth=2)
                ax2.set_xlabel('EPE (pixels)', fontsize=20, fontweight='bold')
                ax2.set_ylabel('Cumulative Probability', fontsize=20, fontweight='bold')
                ax2.set_title(f'Cumulative Distribution Function - {scene}', 
                            fontsize=20, fontweight='bold', pad=10)
                ax2.legend(fontsize=20, loc='lower right')
                ax2.tick_params(labelsize=18)  # Make axis numbers larger
                ax2.grid(True, alpha=0.3)
                ax2.set_xlim(0, min(np.percentile(all_epe, 99), all_epe.max()))
            
            plt.tight_layout()
            dist_path = os.path.join(output_dir, "epe_distributions", f"{scene}_{base}_epe_distribution.png")
            plt.savefig(dist_path, dpi=200, bbox_inches='tight')
            plt.close()
            
            # Also create a detailed statistics plot
            fig, ax = plt.subplots(figsize=(12, 8))
            if len(edge_epe_values) > 0 and len(non_edge_epe_values) > 0:
                # Create more detailed histogram with statistics
                bins = np.linspace(0, min(np.percentile(all_epe, 99), all_epe.max()), 150)
                
                n_edge, bins_edge, patches_edge = ax.hist(edge_epe_values, bins=bins, alpha=0.6, 
                                                         label='Edge', color='red', density=True, 
                                                         edgecolor='darkred', linewidth=0.3)
                n_non_edge, bins_non_edge, patches_non_edge = ax.hist(non_edge_epe_values, bins=bins, 
                                                                      alpha=0.6, label='Non-edge', 
                                                                      color='blue', density=True, 
                                                                      edgecolor='darkblue', linewidth=0.3)
                
                # Add vertical lines for mean values
                mean_edge = edge_epe_values.mean()
                mean_non_edge = non_edge_epe_values.mean()
                ax.axvline(mean_edge, color='red', linestyle='--', linewidth=2, 
                          label=f'Edge Mean: {mean_edge:.3f}')
                ax.axvline(mean_non_edge, color='blue', linestyle='--', linewidth=2, 
                          label=f'Non-edge Mean: {mean_non_edge:.3f}')
                
                # Add text box with statistics
                stats_text = f'Edge Statistics:\n'
                stats_text += f'  Mean: {mean_edge:.4f}\n'
                stats_text += f'  Median: {np.median(edge_epe_values):.4f}\n'
                stats_text += f'  Std: {edge_epe_values.std():.4f}\n'
                stats_text += f'  Min: {edge_epe_values.min():.4f}\n'
                stats_text += f'  Max: {edge_epe_values.max():.4f}\n'
                stats_text += f'  Count: {len(edge_epe_values)}\n\n'
                stats_text += f'Non-edge Statistics:\n'
                stats_text += f'  Mean: {mean_non_edge:.4f}\n'
                stats_text += f'  Median: {np.median(non_edge_epe_values):.4f}\n'
                stats_text += f'  Std: {non_edge_epe_values.std():.4f}\n'
                stats_text += f'  Min: {non_edge_epe_values.min():.4f}\n'
                stats_text += f'  Max: {non_edge_epe_values.max():.4f}\n'
                stats_text += f'  Count: {len(non_edge_epe_values)}'
                
                ax.text(0.98, 0.5, stats_text, transform=ax.transAxes, 
                       fontsize=10, verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
                ax.set_xlabel('EPE (pixels)', fontsize=13, fontweight='bold')
                ax.set_ylabel('Probability Density', fontsize=13, fontweight='bold')
                ax.set_title(f'EPE Probability Distribution - {scene} {base}', 
                           fontsize=14, fontweight='bold', pad=15)
                ax.legend(fontsize=11, loc='upper right')
                ax.tick_params(labelsize=18)  # Make axis numbers larger
                ax.grid(True, alpha=0.3)
                ax.set_xlim(0, min(np.percentile(all_epe, 99), all_epe.max()))
            
            plt.tight_layout()
            detailed_dist_path = os.path.join(output_dir, "epe_distributions", 
                                            f"{scene}_{base}_epe_distribution_detailed.png")
            plt.savefig(detailed_dist_path, dpi=200, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Saved EPE distribution plots: {dist_path}, {detailed_dist_path}")
        except Exception as e:
            logging.warning(f"Could not create EPE distribution plots: {e}")

    logging.info(f"Processed {processed_count} images, skipped {skipped_count} images")
    logging.info(f"EPE list length: {len(epe_list)}, Out list length: {len(out_list)}")
    
    if len(epe_list) == 0:
        logging.warning("No images were processed! Check if:")
        logging.warning("  1. Dataset is empty")
        logging.warning("  2. Scene name filter is too restrictive")
        logging.warning("  3. Edge files are missing or not found")
        return {f'middlebury-edges-epe': 0.0, f'middlebury-edges-d1': 0.0}
    
    epe_list = np.array(epe_list)
    out_list = np.array(out_list)
    # Use nanmean to handle any remaining NaN values
    epe = float(np.nanmean(epe_list)) if len(epe_list) > 0 else 0.0
    d1 = 100.0 * float(np.nanmean(out_list)) if len(out_list) > 0 else 0.0
    print(f"Validation Middlebury{resolution} (edges only): EPE {epe}, D1 {d1}")
    return {f'middlebury-edges-epe': epe, f'middlebury-edges-d1': d1}


@torch.no_grad()
def validate_eth3d_edges(model, iters=32, mixed_prec=False,
                        edges_txt_dir=None, edges_mask_dir=None, mask_filename='edges.png',
                        output_dir=None, scene_name=None):
    """Validation on ETH3D with EPE computed only at provided edge locations."""
    assert output_dir is not None, "--output_dir is required"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "epe_values"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "epe_distributions"), exist_ok=True)

    model.eval()
    aug_params = {}
    val_dataset = datasets.ETH3D(aug_params)
    
    logging.info(f"Loaded ETH3D dataset with {len(val_dataset)} images")

    out_list, epe_list = [], []
    processed_count = 0
    skipped_count = 0
    for val_id in range(len(val_dataset)):
        (imageL_file, imageR_file, GT_file), image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        
        # Filter by scene name if specified
        if scene_name is not None:
            current_scene = _scene_from_image_path(imageL_file, dataset_type='eth3d')
            if current_scene != scene_name:
                skipped_count += 1
                continue
        
        processed_count += 1
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow_pr = padder.unpad(flow_pr.float()).cpu().squeeze(0)
        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)

        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()  # HxW
        epe_np = epe.numpy()
        H, W = epe_np.shape

        # ETH3D non-occlusion mask
        occ_mask = Image.open(GT_file.replace('disp0GT.pfm', 'mask0nocc.png'))
        occ_mask = np.ascontiguousarray(occ_mask)
        # Resize mask if needed to match image dimensions
        if occ_mask.shape != (H, W):
            occ_mask = np.array(Image.fromarray(occ_mask).resize((W, H), resample=Image.NEAREST))
        
        valid = (np.asarray(valid_gt, dtype=np.float32) >= 0.5) & (occ_mask == 255)

        # Build edges mask and get coordinates
        edges_mask = np.zeros((H, W), dtype=bool)
        edge_coords = None
        if edges_txt_dir is not None:
            txt_path = _find_edges_txt_for_image(edges_txt_dir, imageL_file, dataset_type='eth3d')
            if txt_path is not None:
                edge_coords = _load_edges_from_txt(txt_path)
                if edge_coords is not None:
                    # Create mask for visualization (rounded to nearest pixel)
                    edges_mask = _make_edges_mask_from_coords(edge_coords, (H, W))
        elif edges_mask_dir is not None:
            current_scene = _scene_from_image_path(imageL_file, dataset_type='eth3d')
            mask_img = _load_edges_mask_for_scene(edges_mask_dir, current_scene, (H, W), mask_filename)
            if mask_img is not None:
                edges_mask = mask_img.astype(bool)

        # Compute EPE at edge locations and save valid subpixel edges with GT disparities
        if edge_coords is not None and len(edge_coords) > 0:
            # Get ground-truth disparity (flow_gt is [1, H, W], extract first channel)
            gt_disp_np = flow_gt[0].cpu().numpy() if isinstance(flow_gt, torch.Tensor) else flow_gt[0]
            
            # Use subpixel sampling for coordinate-based edges
            epe_at_edges = _sample_epe_at_subpixel(epe_np, edge_coords)
            gt_disp_at_edges = _sample_disparity_at_subpixel(gt_disp_np, edge_coords)
            
            # Check validity at subpixel locations (using nearest pixel for validity check)
            y_round = np.round(np.clip(edge_coords[:, 0], 0, H - 1)).astype(int)
            x_round = np.round(np.clip(edge_coords[:, 1], 0, W - 1)).astype(int)
            valid_at_edges = valid[y_round, x_round]
            
            # Filter out NaN and inf values from EPE
            finite_mask_epe = np.isfinite(epe_at_edges)
            # Filter out NaN and inf values from GT disparity
            finite_mask_disp = np.isfinite(gt_disp_at_edges)
            # Combined validity: valid mask AND finite EPE AND finite GT disparity
            valid_mask = valid_at_edges & finite_mask_epe & finite_mask_disp
            
            # Get valid edges with their original subpixel coordinates and GT disparities
            valid_edge_coords = edge_coords[valid_mask]
            valid_gt_disparities = gt_disp_at_edges[valid_mask]
            
            # Save valid subpixel edges with GT disparities
            current_scene = _scene_from_image_path(imageL_file, dataset_type='eth3d')
            base_name = Path(imageL_file).stem  # im0
            valid_edges_path = os.path.join(output_dir, "epe_values", f"{current_scene}_{base_name}_valid_edges_gt_disp.txt")
            with open(valid_edges_path, 'w') as f:
                f.write(f"# Valid subpixel edge coordinates with ground-truth disparities for {current_scene} {base_name}\n")
                f.write(f"# Total valid edges: {len(valid_edge_coords)}\n")
                if len(valid_gt_disparities) > 0:
                    f.write(f"# GT Disparity stats - Min: {valid_gt_disparities.min():.6f}, Max: {valid_gt_disparities.max():.6f}, Mean: {valid_gt_disparities.mean():.6f}\n")
                f.write("# Format: y x gt_disparity (one per line, space-separated)\n")
                f.write("# Coordinates are subpixel (float values)\n")
                for coord, gt_disp in zip(valid_edge_coords, valid_gt_disparities):
                    f.write(f"{coord[0]:.6f} {coord[1]:.6f} {gt_disp:.6f}\n")
            
            logging.info(f"Saved {len(valid_edge_coords)} valid subpixel edges with GT disparities")
            
            # Continue with EPE computation
            epe_valid = epe_at_edges[valid_mask]
            if len(epe_valid) > 0:
                out_valid = (epe_valid > 1.0)  # ETH3D uses 1.0 threshold
                image_out = out_valid.mean().item()
                image_epe = epe_valid.mean().item()
            else:
                image_out = 0.0
                image_epe = 0.0
        else:
            # Use mask-based approach for mask-based edges
            val = (valid & edges_mask).reshape(-1)
            epe_flat = epe.reshape(-1)
            epe_valid_flat = epe_flat[val]
            # Convert to numpy and filter out NaN and inf values
            epe_valid_np = epe_valid_flat.numpy() if isinstance(epe_valid_flat, torch.Tensor) else epe_valid_flat
            finite_mask = np.isfinite(epe_valid_np)
            epe_valid_np = epe_valid_np[finite_mask]
            if len(epe_valid_np) > 0:
                out = (epe_valid_np > 1.0)  # ETH3D uses 1.0 threshold
                image_out = out.mean().item()
                image_epe = epe_valid_np.mean().item()
            else:
                image_out = 0.0
                image_epe = 0.0

        edge_count = len(edge_coords) if edge_coords is not None and len(edge_coords) > 0 else int(edges_mask.sum())
        current_scene = _scene_from_image_path(imageL_file, dataset_type='eth3d')
        total_info = f"{processed_count}" if scene_name else f"{val_id+1}/{len(val_dataset)}"
        logging.info(f"[{total_info}] {Path(imageL_file).name} scene={current_scene} edges_count={edge_count} EPE {round(image_epe,4)} D1 {round(image_out,4)}")
        epe_list.append(image_epe)
        out_list.append(image_out)

        # Save EPE values to text files with coordinates
        base_name = Path(imageL_file).stem  # im0
        current_scene = _scene_from_image_path(imageL_file, dataset_type='eth3d')
        
        # Get edge pixels with coordinates (valid pixels only)
        edge_mask_valid = edges_mask & valid
        edge_indices = np.where(edge_mask_valid)
        edge_y_coords = edge_indices[0]  # row coordinates
        edge_x_coords = edge_indices[1]  # column coordinates
        edge_epe_values = epe_np[edge_mask_valid]
        
        # Get non-edge pixels with coordinates (valid pixels only)
        non_edge_mask = ~edges_mask
        non_edge_mask_valid = non_edge_mask & valid
        non_edge_indices = np.where(non_edge_mask_valid)
        non_edge_y_coords = non_edge_indices[0]  # row coordinates
        non_edge_x_coords = non_edge_indices[1]  # column coordinates
        non_edge_epe_values = epe_np[non_edge_mask_valid]
        
        # Save edge EPE values with coordinates
        edge_epe_path = os.path.join(output_dir, "epe_values", f"{current_scene}_{base_name}_edge_epe.txt")
        with open(edge_epe_path, 'w') as f:
            f.write(f"# EPE values at edge locations for {current_scene} {base_name}\n")
            f.write(f"# Total edge pixels (valid): {len(edge_epe_values)}\n")
            f.write(f"# Min: {edge_epe_values.min():.6f}, Max: {edge_epe_values.max():.6f}, Mean: {edge_epe_values.mean():.6f}\n")
            f.write("# Format: y x epe_value (one per line, space-separated)\n")
            for y, x, epe_val in zip(edge_y_coords, edge_x_coords, edge_epe_values):
                f.write(f"{y} {x} {epe_val:.6f}\n")
        
        # Save non-edge EPE values with coordinates
        non_edge_epe_path = os.path.join(output_dir, "epe_values", f"{current_scene}_{base_name}_non_edge_epe.txt")
        with open(non_edge_epe_path, 'w') as f:
            f.write(f"# EPE values at non-edge locations for {current_scene} {base_name}\n")
            f.write(f"# Total non-edge pixels (valid): {len(non_edge_epe_values)}\n")
            f.write(f"# Min: {non_edge_epe_values.min():.6f}, Max: {non_edge_epe_values.max():.6f}, Mean: {non_edge_epe_values.mean():.6f}\n")
            f.write("# Format: y x epe_value (one per line, space-separated)\n")
            for y, x, epe_val in zip(non_edge_y_coords, non_edge_x_coords, non_edge_epe_values):
                f.write(f"{y} {x} {epe_val:.6f}\n")
        
        logging.info(f"Saved EPE values: {len(edge_epe_values)} edge pixels, {len(non_edge_epe_values)} non-edge pixels")
        
        # Visualize EPE probability distributions (same as Middlebury)
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # Left plot: Histogram comparison
            ax1 = axes[0]
            if len(edge_epe_values) > 0 and len(non_edge_epe_values) > 0:
                # Determine appropriate bins
                all_epe = np.concatenate([edge_epe_values, non_edge_epe_values])
                bins = np.linspace(0, min(np.percentile(all_epe, 99), all_epe.max()), 100)
                
                ax1.hist(edge_epe_values, bins=bins, alpha=0.7, label='Edge', color='red', 
                        density=True, edgecolor='black', linewidth=0.5)
                ax1.hist(non_edge_epe_values, bins=bins, alpha=0.7, label='Non-edge', color='blue', 
                        density=True, edgecolor='black', linewidth=0.5)
                ax1.set_xlabel('EPE (pixels)', fontsize=12, fontweight='bold')
                ax1.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
                ax1.set_title(f'EPE Distribution Comparison - {current_scene}\nEdge vs Non-edge', 
                            fontsize=13, fontweight='bold', pad=10)
                ax1.legend(fontsize=11, loc='upper right')
                ax1.tick_params(labelsize=18)  # Make axis numbers larger
                ax1.grid(True, alpha=0.3)
                ax1.set_xlim(0, min(np.percentile(all_epe, 99), all_epe.max()))
            
            # Right plot: Cumulative Distribution Function (CDF)
            ax2 = axes[1]
            if len(edge_epe_values) > 0 and len(non_edge_epe_values) > 0:
                sorted_edge = np.sort(edge_epe_values)
                sorted_non_edge = np.sort(non_edge_epe_values)
                p_edge = np.arange(1, len(sorted_edge) + 1) / len(sorted_edge)
                p_non_edge = np.arange(1, len(sorted_non_edge) + 1) / len(sorted_non_edge)
                
                ax2.plot(sorted_edge, p_edge, label='Edge', color='red', linewidth=2)
                ax2.plot(sorted_non_edge, p_non_edge, label='Non-edge', color='blue', linewidth=2)
                ax2.set_xlabel('EPE (pixels)', fontsize=12, fontweight='bold')
                ax2.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
                ax2.set_title(f'Cumulative Distribution Function - {current_scene}', 
                            fontsize=13, fontweight='bold', pad=10)
                ax2.legend(fontsize=11, loc='lower right')
                ax2.tick_params(labelsize=18)  # Make axis numbers larger
                ax2.grid(True, alpha=0.3)
                ax2.set_xlim(0, min(np.percentile(all_epe, 99), all_epe.max()))
            
            plt.tight_layout()
            dist_path = os.path.join(output_dir, "epe_distributions", f"{current_scene}_{base_name}_epe_distribution.png")
            plt.savefig(dist_path, dpi=200, bbox_inches='tight')
            plt.close()
            
            # Also create a detailed statistics plot
            fig, ax = plt.subplots(figsize=(12, 8))
            if len(edge_epe_values) > 0 and len(non_edge_epe_values) > 0:
                # Create more detailed histogram with statistics
                bins = np.linspace(0, min(np.percentile(all_epe, 99), all_epe.max()), 150)
                
                n_edge, bins_edge, patches_edge = ax.hist(edge_epe_values, bins=bins, alpha=0.6, 
                                                         label='Edge', color='red', density=True, 
                                                         edgecolor='darkred', linewidth=0.3)
                n_non_edge, bins_non_edge, patches_non_edge = ax.hist(non_edge_epe_values, bins=bins, 
                                                                      alpha=0.6, label='Non-edge', 
                                                                      color='blue', density=True, 
                                                                      edgecolor='darkblue', linewidth=0.3)
                
                # Add vertical lines for mean values
                mean_edge = edge_epe_values.mean()
                mean_non_edge = non_edge_epe_values.mean()
                ax.axvline(mean_edge, color='red', linestyle='--', linewidth=2, 
                          label=f'Edge Mean: {mean_edge:.3f}')
                ax.axvline(mean_non_edge, color='blue', linestyle='--', linewidth=2, 
                          label=f'Non-edge Mean: {mean_non_edge:.3f}')
                
                # Add text box with statistics
                stats_text = f'Edge Statistics:\n'
                stats_text += f'  Mean: {mean_edge:.4f}\n'
                stats_text += f'  Median: {np.median(edge_epe_values):.4f}\n'
                stats_text += f'  Std: {edge_epe_values.std():.4f}\n'
                stats_text += f'  Min: {edge_epe_values.min():.4f}\n'
                stats_text += f'  Max: {edge_epe_values.max():.4f}\n'
                stats_text += f'  Count: {len(edge_epe_values)}\n\n'
                stats_text += f'Non-edge Statistics:\n'
                stats_text += f'  Mean: {mean_non_edge:.4f}\n'
                stats_text += f'  Median: {np.median(non_edge_epe_values):.4f}\n'
                stats_text += f'  Std: {non_edge_epe_values.std():.4f}\n'
                stats_text += f'  Min: {non_edge_epe_values.min():.4f}\n'
                stats_text += f'  Max: {non_edge_epe_values.max():.4f}\n'
                stats_text += f'  Count: {len(non_edge_epe_values)}'
                
                ax.text(0.98, 0.5, stats_text, transform=ax.transAxes, 
                       fontsize=10, verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
                ax.set_xlabel('EPE (pixels)', fontsize=13, fontweight='bold')
                ax.set_ylabel('Probability Density', fontsize=13, fontweight='bold')
                ax.set_title(f'EPE Probability Distribution - {current_scene} {base_name}', 
                           fontsize=14, fontweight='bold', pad=15)
                ax.legend(fontsize=11, loc='upper right')
                ax.tick_params(labelsize=18)  # Make axis numbers larger
                ax.grid(True, alpha=0.3)
                ax.set_xlim(0, min(np.percentile(all_epe, 99), all_epe.max()))
            
            plt.tight_layout()
            detailed_dist_path = os.path.join(output_dir, "epe_distributions", 
                                            f"{current_scene}_{base_name}_epe_distribution_detailed.png")
            plt.savefig(detailed_dist_path, dpi=200, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Saved EPE distribution plots: {dist_path}, {detailed_dist_path}")
        except Exception as e:
            logging.warning(f"Could not create EPE distribution plots: {e}")
        
        break

    logging.info(f"Processed {processed_count} images, skipped {skipped_count} images")
    logging.info(f"EPE list length: {len(epe_list)}, Out list length: {len(out_list)}")
    
    if len(epe_list) == 0:
        logging.warning("No images were processed! Check if:")
        logging.warning("  1. Dataset is empty")
        logging.warning("  2. Scene name filter is too restrictive")
        logging.warning("  3. Edge files are missing or not found")
        return {f'eth3d-edges-epe': 0.0, f'eth3d-edges-d1': 0.0}
    
    epe_list = np.array(epe_list)
    out_list = np.array(out_list)
    # Use nanmean to handle any remaining NaN values
    epe = float(np.nanmean(epe_list)) if len(epe_list) > 0 else 0.0
    d1 = 100.0 * float(np.nanmean(out_list)) if len(out_list) > 0 else 0.0
    print(f"Validation ETH3D (edges only): EPE {epe}, D1 {d1}")
    return {f'eth3d-edges-epe': epe, f'eth3d-edges-d1': d1}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default=None)
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument("--precision_dtype", default="float16", choices=["float16", "bfloat16", "float32"], help="precision type")
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
    parser.add_argument('--max_disp', type=int, default=192, help='max disp of geometry encoding volume')
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of correlation pyramid levels")
    parser.add_argument('--corr_radius', type=int, default=4, help="correlation radius")

    # Edge processing inputs
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--edges_txt_dir', type=str, help="Directory containing .txt files with edge coordinates. Format: one 'y x' pair per line (supports subpixel float coordinates). Files can be named <scene>/edges.txt, <scene>.txt, or <image_basename>.txt")
    group.add_argument('--edges_mask_dir', type=str, help="Directory containing per-scene edge masks (e.g., <dir>/<scene>/<mask_filename>)")
    parser.add_argument('--mask_filename', type=str, default='edges.png', help="Edges mask filename within scene folder (used with --edges_mask_dir)")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save outputs")
    parser.add_argument('--scene_name', type=str, default=None, help="Optional: process only a specific scene name")
    parser.add_argument('--dataset', type=str, default='middlebury', choices=['middlebury', 'eth3d'], help="Dataset to evaluate on")

    args = parser.parse_args()

    model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(args.restore_ckpt, map_location='cpu')
        model.load_state_dict(checkpoint, strict=True)
        logging.info("Done loading checkpoint")

    model.cuda()
    model.eval()

    print(f"The model has {format(count_parameters(model)/1e6, '.2f')}M learnable parameters.")
    use_mixed_precision = args.corr_implementation.endswith("_cuda")

    if args.dataset == 'middlebury':
        validate_middlebury_edges(
            model,
            iters=args.valid_iters,
            mixed_prec=use_mixed_precision,
            resolution='F',
            edges_txt_dir=args.edges_txt_dir,
            edges_mask_dir=args.edges_mask_dir,
            mask_filename=args.mask_filename,
            output_dir=args.output_dir,
            scene_name=args.scene_name
        )
    elif args.dataset == 'eth3d':
        validate_eth3d_edges(
            model,
            iters=args.valid_iters,
            mixed_prec=use_mixed_precision,
            edges_txt_dir=args.edges_txt_dir,
            edges_mask_dir=args.edges_mask_dir,
            mask_filename=args.mask_filename,
            output_dir=args.output_dir,
            scene_name=args.scene_name
        )


