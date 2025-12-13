"""
Skull Thickness Measurement using Ray Tracing

This script computes skull thickness using two available methods:

1. SURFACE-NORMAL METHOD (default, more accurate):
   - Measures thickness perpendicular to the skull surface
   - Gives true geometric thickness (shortest distance through bone)
   - Better for accurate aperture design

2. RADIAL METHOD (Attali et al. 2023):
   - Rays emanate from brain cavity center outward
   - May overestimate thickness on curved surfaces
   - Conservative approach for safety estimates

Pipeline steps:
1. Loading NRRD volumetric CT data
2. Creating a skull mask using Otsu thresholding + morphological closing
3. Extracting outer surface mesh with normals
4. Ray tracing to measure thickness (using selected method)
5. Visualizing results in an interactive 3D viewer
"""

import numpy as np
import nrrd
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.filters import threshold_otsu
from skimage.measure import marching_cubes
import pyvista as pv

# Default data path (used if no specimen specified)
DEFAULT_SPECIMEN = "A0001"
DATA_DIR = "/Users/thomasribeiro/code/human-skulls-ct/data"


def find_nrrd_path(specimen_id: str, data_dir: str = DATA_DIR) -> str:
    """
    Find the NRRD file path for a given specimen ID.

    Searches recursively in the data directory for {specimen_id}.nrrd

    Args:
        specimen_id: Specimen identifier (e.g., "A0001")
        data_dir: Base data directory

    Returns:
        Full path to the NRRD file

    Raises:
        FileNotFoundError: If specimen NRRD file not found
    """
    from pathlib import Path

    data_path = Path(data_dir)

    # Search for the NRRD file
    matches = list(data_path.rglob(f"{specimen_id}.nrrd"))

    if not matches:
        raise FileNotFoundError(f"No NRRD file found for specimen {specimen_id} in {data_dir}")

    if len(matches) > 1:
        print(f"Warning: Multiple NRRD files found for {specimen_id}, using first: {matches[0]}")

    return str(matches[0])


# =============================================================================
# 1. Data Loading and Preprocessing
# =============================================================================

def load_nrrd(path: str) -> tuple[np.ndarray, dict]:
    """Load NRRD file and return data with header."""
    data, header = nrrd.read(path)
    return data, header


def get_spacing(header: dict) -> np.ndarray:
    """Extract voxel spacing from NRRD header."""
    space_directions = header['space directions']
    spacing = np.array([
        np.linalg.norm(space_directions[0]),
        np.linalg.norm(space_directions[1]),
        np.linalg.norm(space_directions[2])
    ])
    return spacing


def resample_isotropic(data: np.ndarray, original_spacing: np.ndarray,
                       target_spacing: float = 0.5) -> tuple[np.ndarray, float]:
    """
    Resample volume to isotropic resolution.

    Args:
        data: 3D volume
        original_spacing: Original voxel spacing [x, y, z]
        target_spacing: Target isotropic spacing in mm

    Returns:
        Resampled data and the actual spacing used
    """
    zoom_factors = original_spacing / target_spacing
    resampled = ndimage.zoom(data, zoom_factors, order=1)  # Linear interpolation
    print(f"Resampled from {data.shape} to {resampled.shape}")
    print(f"Original spacing: {original_spacing} mm")
    print(f"Target spacing: {target_spacing} mm isotropic")
    return resampled, target_spacing


# =============================================================================
# 2. Skull Mask Creation
# =============================================================================

def create_skull_mask(data: np.ndarray, spacing: float,
                      closing_radius_mm: float = 1.5,
                      keep_top_fraction: float = 0.65) -> tuple[np.ndarray, float]:
    """
    Create binary skull mask from CT data or use existing binary mask.

    Isolates the skull (calvarium) from other structures like jaw/facial bones by:
    1. Keeping only the largest connected component
    2. Optionally filtering to upper portion of the volume

    If data is already binary (0/1), use it directly.
    Otherwise, apply Otsu thresholding.

    Args:
        data: CT volume (Hounsfield units) or pre-segmented binary mask
        spacing: Voxel spacing in mm
        closing_radius_mm: Radius of closing structuring element
        keep_top_fraction: Keep only this fraction of volume from top (0.65 = top 65%)

    Returns:
        Binary mask and threshold value (or 0.5 if already binary)
    """
    # Check if data is already binary (only 0 and 1 values)
    unique_values = np.unique(data)
    is_binary = len(unique_values) <= 2 and set(unique_values).issubset({0, 1})

    if is_binary:
        print("Data is already a binary mask - using directly")
        mask = data > 0
        threshold = 0.5
    else:
        # Set negative values to zero (minimal thresholding)
        data_positive = np.maximum(data, 0)

        # Otsu thresholding on non-zero voxels
        nonzero_data = data_positive[data_positive > 0]
        if len(nonzero_data) == 0:
            raise ValueError("No non-zero voxels found in data")

        threshold = threshold_otsu(nonzero_data)
        print(f"Otsu threshold: {threshold:.1f} HU")

        # Create binary mask
        mask = data_positive > threshold

    n_bone_voxels = mask.sum()
    print(f"Bone voxels (before filtering): {n_bone_voxels:,}")

    if n_bone_voxels == 0:
        raise ValueError("No bone voxels found")

    # Filter to keep only upper portion (removes jaw and lower facial structures)
    # Z-axis is typically superior-inferior in medical imaging
    if keep_top_fraction < 1.0:
        z_cutoff = int(mask.shape[2] * (1 - keep_top_fraction))
        print(f"Filtering to top {keep_top_fraction*100:.0f}% of volume (z >= {z_cutoff})")
        mask[:, :, :z_cutoff] = False
        n_after_z_filter = mask.sum()
        print(f"Bone voxels after Z-filter: {n_after_z_filter:,}")

    # Keep only the largest connected component (the skull)
    print("Finding largest connected component (skull)...")
    labeled, num_features = ndimage.label(mask)
    if num_features > 1:
        component_sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
        largest_component = np.argmax(component_sizes) + 1
        mask = labeled == largest_component
        print(f"Found {num_features} components, keeping largest ({component_sizes[largest_component-1]:,.0f} voxels)")
    else:
        print(f"Single component found ({mask.sum():,} voxels)")

    n_skull_voxels = mask.sum()
    print(f"Skull voxels: {n_skull_voxels:,}")

    # Create spherical structuring element for morphological closing
    # Using smaller radius (1.5mm) for faster processing
    radius_voxels = int(np.ceil(closing_radius_mm / spacing))
    struct_size = 2 * radius_voxels + 1

    # Generate sphere
    x, y, z = np.ogrid[:struct_size, :struct_size, :struct_size]
    center = radius_voxels
    sphere = ((x - center)**2 + (y - center)**2 + (z - center)**2) <= radius_voxels**2

    print(f"Applying morphological closing with {closing_radius_mm}mm radius ({radius_voxels} voxels)...")
    mask_closed = ndimage.binary_closing(mask, structure=sphere)

    n_closed_voxels = mask_closed.sum()
    print(f"Bone voxels after closing: {n_closed_voxels:,}")

    return mask_closed, threshold


# =============================================================================
# 3. Ray Generation
# =============================================================================

def generate_sphere_points(n_points: int = 100000) -> np.ndarray:
    """
    Generate uniformly distributed points on a full sphere using Fibonacci lattice.

    Rays will radiate in all directions from the brain cavity center.

    Args:
        n_points: Number of points to generate

    Returns:
        Array of shape (n_points, 3) with unit vectors
    """
    # Using golden spiral method for uniform distribution on sphere
    indices = np.arange(0, n_points, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n_points)
    theta = np.pi * (1 + np.sqrt(5)) * indices

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    points = np.column_stack([x, y, z])

    print(f"Generated {len(points)} sphere points (full sphere)")
    return points


def compute_brain_cavity_center(mask: np.ndarray, spacing: float) -> np.ndarray:
    """
    Compute the center of the brain cavity for ray tracing origin.

    Strategy:
    1. Find the bounding box of the skull
    2. Look at the interior (non-bone) voxels within that bounding box
    3. Find the largest connected component of interior space (the brain cavity)
    4. Use its centroid as the ray origin

    Args:
        mask: Binary skull mask (bone only)
        spacing: Voxel spacing in mm

    Returns:
        Center coordinates in mm (inside brain cavity)
    """
    print("Computing brain cavity center...")

    # Get bounding box of skull
    bone_coords = np.array(np.where(mask)).T
    min_coords = bone_coords.min(axis=0)
    max_coords = bone_coords.max(axis=0)

    print(f"Skull bounding box: {min_coords} to {max_coords}")

    # Create a volume representing potential interior space
    # Shrink the bounding box slightly to avoid edge effects
    margin = 10  # voxels
    x_min, y_min, z_min = min_coords + margin
    x_max, y_max, z_max = max_coords - margin

    # Create interior mask: non-bone voxels within the skull bounding box
    interior = np.zeros_like(mask, dtype=bool)
    interior[x_min:x_max, y_min:y_max, z_min:z_max] = True
    interior = interior & ~mask  # Remove bone voxels

    n_interior = interior.sum()
    print(f"Potential interior voxels: {n_interior:,}")

    if n_interior == 0:
        # Fallback to bounding box center
        print("Warning: No interior found, using bounding box center")
        centroid_voxels = (min_coords + max_coords) / 2.0
    else:
        # Find largest connected component of interior (brain cavity)
        labeled_interior, num_components = ndimage.label(interior)
        print(f"Found {num_components} interior components")

        if num_components > 1:
            # Find the largest component
            component_sizes = ndimage.sum(interior, labeled_interior, range(1, num_components + 1))
            largest_idx = np.argmax(component_sizes) + 1
            brain_cavity = labeled_interior == largest_idx
            print(f"Largest component (brain cavity): {component_sizes[largest_idx-1]:,.0f} voxels")
        else:
            brain_cavity = interior

        # Compute centroid of brain cavity
        cavity_coords = np.array(np.where(brain_cavity)).T
        centroid_voxels = cavity_coords.mean(axis=0)

    centroid_mm = centroid_voxels * spacing
    print(f"Brain cavity center (voxels): {centroid_voxels}")
    print(f"Brain cavity center (mm): {centroid_mm}")

    return centroid_mm


# =============================================================================
# 4. Thickness Computation
# =============================================================================

def trace_ray_thickness_radial(mask: np.ndarray, spacing: float, origin: np.ndarray,
                               direction: np.ndarray, max_distance: float = 150.0,
                               step_size: float = 0.1) -> tuple[float | None, np.ndarray | None, np.ndarray | None]:
    """
    Trace a ray from origin outward and compute skull thickness (RADIAL METHOD).

    This is the original Attali et al. 2023 method where rays emanate from
    the brain cavity center. May overestimate thickness on curved surfaces.

    The ray starts from inside the skull (origin = centroid) and goes outward.
    Thickness is measured from FIRST bone entry to LAST bone exit.
    This correctly handles air cavities/diploe within the skull.

    Args:
        mask: Binary skull mask
        spacing: Voxel spacing in mm
        origin: Ray origin in mm (brain cavity center)
        direction: Unit direction vector
        max_distance: Maximum ray distance in mm
        step_size: Ray marching step size in mm

    Returns:
        Thickness in mm, inner surface point, outer surface point, or (None, None, None) if invalid
    """
    direction = direction / np.linalg.norm(direction)

    # Ray march from origin outward
    n_steps = int(max_distance / step_size)

    entry_point = None  # First contact with bone (inner surface)
    exit_point = None   # Last exit from bone (outer surface)
    was_in_bone = False

    for i in range(n_steps):
        t = i * step_size
        point_mm = origin + t * direction
        point_voxel = (point_mm / spacing).astype(int)

        # Check bounds
        if not (0 <= point_voxel[0] < mask.shape[0] and
                0 <= point_voxel[1] < mask.shape[1] and
                0 <= point_voxel[2] < mask.shape[2]):
            break

        in_bone = mask[point_voxel[0], point_voxel[1], point_voxel[2]]

        # Detect FIRST bone entry (set once, never update)
        if in_bone and entry_point is None:
            entry_point = point_mm.copy()

        # Detect bone exit - keep updating to get the LAST exit
        # (handles air cavities within skull)
        if not in_bone and was_in_bone:
            exit_point = point_mm.copy()
            # Don't break - continue to find if there's more bone ahead

        was_in_bone = in_bone

    # Calculate thickness from first entry to last exit
    if entry_point is not None and exit_point is not None:
        thickness = np.linalg.norm(exit_point - entry_point)
        # Clamp to valid range (1-20mm as per paper)
        if 1.0 <= thickness <= 20.0:
            return thickness, entry_point, exit_point

    return None, None, None


def trace_ray_thickness_normal(mask: np.ndarray, spacing: float,
                               surface_point: np.ndarray, normal: np.ndarray,
                               max_distance: float = 30.0,
                               step_size: float = 0.1) -> float | None:
    """
    Trace inward along surface normal to measure thickness (SURFACE-NORMAL METHOD).

    This method measures the true geometric thickness perpendicular to the
    skull surface, giving the shortest distance through the bone.

    Starting from the outer surface, traces inward along the negative normal
    direction until exiting the bone on the inner surface.

    Args:
        mask: Binary skull mask
        spacing: Voxel spacing in mm
        surface_point: Point on outer surface in mm
        normal: Outward-pointing unit normal at surface point
        max_distance: Maximum ray distance in mm (skull rarely >20mm thick)
        step_size: Ray marching step size in mm

    Returns:
        Thickness in mm, or None if invalid measurement
    """
    # Negate normal to trace inward (normals point outward from filled volume)
    inward = -normal / np.linalg.norm(normal)

    entry_point = None
    exit_point = None
    in_bone = False

    n_steps = int(max_distance / step_size)

    for i in range(n_steps):
        t = i * step_size
        point_mm = surface_point + t * inward
        point_voxel = (point_mm / spacing).astype(int)

        # Check bounds
        if not (0 <= point_voxel[0] < mask.shape[0] and
                0 <= point_voxel[1] < mask.shape[1] and
                0 <= point_voxel[2] < mask.shape[2]):
            break

        currently_in_bone = mask[point_voxel[0], point_voxel[1], point_voxel[2]]

        # Detect first entry into bone
        if currently_in_bone and entry_point is None:
            entry_point = point_mm.copy()

        # Detect exit from bone (transition from bone to non-bone)
        if not currently_in_bone and in_bone:
            exit_point = point_mm.copy()
            break  # Found inner surface, stop

        in_bone = currently_in_bone

    # Calculate thickness
    if entry_point is not None and exit_point is not None:
        thickness = np.linalg.norm(exit_point - entry_point)
        # Clamp to valid range (1-20mm as per Attali et al.)
        if 1.0 <= thickness <= 20.0:
            return thickness

    return None


def compute_thickness_along_normals(mask: np.ndarray, spacing: float,
                                    vertices: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """
    Compute thickness at each surface vertex using surface-normal method.

    For each vertex on the outer skull surface, traces inward along the
    surface normal to measure the perpendicular thickness through the bone.

    Args:
        mask: Binary skull mask
        spacing: Voxel spacing in mm
        vertices: Surface mesh vertices (N, 3) in mm
        normals: Outward-pointing unit normals (N, 3)

    Returns:
        thicknesses: Array of thickness values per vertex (N,)
                     Invalid measurements are set to NaN
    """
    n_verts = len(vertices)
    thicknesses = np.full(n_verts, np.nan)

    print(f"Computing thickness along surface normals for {n_verts} vertices...")

    valid_count = 0
    for i in range(n_verts):
        if i % 10000 == 0:
            print(f"  Progress: {i}/{n_verts} ({100*i/n_verts:.1f}%)")

        thickness = trace_ray_thickness_normal(
            mask, spacing, vertices[i], normals[i]
        )

        if thickness is not None:
            thicknesses[i] = thickness
            valid_count += 1

    valid_mask = ~np.isnan(thicknesses)
    valid_thicknesses = thicknesses[valid_mask]

    print(f"Valid measurements: {valid_count}/{n_verts} ({100*valid_count/n_verts:.1f}%)")
    if valid_count > 0:
        print(f"Thickness range: {valid_thicknesses.min():.2f} - {valid_thicknesses.max():.2f} mm")
        print(f"Mean thickness: {valid_thicknesses.mean():.2f} mm")

    # Fill NaN values with nearest valid neighbor for visualization
    if valid_count < n_verts and valid_count > 0:
        print("Filling invalid measurements with nearest neighbor interpolation...")
        from scipy.spatial import cKDTree
        valid_indices = np.where(valid_mask)[0]
        invalid_indices = np.where(~valid_mask)[0]

        tree = cKDTree(vertices[valid_indices])
        _, nearest_idx = tree.query(vertices[invalid_indices])
        thicknesses[invalid_indices] = thicknesses[valid_indices[nearest_idx]]

    return thicknesses


def compute_all_thicknesses_radial(mask: np.ndarray, spacing: float, center: np.ndarray,
                                   directions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute thickness for all ray directions (RADIAL METHOD).

    This is the original Attali et al. 2023 method where rays emanate from
    the brain cavity center outward.

    Args:
        mask: Binary skull mask
        spacing: Voxel spacing in mm
        center: Ray origin (brain cavity centroid) in mm
        directions: Array of unit direction vectors (N, 3)

    Returns:
        thicknesses: Array of thickness values
        inner_points: Array of inner surface hit points (entry into bone)
        outer_points: Array of outer surface hit points (exit from bone)
        valid_mask: Boolean mask indicating valid measurements
    """
    n_rays = len(directions)
    thicknesses = np.zeros(n_rays)
    inner_points = np.zeros((n_rays, 3))
    outer_points = np.zeros((n_rays, 3))
    valid_mask = np.zeros(n_rays, dtype=bool)

    print(f"Computing thickness for {n_rays} rays...")

    for i, direction in enumerate(directions):
        if i % 10000 == 0:
            print(f"  Progress: {i}/{n_rays} ({100*i/n_rays:.1f}%)")

        thickness, inner_pt, outer_pt = trace_ray_thickness_radial(mask, spacing, center, direction)

        if thickness is not None:
            thicknesses[i] = thickness
            inner_points[i] = inner_pt
            outer_points[i] = outer_pt
            valid_mask[i] = True

    n_valid = valid_mask.sum()
    print(f"Valid measurements: {n_valid}/{n_rays} ({100*n_valid/n_rays:.1f}%)")
    print(f"Thickness range: {thicknesses[valid_mask].min():.2f} - {thicknesses[valid_mask].max():.2f} mm")
    print(f"Mean thickness: {thicknesses[valid_mask].mean():.2f} mm")

    return thicknesses, inner_points, outer_points, valid_mask


# =============================================================================
# 5. Surface Extraction
# =============================================================================

def extract_outer_surface(mask: np.ndarray, spacing: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract OUTER skull surface mesh using marching cubes.

    To get only the outer surface (not inner), we first fill the skull interior
    so marching cubes only extracts the external boundary.

    Args:
        mask: Binary skull mask
        spacing: Voxel spacing in mm

    Returns:
        vertices: (N, 3) array of vertex positions in mm
        faces: (M, 3) array of triangle indices
        normals: (N, 3) array of outward-pointing surface normals (unit vectors)
    """
    print("Filling skull interior to extract outer surface only...")

    n_original = mask.sum()

    # Strategy: Use morphological closing with large structuring element
    # to close all gaps, then fill holes
    # This ensures we get a solid volume for outer surface extraction

    # First, close small gaps with moderate iterations
    struct = ndimage.generate_binary_structure(3, 1)
    closed = ndimage.binary_closing(mask, structure=struct, iterations=15)

    # Then fill any remaining interior holes
    filled = ndimage.binary_fill_holes(closed)

    # If still not filled enough, use dilation then erosion
    n_filled = filled.sum()
    if n_filled < n_original * 1.5:  # Should have significant interior
        print("  Additional morphological filling...")
        # Dilate to close gaps, fill, then erode back
        dilated = ndimage.binary_dilation(mask, structure=struct, iterations=5)
        filled = ndimage.binary_fill_holes(dilated)
        # Erode back but keep interior filled
        # Don't erode - keep the filled volume

    print(f"  Original mask: {n_original:,} voxels")
    print(f"  Filled mask: {filled.sum():,} voxels")

    print("Extracting surface mesh using marching cubes...")
    verts, faces, normals, values = marching_cubes(
        filled.astype(float),
        level=0.5,
        spacing=(spacing, spacing, spacing)
    )
    print(f"Surface mesh: {len(verts)} vertices, {len(faces)} faces")

    # Normalize the normals (marching_cubes returns them but they may not be unit length)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    normals = normals / norms

    return verts, faces, normals


def map_thickness_to_surface(surface_verts: np.ndarray, hit_points: np.ndarray,
                              thicknesses: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """
    Map thickness values from ray hit points to surface mesh vertices.

    Uses nearest-neighbor interpolation with KD-tree.

    Args:
        surface_verts: Surface mesh vertices (N, 3)
        hit_points: Ray hit points (M, 3)
        thicknesses: Thickness values (M,)
        valid_mask: Boolean mask for valid measurements

    Returns:
        Thickness values for each surface vertex
    """
    print("Mapping thickness to surface vertices...")

    # Build KD-tree from valid hit points
    valid_hit_points = hit_points[valid_mask]
    valid_thicknesses = thicknesses[valid_mask]

    tree = cKDTree(valid_hit_points)

    # Find nearest hit point for each surface vertex
    distances, indices = tree.query(surface_verts, k=1)

    # Map thickness values
    vertex_thicknesses = valid_thicknesses[indices]

    print(f"Mapped thickness to {len(surface_verts)} vertices")
    return vertex_thicknesses


# =============================================================================
# 6. Visualization
# =============================================================================

def visualize_interactive(verts: np.ndarray, faces: np.ndarray,
                          thicknesses: np.ndarray,
                          center: np.ndarray = None,
                          outer_points: np.ndarray = None,
                          valid_mask: np.ndarray = None,
                          n_sample_rays: int = 100,
                          cmap: str = 'jet', clim: tuple = (3, 10),
                          regions: dict = None,
                          thin_indices: dict = None,
                          apertures: dict = None):
    """
    Launch interactive 3D visualization with thickness colormap.

    Args:
        verts: Surface mesh vertices (N, 3)
        faces: Surface mesh faces (M, 3)
        thicknesses: Thickness values per vertex (N,)
        center: Ray origin (centroid) for visualization
        outer_points: Outer surface hit points from ray tracing
        valid_mask: Boolean mask for valid measurements
        n_sample_rays: Number of sample rays to display
        cmap: Colormap name
        clim: Color limits (min, max) in mm
        regions: Dictionary mapping region name to vertex indices (optional)
        thin_indices: Dictionary mapping region name to thin vertex indices (optional)
        apertures: Dictionary mapping region name to aperture results (optional)
    """
    print("Launching interactive 3D viewer...")

    # Create PyVista mesh
    # PyVista expects faces in format [n_points, idx1, idx2, idx3, ...]
    faces_pv = np.column_stack([
        np.full(len(faces), 3),
        faces
    ]).flatten()

    mesh = pv.PolyData(verts, faces_pv)
    mesh['Thickness (mm)'] = thicknesses

    # Create plotter
    plotter = pv.Plotter()

    # Add skull mesh with thickness colormap - fully opaque outer surface
    plotter.add_mesh(
        mesh,
        scalars='Thickness (mm)',
        cmap=cmap,
        clim=clim,
        show_scalar_bar=True,
        opacity=1.0,  # Fully opaque - only outer surface should be visible
        scalar_bar_args={
            'title': 'Thickness (mm)',
            'vertical': True,
            'position_x': 0.85,
            'position_y': 0.1,
            'width': 0.1,
            'height': 0.8
        }
    )

    # Add centroid as a sphere
    if center is not None:
        centroid_sphere = pv.Sphere(radius=3.0, center=center)
        plotter.add_mesh(centroid_sphere, color='yellow', label='Centroid')
        print(f"Added centroid at {center}")

    # Add sample hit points on surface
    if outer_points is not None and valid_mask is not None:
        valid_outer = outer_points[valid_mask]
        # Sample subset of points
        n_points = min(500, len(valid_outer))
        sample_idx = np.random.choice(len(valid_outer), n_points, replace=False)
        sample_points = valid_outer[sample_idx]

        point_cloud = pv.PolyData(sample_points)
        plotter.add_mesh(point_cloud, color='green', point_size=5,
                        render_points_as_spheres=True, label='Hit points')
        print(f"Added {n_points} sample hit points")

    # Add sample rays from centroid to surface
    if center is not None and outer_points is not None and valid_mask is not None:
        valid_outer = outer_points[valid_mask]
        n_rays_to_show = min(n_sample_rays, len(valid_outer))
        sample_idx = np.random.choice(len(valid_outer), n_rays_to_show, replace=False)

        # Create lines from center to hit points
        lines = []
        for idx in sample_idx:
            line = pv.Line(center, valid_outer[idx])
            lines.append(line)

        if lines:
            rays_mesh = lines[0]
            for line in lines[1:]:
                rays_mesh = rays_mesh.merge(line)
            plotter.add_mesh(rays_mesh, color='red', line_width=1, label='Sample rays')
            print(f"Added {n_rays_to_show} sample rays")

    # Add thin region highlighting for apertures
    if thin_indices is not None:
        region_colors = {
            'temporal_left': 'cyan',
            'temporal_right': 'magenta',
            'occipital': 'orange'
        }
        for region_name, indices in thin_indices.items():
            if len(indices) > 0:
                thin_points = pv.PolyData(verts[indices])
                color = region_colors.get(region_name, 'white')
                plotter.add_mesh(thin_points, color=color, point_size=8,
                                render_points_as_spheres=True,
                                label=f'{region_name} aperture')
                print(f"Added {len(indices)} thin vertices for {region_name}")

    # Add aperture centroid markers
    if apertures is not None:
        for region_name, aperture_data in apertures.items():
            centroid = aperture_data.get('centroid_mm')
            if centroid is not None and aperture_data.get('width_mm', 0) > 0:
                centroid_sphere = pv.Sphere(radius=2.0, center=centroid)
                plotter.add_mesh(centroid_sphere, color='white', label=f'{region_name} center')

    plotter.add_title("Skull Thickness Map (Attali et al. 2023 Methodology)", font_size=12)
    plotter.add_axes()
    plotter.add_legend()

    # Set initial camera position (lateral view)
    plotter.camera_position = 'xz'

    plotter.show()


def visualize_with_toggles(verts: np.ndarray, faces: np.ndarray,
                           thicknesses: np.ndarray,
                           center: np.ndarray,
                           regions: dict,
                           apertures: dict,
                           region_stats: dict,
                           cmap: str = 'jet', clim: tuple = (3, 10)):
    """
    Launch interactive 3D visualization with toggle controls for layers.

    Features:
    - Checkbox to toggle thickness map visibility
    - Checkbox to toggle region boundaries
    - Checkbox to toggle aperture rectangles
    - 3D rectangles showing transducer aperture footprints

    Args:
        verts: Surface mesh vertices (N, 3)
        faces: Surface mesh faces (M, 3)
        thicknesses: Thickness values per vertex (N,)
        center: Brain cavity center
        regions: Dictionary mapping region name to vertex indices
        apertures: Dictionary mapping aperture name to aperture data
        region_stats: Dictionary with region statistics
        cmap: Colormap name
        clim: Color limits (min, max) in mm
    """
    print("Launching interactive 3D viewer with toggle controls...")

    # Create PyVista mesh
    faces_pv = np.column_stack([
        np.full(len(faces), 3),
        faces
    ]).flatten()

    mesh = pv.PolyData(verts, faces_pv)
    mesh['Thickness (mm)'] = thicknesses

    # Create plotter
    plotter = pv.Plotter()

    # Store actors for toggling
    actors = {}

    # 1. Add skull mesh with thickness colormap
    actors['thickness'] = plotter.add_mesh(
        mesh,
        scalars='Thickness (mm)',
        cmap=cmap,
        clim=clim,
        show_scalar_bar=True,
        opacity=1.0,
        scalar_bar_args={
            'title': 'Thickness (mm)',
            'vertical': True,
            'position_x': 0.85,
            'position_y': 0.1,
            'width': 0.1,
            'height': 0.8
        }
    )

    # 2. Add region boundary visualization (as colored point clouds)
    region_colors = {
        'temporal_left': 'cyan',
        'temporal_right': 'magenta',
        'occipital_left': 'orange',
        'occipital_right': 'yellow',
        'occipital': 'orange'  # Legacy support
    }
    region_actors = []
    for region_name, indices in regions.items():
        if len(indices) > 0:
            # Sample boundary points (edge of region)
            region_verts = verts[indices]
            # Just show a subset as points to indicate region
            sample_size = min(1000, len(indices))
            sample_idx = np.random.choice(len(indices), sample_size, replace=False)
            boundary_points = pv.PolyData(region_verts[sample_idx])
            color = region_colors.get(region_name, 'white')
            actor = plotter.add_mesh(
                boundary_points,
                color=color,
                point_size=3,
                render_points_as_spheres=True,
                opacity=0.5
            )
            region_actors.append(actor)
    actors['regions'] = region_actors

    # 3. Add aperture rectangles (3D visualization of transducer footprints)
    aperture_actors = []
    aperture_colors = {
        'temporal_left': 'cyan',
        'temporal_right': 'magenta',
        'occipital_left': 'orange',
        'occipital_right': 'yellow',
        'occipital': 'orange',
        'occipital_0': 'orange',
        'occipital_1': 'yellow'
    }

    for aperture_name, ap_data in apertures.items():
        corners = ap_data.get('rect_corners_3d')
        if corners is not None and len(corners) == 4:
            corners = np.array(corners)
            # Create rectangle mesh from 4 corners
            rect_faces = np.array([[4, 0, 1, 2, 3]])  # Single quad face
            rect_mesh = pv.PolyData(corners, rect_faces)

            # Determine color based on aperture name
            base_region = aperture_name.split('_')[0] if '_' in aperture_name else aperture_name
            color = aperture_colors.get(aperture_name, aperture_colors.get(base_region, 'white'))

            actor = plotter.add_mesh(
                rect_mesh,
                color=color,
                opacity=0.7,
                show_edges=True,
                edge_color='black',
                line_width=3
            )
            aperture_actors.append(actor)

            # Add label with dimensions
            centroid = np.array(ap_data['centroid_3d'])
            label = f"{ap_data['width_mm']:.1f}x{ap_data['height_mm']:.1f}mm"
            plotter.add_point_labels(
                [centroid],
                [label],
                font_size=12,
                point_color='white',
                point_size=1,
                render_points_as_spheres=True,
                always_visible=True,
                fill_shape=True,
                shape_color='black',
                shape_opacity=0.7
            )

    actors['apertures'] = aperture_actors

    # Add centroid marker
    centroid_sphere = pv.Sphere(radius=3.0, center=center)
    plotter.add_mesh(centroid_sphere, color='yellow', label='Brain Center')

    # Toggle callback functions
    def toggle_thickness(state):
        actors['thickness'].SetVisibility(state)

    def toggle_regions(state):
        for actor in actors['regions']:
            actor.SetVisibility(state)

    def toggle_apertures(state):
        for actor in actors['apertures']:
            actor.SetVisibility(state)

    # Add checkbox widgets
    # Position: (x, y) from bottom-left, normalized coordinates
    plotter.add_checkbox_button_widget(
        toggle_thickness,
        value=True,
        position=(5, 12),
        size=30,
        border_size=3,
        color_on='blue',
        color_off='grey'
    )
    plotter.add_text("Thickness Map", position=(45, 10), font_size=10)

    plotter.add_checkbox_button_widget(
        toggle_regions,
        value=False,  # Start with regions hidden
        position=(5, 52),
        size=30,
        border_size=3,
        color_on='green',
        color_off='grey'
    )
    plotter.add_text("Regions", position=(45, 50), font_size=10)

    plotter.add_checkbox_button_widget(
        toggle_apertures,
        value=True,
        position=(5, 92),
        size=30,
        border_size=3,
        color_on='red',
        color_off='grey'
    )
    plotter.add_text("Apertures", position=(45, 90), font_size=10)

    # Add title and axes
    plotter.add_title("Skull Thickness & Transducer Apertures", font_size=12)
    plotter.add_axes()

    # Set initial camera position (lateral view)
    plotter.camera_position = 'xz'

    plotter.show()


# =============================================================================
# Main Pipeline
# =============================================================================

def run_thickness_pipeline(nrrd_path: str, method: str = 'normal', n_rays: int = 20000) -> dict:
    """
    Run thickness analysis for a single specimen.

    Args:
        nrrd_path: Path to NRRD file
        method: Thickness measurement method
            - 'normal': Surface-normal method (default) - measures perpendicular
              to surface for accurate geometric thickness
            - 'radial': Original Attali et al. 2023 method - rays from brain
              center, may overestimate thickness on curved surfaces
        n_rays: Number of rays for radial method (ignored for normal method)

    Returns:
        Dictionary with: verts, faces, thicknesses, center, outer_points, valid_mask
    """
    print(f"Using thickness method: {method.upper()}")
    print()

    # Step 1: Load and preprocess data
    print("Step 1: Loading NRRD data...")
    data, header = load_nrrd(nrrd_path)
    spacing_original = get_spacing(header)
    print(f"Data shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print()

    # Step 2: Resample to isotropic
    print("Step 2: Resampling to isotropic resolution...")
    data_iso, spacing = resample_isotropic(data, spacing_original, target_spacing=0.5)
    print()

    # Step 3: Create skull mask
    print("Step 3: Creating skull mask...")
    mask, otsu_threshold = create_skull_mask(data_iso, spacing)
    print()

    # Step 4: Compute brain cavity center (needed for visualization and radial method)
    print("Step 4: Computing brain cavity center...")
    center = compute_brain_cavity_center(mask, spacing)
    print()

    # Step 5: Extract outer surface with normals
    print("Step 5: Extracting outer skull surface...")
    verts, faces, normals = extract_outer_surface(mask, spacing)
    print()

    # Step 6: Compute thickness using selected method
    if method == 'normal':
        # Surface-normal method: perpendicular to surface (more accurate)
        print("Step 6: Computing skull thickness (surface-normal method)...")
        vertex_thicknesses = compute_thickness_along_normals(mask, spacing, verts, normals)
        outer_points = verts  # For normal method, outer points are the vertices
        valid_mask = ~np.isnan(vertex_thicknesses)
    else:
        # Radial method: original Attali et al. approach
        print("Step 6: Computing skull thickness (radial method)...")
        directions = generate_sphere_points(n_points=n_rays)
        thicknesses, _inner_points, outer_points, valid_mask = compute_all_thicknesses_radial(
            mask, spacing, center, directions
        )
        print()

        # Step 7: Map thickness to surface (only needed for radial method)
        print("Step 7: Mapping thickness to surface...")
        vertex_thicknesses = map_thickness_to_surface(
            verts, outer_points, thicknesses, valid_mask
        )

    print()

    return {
        'verts': verts,
        'faces': faces,
        'thicknesses': vertex_thicknesses,
        'center': center,
        'outer_points': outer_points,
        'valid_mask': valid_mask
    }


def main():
    """Run the complete skull thickness analysis pipeline with aperture analysis."""
    import argparse
    from src.aperture_analysis import (
        segment_regions, analyze_all_regions,
        cache_exists, load_thickness_cache, save_thickness_cache,
        save_aperture_results
    )

    parser = argparse.ArgumentParser(
        description="Skull thickness measurement and aperture analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py A0001          # Process specimen A0001
    python main.py A0005 --force  # Recompute A0005 even if cached
    python main.py                # Process default specimen (A0001)
        """
    )
    parser.add_argument(
        'specimen_id', type=str, nargs='?', default=DEFAULT_SPECIMEN,
        help=f'Specimen ID (e.g., A0001). Default: {DEFAULT_SPECIMEN}'
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Force recompute even if cached data exists'
    )
    parser.add_argument(
        '--no-viz', action='store_true',
        help='Skip visualization (useful for batch processing)'
    )
    parser.add_argument(
        '--data-dir', type=str, default=DATA_DIR,
        help=f'Data directory containing specimen folders. Default: {DATA_DIR}'
    )

    args = parser.parse_args()
    specimen_id = args.specimen_id

    print("=" * 60)
    print("Skull Thickness Measurement + Aperture Analysis")
    print("Methodology: Attali et al. 2023")
    print("=" * 60)
    print()

    print(f"Specimen: {specimen_id}")
    print()

    # Find NRRD file
    try:
        nrrd_path = find_nrrd_path(specimen_id, args.data_dir)
        print(f"NRRD file: {nrrd_path}")
        print()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return

    # Check for cached data
    if not args.force and cache_exists(specimen_id):
        print("Found cached thickness data, loading...")
        cached = load_thickness_cache(specimen_id)
        verts = cached['vertices']
        faces = cached['faces']
        vertex_thicknesses = cached['thicknesses']
        center = cached['center']
        outer_points = None
        valid_mask = None
        print()
    else:
        if args.force:
            print("Force recompute requested...")
        else:
            print("No cache found, computing thickness map...")
        result = run_thickness_pipeline(nrrd_path)
        verts = result['verts']
        faces = result['faces']
        vertex_thicknesses = result['thicknesses']
        center = result['center']
        outer_points = result['outer_points']
        valid_mask = result['valid_mask']

        # Save to cache
        print("Saving thickness data to cache...")
        save_thickness_cache(specimen_id, verts, faces, vertex_thicknesses, center)
        print()

    # Step 8: Regional aperture analysis
    print("Step 8: Analyzing regional apertures...")
    percentile = 25  # Thinnest 25% of each region

    print("Segmenting surface into anatomical regions...")
    regions = segment_regions(verts, center)
    print()

    print(f"Computing optimal apertures (thinnest {percentile}% per region, DBSCAN clustering)...")
    analysis_result = analyze_all_regions(verts, vertex_thicknesses, regions, percentile)
    apertures = analysis_result['apertures']
    region_stats = analysis_result['region_stats']

    print(f"\nFound {len(apertures)} contiguous apertures:")
    for aperture_name, ap in apertures.items():
        print(f"  {aperture_name}: {ap['width_mm']:.1f} x {ap['height_mm']:.1f} mm "
              f"(area: {ap['area_mm2']:.0f} mm², thickness: {ap['mean_thickness_mm']:.1f} mm, "
              f"vertices: {ap['n_vertices']})")
    print()

    # Save aperture results
    save_aperture_results(specimen_id, apertures, percentile)
    print()

    # Step 9: Visualize with aperture overlays and toggle controls
    if args.no_viz:
        print("Skipping visualization (--no-viz flag set)")
    else:
        print("Step 9: Launching visualization with aperture overlays...")
        visualize_with_toggles(
            verts, faces, vertex_thicknesses,
            center=center,
            regions=regions,
            apertures=apertures,
            region_stats=region_stats
        )

    print()
    print("Done!")


if __name__ == "__main__":
    main()
