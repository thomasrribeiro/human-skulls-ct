"""
Regional Aperture Analysis for Transcranial Ultrasound Transducers

This module provides functions to:
1. Segment skull surface into anatomical regions (temporal, occipital)
2. Compute optimal rectangular aperture for each region
3. Cache thickness data and save results
4. Support batch processing across multiple specimens

Coordinate system (LAS):
- X = Left (+), Right (-)
- Y = Anterior (-), Posterior (+)
- Z = Superior (+), Inferior (-)
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN


# =============================================================================
# Regional Segmentation
# =============================================================================

def compute_spherical_coords(vertices: np.ndarray, center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert vertex positions to spherical coordinates relative to center.

    Coordinate convention (LAS orientation):
    - azimuth: angle in XY plane, 0° = posterior (+Y), +90° = left (+X), -90° = right (-X)
    - elevation: angle from XY plane, +90° = superior, -90° = inferior

    Args:
        vertices: (N, 3) array of vertex positions in mm
        center: (3,) brain cavity center in mm

    Returns:
        azimuth: (N,) angles in degrees
        elevation: (N,) angles in degrees
    """
    # Vector from center to each vertex
    vectors = vertices - center

    # Compute distance from center
    r = np.linalg.norm(vectors, axis=1)
    r = np.maximum(r, 1e-10)  # Avoid division by zero

    # Azimuth: atan2(x, y) gives 0° at +Y (posterior), +90° at +X (left)
    azimuth = np.degrees(np.arctan2(vectors[:, 0], vectors[:, 1]))

    # Elevation: asin(z / r) gives +90° at superior, -90° at inferior
    elevation = np.degrees(np.arcsin(np.clip(vectors[:, 2] / r, -1, 1)))

    return azimuth, elevation


def filter_outer_surface_vertices(vertices: np.ndarray, center: np.ndarray,
                                    distance_percentile: float = 70) -> np.ndarray:
    """
    Filter vertices to keep only those on the outer skull surface.

    For each angular sector, keeps only vertices beyond a distance threshold
    (vertices farther from center are outer surface).

    Args:
        vertices: (N, 3) array of vertex positions in mm
        center: (3,) brain cavity center in mm
        distance_percentile: Percentile of distance to use as threshold per sector

    Returns:
        Boolean mask indicating outer surface vertices
    """
    # Compute distance from center for each vertex
    distances = np.linalg.norm(vertices - center, axis=1)

    # Compute spherical coordinates for angular binning
    azimuth, elevation = compute_spherical_coords(vertices, center)

    # Create angular bins (10-degree bins in azimuth and elevation)
    az_bins = np.floor(azimuth / 10).astype(int)
    el_bins = np.floor(elevation / 10).astype(int)

    # Combine into a single bin index
    bin_ids = az_bins * 100 + el_bins
    unique_bins = np.unique(bin_ids)

    # For each bin, keep only vertices beyond the distance threshold
    outer_mask = np.zeros(len(vertices), dtype=bool)

    for bin_id in unique_bins:
        bin_mask = bin_ids == bin_id
        if bin_mask.sum() < 5:
            # Too few vertices in bin, keep all
            outer_mask[bin_mask] = True
            continue

        bin_distances = distances[bin_mask]
        threshold = np.percentile(bin_distances, distance_percentile)

        # Mark vertices beyond threshold as outer surface
        bin_indices = np.where(bin_mask)[0]
        outer_in_bin = bin_distances >= threshold
        outer_mask[bin_indices[outer_in_bin]] = True

    return outer_mask


def segment_regions(vertices: np.ndarray, center: np.ndarray,
                    filter_outer: bool = True) -> dict[str, np.ndarray]:
    """
    Segment skull surface vertices into anatomical regions.

    Region definitions (based on angular sectors from brain center):
    - temporal_left: left lateral (azimuth 45° to 135°, elevation -30° to 60°)
    - temporal_right: right lateral (azimuth -135° to -45°, elevation -30° to 60°)
    - occipital_left: posterior left (azimuth -45° to 45°, elevation -45° to 30°, X > 0)
    - occipital_right: posterior right (azimuth -45° to 45°, elevation -45° to 30°, X <= 0)

    The occipital region is explicitly split by the midline (X=0 plane) to ensure
    left and right apertures are always separate.

    Args:
        vertices: (N, 3) array of vertex positions in mm
        center: (3,) brain cavity center in mm
        filter_outer: If True, only include outer surface vertices

    Returns:
        Dictionary mapping region name to array of vertex indices
    """
    azimuth, elevation = compute_spherical_coords(vertices, center)

    # Filter to outer surface only
    if filter_outer:
        outer_mask = filter_outer_surface_vertices(vertices, center)
        print(f"  Filtering to outer surface: {outer_mask.sum()}/{len(vertices)} vertices")
    else:
        outer_mask = np.ones(len(vertices), dtype=bool)

    regions = {}

    # Temporal left: lateral left side
    temporal_left_mask = (
        (azimuth >= 45) & (azimuth <= 135) &
        (elevation >= -30) & (elevation <= 60) &
        outer_mask
    )
    regions['temporal_left'] = np.where(temporal_left_mask)[0]

    # Temporal right: lateral right side
    temporal_right_mask = (
        (azimuth >= -135) & (azimuth <= -45) &
        (elevation >= -30) & (elevation <= 60) &
        outer_mask
    )
    regions['temporal_right'] = np.where(temporal_right_mask)[0]

    # Occipital: posterior region - split by midline (X=center[0])
    occipital_base_mask = (
        (azimuth >= -45) & (azimuth <= 45) &
        (elevation >= -45) & (elevation <= 30) &
        outer_mask
    )

    # Split by X coordinate relative to brain center (midline)
    # X > center[0] is left side in LAS coordinate system
    midline_x = center[0]
    occipital_left_mask = occipital_base_mask & (vertices[:, 0] > midline_x)
    occipital_right_mask = occipital_base_mask & (vertices[:, 0] <= midline_x)

    regions['occipital_left'] = np.where(occipital_left_mask)[0]
    regions['occipital_right'] = np.where(occipital_right_mask)[0]

    # Print summary
    for region_name, indices in regions.items():
        print(f"  {region_name}: {len(indices)} vertices")

    return regions


# =============================================================================
# Aperture Computation
# =============================================================================

def find_thin_vertices(thicknesses: np.ndarray, percentile: float = 25) -> np.ndarray:
    """
    Find vertices in the thinnest percentile.

    Args:
        thicknesses: (N,) thickness values for region vertices
        percentile: percentile threshold (e.g., 25 = thinnest 25%)

    Returns:
        Boolean mask for thin vertices
    """
    threshold = np.percentile(thicknesses, percentile)
    return thicknesses <= threshold, threshold


def minimum_bounding_rectangle(points_2d: np.ndarray) -> tuple[float, float, np.ndarray]:
    """
    Compute minimum area bounding rectangle for 2D points using rotating calipers.

    Args:
        points_2d: (N, 2) array of 2D points

    Returns:
        width: rectangle width (larger dimension)
        height: rectangle height (smaller dimension)
        center: rectangle center in 2D
    """
    if len(points_2d) < 3:
        # Not enough points for convex hull
        if len(points_2d) == 0:
            return 0.0, 0.0, np.array([0.0, 0.0])
        elif len(points_2d) == 1:
            return 0.0, 0.0, points_2d[0]
        else:
            # Two points - return line length
            diff = points_2d[1] - points_2d[0]
            return np.linalg.norm(diff), 0.0, np.mean(points_2d, axis=0)

    try:
        hull = ConvexHull(points_2d)
        hull_points = points_2d[hull.vertices]
    except Exception:
        # Degenerate case - points are collinear
        center = np.mean(points_2d, axis=0)
        dists = np.linalg.norm(points_2d - center, axis=1)
        return 2 * dists.max(), 0.0, center

    # Rotating calipers to find minimum area rectangle
    n = len(hull_points)
    min_area = float('inf')
    best_rect = None

    for i in range(n):
        # Edge vector
        edge = hull_points[(i + 1) % n] - hull_points[i]
        edge_len = np.linalg.norm(edge)
        if edge_len < 1e-10:
            continue

        # Unit vectors along and perpendicular to edge
        u = edge / edge_len
        v = np.array([-u[1], u[0]])

        # Project all hull points onto these axes
        projections = hull_points @ np.column_stack([u, v])

        # Bounding box in rotated frame
        min_proj = projections.min(axis=0)
        max_proj = projections.max(axis=0)

        width = max_proj[0] - min_proj[0]
        height = max_proj[1] - min_proj[1]
        area = width * height

        if area < min_area:
            min_area = area
            # Center in rotated frame
            center_rotated = (min_proj + max_proj) / 2
            # Transform back to original frame
            center_original = center_rotated[0] * u + center_rotated[1] * v
            best_rect = (max(width, height), min(width, height), center_original)

    if best_rect is None:
        center = np.mean(hull_points, axis=0)
        return 0.0, 0.0, center

    return best_rect


def find_contiguous_clusters(points_2d: np.ndarray, eps: float = 3.0, min_samples: int = 10) -> tuple[np.ndarray, int]:
    """
    Find contiguous clusters in 2D projected points using DBSCAN.

    Args:
        points_2d: (N, 2) array of 2D projected points
        eps: Maximum distance between points in a cluster (mm)
        min_samples: Minimum points to form a cluster

    Returns:
        labels: (N,) cluster labels (-1 = noise)
        n_clusters: Number of clusters found
    """
    if len(points_2d) < min_samples:
        return np.zeros(len(points_2d), dtype=int), 1 if len(points_2d) > 0 else 0

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points_2d)
    labels = clustering.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    return labels, n_clusters


# =============================================================================
# Graph-Based Cluster Joining (New Algorithm)
# =============================================================================

def build_cluster_adjacency_graph(clusters: list[np.ndarray], vertices: np.ndarray,
                                   max_distance: float = 15.0) -> dict[int, set]:
    """
    Build adjacency graph where clusters within max_distance are connected.

    Args:
        clusters: List of arrays, each containing vertex indices for a cluster
        vertices: Full vertex array (N, 3)
        max_distance: Maximum distance (mm) between clusters to be adjacent

    Returns:
        adjacency: Dict mapping cluster_id -> set of adjacent cluster_ids
    """
    from scipy.spatial import cKDTree

    adjacency = {i: set() for i in range(len(clusters))}

    for i in range(len(clusters)):
        if len(clusters[i]) == 0:
            continue
        tree_i = cKDTree(vertices[clusters[i]])
        for j in range(i + 1, len(clusters)):
            if len(clusters[j]) == 0:
                continue
            # Find minimum distance between clusters
            distances, _ = tree_i.query(vertices[clusters[j]], k=1)
            min_dist = distances.min()
            if min_dist <= max_distance:
                adjacency[i].add(j)
                adjacency[j].add(i)

    return adjacency


def find_connected_components(adjacency: dict[int, set]) -> list[set]:
    """
    Find connected components in cluster adjacency graph.

    Args:
        adjacency: Dict mapping cluster_id -> set of adjacent cluster_ids

    Returns:
        List of sets, each containing cluster IDs in a connected component
    """
    visited = set()
    components = []

    for node in adjacency:
        if node not in visited:
            component = set()
            stack = [node]
            while stack:
                current = stack.pop()
                if current not in visited:
                    visited.add(current)
                    component.add(current)
                    stack.extend(adjacency[current] - visited)
            components.append(component)

    return components


def powerset(iterable):
    """Generate all non-empty subsets of an iterable."""
    from itertools import combinations, chain
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))


def points_in_rectangle(points_2d: np.ndarray, rect_corners: np.ndarray) -> np.ndarray:
    """
    Check which 2D points fall inside a rectangle defined by 4 corners.

    Uses cross-product method for convex polygon containment.
    Assumes corners are ordered counter-clockwise.

    Args:
        points_2d: (N, 2) array of 2D points to test
        rect_corners: (4, 2) array of rectangle corners, ordered counter-clockwise

    Returns:
        Boolean array of shape (N,) indicating which points are inside
    """
    if len(points_2d) == 0:
        return np.array([], dtype=bool)

    n = len(rect_corners)
    inside = np.ones(len(points_2d), dtype=bool)

    for i in range(n):
        p1 = rect_corners[i]
        p2 = rect_corners[(i + 1) % n]
        edge = p2 - p1
        to_point = points_2d - p1
        cross = edge[0] * to_point[:, 1] - edge[1] * to_point[:, 0]
        inside &= (cross >= -1e-10)  # Small tolerance for numerical precision

    return inside


def point_in_convex_hull_2d(points_2d: np.ndarray, hull_vertices_2d: np.ndarray) -> np.ndarray:
    """
    Check which 2D points fall inside a 2D convex hull.

    Uses the cross-product method for convex polygon containment.

    Args:
        points_2d: (N, 2) array of 2D points to test
        hull_vertices_2d: (M, 2) array of hull vertices, ordered counter-clockwise

    Returns:
        Boolean array of shape (N,) indicating which points are inside
    """
    if len(points_2d) == 0:
        return np.array([], dtype=bool)

    n = len(hull_vertices_2d)
    inside = np.ones(len(points_2d), dtype=bool)

    for i in range(n):
        p1 = hull_vertices_2d[i]
        p2 = hull_vertices_2d[(i + 1) % n]
        edge = p2 - p1
        to_point = points_2d - p1
        cross = edge[0] * to_point[:, 1] - edge[1] * to_point[:, 0]
        inside &= (cross >= -1e-10)  # Small tolerance for numerical precision

    return inside


def evaluate_cluster_combination(
    cluster_indices: tuple,
    clusters: list[np.ndarray],
    region_vertices: np.ndarray,
    region_thicknesses: np.ndarray,
    thin_threshold: float,
    quality_threshold: float = 0.8,
    grid_resolution: float = 0.5
) -> tuple[dict | None, bool]:
    """
    Evaluate a combination of clusters for aperture fitting.

    Algorithm:
    1. Merge specified clusters
    2. Check quality: ≥quality_threshold of merged cluster vertices must be thin
    3. Compute convex hull of merged vertices
    4. Find largest inscribed rectangle within the convex hull
    5. Return aperture info and whether it passed quality threshold

    The quality check validates that the merged cluster vertices are predominantly
    thin (≤ thin_threshold). Since clusters come from DBSCAN on thin vertices,
    this should typically be ~100%, but serves as a sanity check.

    Args:
        cluster_indices: Tuple of cluster IDs to merge
        clusters: All clusters (list of vertex index arrays, indices into region_vertices)
        region_vertices: All vertices in the region (N, 3)
        region_thicknesses: All thicknesses in the region (N,)
        thin_threshold: Thickness threshold for "thin" classification
        quality_threshold: Minimum fraction of merged cluster vertices that must be thin
        grid_resolution: Grid resolution for inscribed rectangle algorithm

    Returns:
        Tuple of (aperture_dict or None, passes_threshold)
    """
    # 1. Merge cluster vertices (clusters contain indices into region_vertices)
    merged_indices = np.concatenate([clusters[i] for i in cluster_indices])
    merged_vertices = region_vertices[merged_indices]
    merged_thicknesses = region_thicknesses[merged_indices]

    if len(merged_vertices) < 4:
        return None, False

    # 2. Quality check: what % of MERGED CLUSTER vertices are thin?
    # This validates the cluster merging produces a valid thin region
    thin_count = (merged_thicknesses <= thin_threshold).sum()
    thin_percentage = thin_count / len(merged_thicknesses)

    # Check if this combination passes the quality threshold
    passes_threshold = thin_percentage >= quality_threshold

    # 3. Project merged cluster vertices to 2D tangent plane via PCA
    centroid = merged_vertices.mean(axis=0)
    centered = merged_vertices - centroid
    try:
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    except Exception:
        return None, passes_threshold

    basis = Vt[:2]  # First two principal components
    merged_2d = centered @ basis.T  # Project merged cluster vertices to 2D

    # 4. Find largest inscribed rectangle within the merged cluster point cloud
    width, height, center_2d, rect_corners_2d = largest_inscribed_rectangle(
        merged_2d, grid_resolution=grid_resolution
    )

    if width < 5.0 or height < 5.0 or rect_corners_2d is None:
        return None, passes_threshold

    # 5. Transform rectangle corners back to 3D
    corners_3d = (rect_corners_2d @ basis) + centroid

    # 6. Return aperture info
    rect_width = max(width, height)
    rect_height = min(width, height)

    aperture = {
        'width_mm': float(rect_width),
        'height_mm': float(rect_height),
        'area_mm2': float(rect_width * rect_height),
        'centroid_3d': centroid.tolist(),
        'rect_corners_3d': corners_3d.tolist(),
        'thin_percentage': float(thin_percentage),
        'n_clusters_merged': len(cluster_indices),
        'n_vertices': len(merged_indices),
        '_merged_indices': merged_indices  # Internal use for thickness stats
    }

    return aperture, passes_threshold


def analyze_region_improved(
    region_name: str,
    region_vertices: np.ndarray,
    region_thicknesses: np.ndarray,
    percentile: float = 25,
    quality_threshold: float = 0.8,
    dbscan_eps: float = 5.0,
    dbscan_min_samples: int = 10,
    adjacency_max_dist: float = 15.0,
    max_clusters_per_component: int = 8
) -> dict | None:
    """
    Improved region analysis with graph-based cluster joining.

    Algorithm:
    1. Find thin vertices (≤percentile threshold)
    2. DBSCAN clustering on thin vertices
    3. Build cluster adjacency graph
    4. For each connected component, try all cluster combinations
    5. Keep largest valid rectangle that passes quality threshold

    Args:
        region_name: Name of the region (for logging)
        region_vertices: All vertices in the region (N, 3)
        region_thicknesses: All thicknesses in the region (N,)
        percentile: Thickness percentile for "thin" classification
        quality_threshold: Min fraction of rectangle vertices that must be thin
        dbscan_eps: DBSCAN clustering radius (mm)
        dbscan_min_samples: DBSCAN minimum samples per cluster
        adjacency_max_dist: Max distance for clusters to be adjacent (mm)
        max_clusters_per_component: Max clusters to consider in exhaustive search (default: 8)

    Returns:
        Best aperture dict, or None if no valid aperture found
    """
    if len(region_vertices) < 20:
        return None

    # 1. Find thin threshold (percentile of region)
    thin_threshold = np.percentile(region_thicknesses, percentile)

    # 2. Identify thin vertices
    thin_mask = region_thicknesses <= thin_threshold
    thin_vertices = region_vertices[thin_mask]
    thin_indices = np.where(thin_mask)[0]

    if len(thin_vertices) < dbscan_min_samples:
        return None

    # 3. DBSCAN clustering on thin vertices (in 3D)
    clustering = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(thin_vertices)
    labels = clustering.labels_
    unique_labels = set(labels) - {-1}

    if len(unique_labels) == 0:
        return None

    # 4. Build clusters list (indices into region_vertices, not thin_vertices)
    # Sort by cluster size (largest first) so we prioritize bigger clusters
    cluster_info = []
    for label in unique_labels:
        cluster_mask = labels == label
        indices = thin_indices[cluster_mask]
        cluster_info.append((len(indices), indices))

    # Sort by size descending
    cluster_info.sort(key=lambda x: x[0], reverse=True)
    clusters = [info[1] for info in cluster_info]

    print(f"    {region_name}: Found {len(clusters)} clusters")

    # 5. Build adjacency graph
    adjacency = build_cluster_adjacency_graph(
        clusters, region_vertices, max_distance=adjacency_max_dist
    )

    # 6. Find connected components
    components = find_connected_components(adjacency)

    # 7. For each component, try cluster combinations
    # Only keep combinations that pass quality threshold (≥80% of hull vertices are thin)
    # Among valid combinations, return the one with largest inscribed rectangle
    best_aperture = None
    best_area = 0

    for component in components:
        component_list = list(component)

        # If component is too large, keep only the largest clusters
        if len(component_list) > max_clusters_per_component:
            # Sort by cluster size and keep top N
            component_list.sort(key=lambda i: len(clusters[i]), reverse=True)
            component_list = component_list[:max_clusters_per_component]
            print(f"    {region_name}: Limiting component from {len(component)} to {max_clusters_per_component} largest clusters")

        # Try all combinations of clusters in this (possibly reduced) component
        for combo in powerset(component_list):
            aperture, passes_threshold = evaluate_cluster_combination(
                cluster_indices=combo,
                clusters=clusters,
                region_vertices=region_vertices,
                region_thicknesses=region_thicknesses,
                thin_threshold=thin_threshold,
                quality_threshold=quality_threshold
            )

            # Only consider combinations that pass quality threshold
            # (≥80% of region vertices inside the convex hull are thin)
            if aperture and passes_threshold:
                if aperture['area_mm2'] > best_area:
                    best_aperture = aperture
                    best_area = aperture['area_mm2']

    # 8. Add thickness stats to best aperture
    if best_aperture:
        merged_idx = best_aperture['_merged_indices']
        cluster_thick = region_thicknesses[merged_idx]
        best_aperture['mean_thickness_mm'] = float(cluster_thick.mean())
        best_aperture['min_thickness_mm'] = float(cluster_thick.min())
        best_aperture['max_thickness_mm'] = float(cluster_thick.max())
        best_aperture['n_vertices'] = len(merged_idx)
        del best_aperture['_merged_indices']  # Clean up internal field

        print(f"    {region_name}: Best aperture {best_aperture['width_mm']:.1f}x{best_aperture['height_mm']:.1f}mm "
              f"({best_aperture['thin_percentage']*100:.0f}% thin, {best_aperture['n_clusters_merged']} clusters merged)")
    else:
        print(f"    {region_name}: No valid aperture found (no cluster combinations passed {quality_threshold*100:.0f}% threshold)")

    return best_aperture


def fit_rectangular_aperture(vertices: np.ndarray, thin_mask: np.ndarray,
                              cluster_eps: float = 5.0, use_inscribed: bool = True) -> list[dict]:
    """
    Fit rectangles to contiguous thin region clusters.

    Projects 3D vertices onto a local tangent plane (using PCA), finds
    contiguous clusters using DBSCAN, and computes rectangle for each.

    Args:
        vertices: (N, 3) all vertices in the region
        thin_mask: (N,) boolean mask for thin vertices
        cluster_eps: DBSCAN epsilon for clustering (mm)
        use_inscribed: If True, find largest inscribed rectangle (fits inside).
                      If False, find minimum bounding rectangle (contains all points).

    Returns:
        List of aperture dictionaries, one per contiguous cluster, sorted by area (largest first).
        Each dict contains: width_mm, height_mm, area_mm2, centroid_3d, n_vertices,
                           rect_corners_3d (for visualization)
    """
    thin_vertices = vertices[thin_mask]

    if len(thin_vertices) < 3:
        return []

    # IMPORTANT: Cluster in 3D space, not 2D projection
    # This preserves separation between regions at different depths
    # (e.g., left/right occipital separated by thick midline ridge)
    labels, n_clusters = find_contiguous_clusters(thin_vertices, eps=cluster_eps)

    if n_clusters == 0:
        return []

    apertures = []
    for cluster_id in range(n_clusters):
        cluster_mask = labels == cluster_id
        if cluster_mask.sum() < 10:  # Skip tiny clusters
            continue

        cluster_3d = thin_vertices[cluster_mask]

        # Compute cluster-specific PCA for better rectangle fitting
        cluster_centroid_3d = np.mean(cluster_3d, axis=0)
        cluster_centered = cluster_3d - cluster_centroid_3d

        cluster_cov = np.cov(cluster_centered.T)
        cluster_eig_vals, cluster_eig_vecs = np.linalg.eigh(cluster_cov)
        cluster_idx = np.argsort(cluster_eig_vals)[::-1]
        cluster_eig_vecs = cluster_eig_vecs[:, cluster_idx]

        # Project cluster onto its own tangent plane
        cluster_2d_local = cluster_centered @ cluster_eig_vecs[:, :2]

        # Find rectangle for this cluster
        if use_inscribed:
            # Largest rectangle that fits INSIDE the thin region
            width, height, center_2d, rect_corners_2d = largest_inscribed_rectangle(cluster_2d_local, grid_resolution=0.5)
        else:
            # Minimum bounding rectangle (contains all points)
            width, height, center_2d = minimum_bounding_rectangle(cluster_2d_local)
            rect_corners_2d = get_rectangle_corners(cluster_2d_local)

        if width < 5.0 or height < 5.0:  # Skip small rectangles (< 5mm)
            continue

        # Transform rectangle corners back to 3D
        if rect_corners_2d is not None:
            # Corners are in cluster's local 2D PCA space (centered on cluster centroid)
            # Add zero for the third component (normal direction)
            corners_2d_3col = np.hstack([rect_corners_2d, np.zeros((4, 1))])
            # Transform from PCA space back to world coordinates
            corners_3d_centered = corners_2d_3col @ cluster_eig_vecs.T
            rect_corners_3d = corners_3d_centered + cluster_centroid_3d
            # Note: We keep the true rectangle corners (not snapped to surface)
            # for accurate visualization of the transducer footprint
        else:
            rect_corners_3d = None

        # Get indices of thin vertices in this cluster (relative to thin_mask)
        # This is used internally for thickness stats computation
        cluster_indices = np.where(cluster_mask)[0]

        apertures.append({
            'width_mm': float(width),
            'height_mm': float(height),
            'area_mm2': float(width * height),
            'centroid_3d': cluster_centroid_3d.tolist(),
            'n_vertices': int(cluster_mask.sum()),
            'rect_corners_3d': rect_corners_3d.tolist() if rect_corners_3d is not None else None,
            '_cluster_indices': cluster_indices  # Internal use only, stripped before JSON save
        })

    # Sort by area (largest first)
    apertures.sort(key=lambda x: x['area_mm2'], reverse=True)

    return apertures


def get_rectangle_corners(points_2d: np.ndarray) -> np.ndarray | None:
    """
    Get the 4 corners of the minimum bounding rectangle.

    Args:
        points_2d: (N, 2) array of 2D points

    Returns:
        (4, 2) array of corner coordinates, or None if degenerate
    """
    if len(points_2d) < 3:
        return None

    try:
        hull = ConvexHull(points_2d)
        hull_points = points_2d[hull.vertices]
    except Exception:
        return None

    n = len(hull_points)
    min_area = float('inf')
    best_corners = None

    for i in range(n):
        edge = hull_points[(i + 1) % n] - hull_points[i]
        edge_len = np.linalg.norm(edge)
        if edge_len < 1e-10:
            continue

        u = edge / edge_len
        v = np.array([-u[1], u[0]])

        projections = hull_points @ np.column_stack([u, v])
        min_proj = projections.min(axis=0)
        max_proj = projections.max(axis=0)

        width = max_proj[0] - min_proj[0]
        height = max_proj[1] - min_proj[1]
        area = width * height

        if area < min_area:
            min_area = area
            # Compute 4 corners in original coordinate system
            corners_rotated = np.array([
                [min_proj[0], min_proj[1]],
                [max_proj[0], min_proj[1]],
                [max_proj[0], max_proj[1]],
                [min_proj[0], max_proj[1]]
            ])
            # Transform back
            best_corners = corners_rotated[:, 0:1] * u + corners_rotated[:, 1:2] * v

    return best_corners


def largest_inscribed_rectangle(points_2d: np.ndarray, grid_resolution: float = 0.5) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    Find the largest axis-aligned rectangle that fits inside a 2D point cloud.

    Uses a grid-based approach: creates a binary occupancy grid from the point cloud,
    then finds the largest rectangle in the grid using the histogram method.

    Args:
        points_2d: (N, 2) array of 2D points representing the region
        grid_resolution: Grid cell size in mm

    Returns:
        width: Rectangle width (mm) - larger dimension
        height: Rectangle height (mm) - smaller dimension
        center_2d: Rectangle center in 2D (PCA space)
        corners_2d: (4, 2) array of corner coordinates (PCA space)
    """
    if len(points_2d) < 10:
        return 0.0, 0.0, np.array([0.0, 0.0]), None

    # Compute bounding box
    min_xy = points_2d.min(axis=0)
    max_xy = points_2d.max(axis=0)

    # Create grid - note: grid[i,j] corresponds to point [x,y] where x=i, y=j
    grid_size = ((max_xy - min_xy) / grid_resolution).astype(int) + 1
    if grid_size[0] < 2 or grid_size[1] < 2:
        return 0.0, 0.0, np.mean(points_2d, axis=0), None

    # Limit grid size for performance
    max_grid = 300
    if max(grid_size) > max_grid:
        scale = max(grid_size) / max_grid
        grid_resolution *= scale
        grid_size = ((max_xy - min_xy) / grid_resolution).astype(int) + 1

    # Create binary occupancy grid
    grid = np.zeros(grid_size, dtype=bool)

    # Mark cells that contain points
    point_indices = ((points_2d - min_xy) / grid_resolution).astype(int)
    point_indices = np.clip(point_indices, 0, grid_size - 1)
    grid[point_indices[:, 0], point_indices[:, 1]] = True

    # For sparse point clouds, dilate first to connect nearby points, then fill holes
    from scipy import ndimage

    # Dilate to connect nearby points (sparse vertex sampling)
    # Use small dilation that's proportional to grid resolution
    dilation_size = max(1, int(2.0 / grid_resolution))  # ~2mm dilation
    grid_dilated = ndimage.binary_dilation(grid, iterations=dilation_size)

    # Fill holes in the dilated grid
    grid_filled = ndimage.binary_fill_holes(grid_dilated)

    # Erode back by same amount to approximate original boundary
    # Then erode a bit more to ensure we're safely inside
    total_erosion = dilation_size + 2  # +2 for safety margin (~1mm)
    grid = ndimage.binary_erosion(grid_filled, iterations=total_erosion)

    # If erosion removed everything, try with less erosion
    if not grid.any():
        grid = ndimage.binary_erosion(grid_filled, iterations=dilation_size)
    if not grid.any():
        grid = grid_filled  # Fall back to filled

    if not grid.any():
        return 0.0, 0.0, np.mean(points_2d, axis=0), None

    # Find largest rectangle in binary grid using histogram method
    # We iterate over rows (first axis, x) and build histogram along columns (y)
    best_area = 0
    best_rect = None  # (x_start, y_start, x_size, y_size) in grid coords

    # For each x row, compute histogram of consecutive 1s (height in y direction)
    heights = np.zeros(grid_size[1], dtype=int)

    for x in range(grid_size[0]):
        # Update heights - height represents how many consecutive 1s above
        for y in range(grid_size[1]):
            if grid[x, y]:
                heights[y] += 1
            else:
                heights[y] = 0

        # Find largest rectangle in histogram using stack method
        stack = []  # (start_y, height_in_x)
        for y in range(grid_size[1] + 1):
            h = heights[y] if y < grid_size[1] else 0
            start_y = y

            while stack and stack[-1][1] > h:
                prev_y, prev_h = stack.pop()
                width_y = y - prev_y  # Width in y direction
                area = width_y * prev_h  # prev_h is height in x direction

                if area > best_area:
                    best_area = area
                    # Rectangle in grid coords: starts at (x - prev_h + 1, prev_y)
                    # with size (prev_h in x, width_y in y)
                    best_rect = (x - prev_h + 1, prev_y, prev_h, width_y)

                start_y = prev_y

            stack.append((start_y, h))

    if best_rect is None or best_area == 0:
        return 0.0, 0.0, np.mean(points_2d, axis=0), None

    # Convert grid coordinates back to 2D PCA coordinates
    x_start, y_start, x_size, y_size = best_rect

    # Convert to real coordinates
    x0 = min_xy[0] + x_start * grid_resolution
    y0 = min_xy[1] + y_start * grid_resolution
    x1 = x0 + x_size * grid_resolution
    y1 = y0 + y_size * grid_resolution

    dim_x = x_size * grid_resolution
    dim_y = y_size * grid_resolution

    # Width is larger dimension, height is smaller
    if dim_x >= dim_y:
        width_mm, height_mm = dim_x, dim_y
    else:
        width_mm, height_mm = dim_y, dim_x

    center_2d = np.array([(x0 + x1) / 2, (y0 + y1) / 2])

    # Corners in PCA space (counter-clockwise from bottom-left)
    corners_2d = np.array([
        [x0, y0],
        [x1, y0],
        [x1, y1],
        [x0, y1]
    ])

    return float(width_mm), float(height_mm), center_2d, corners_2d


def analyze_region(vertices: np.ndarray, thicknesses: np.ndarray,
                   region_indices: np.ndarray, percentile: float = 25,
                   cluster_eps: float = 3.0) -> dict:
    """
    Complete aperture analysis for one anatomical region.

    Finds all contiguous thin regions (apertures) within the anatomical region.
    For regions like occipital, this may detect multiple apertures separated by
    thick ridges (e.g., external occipital protuberance).

    Args:
        vertices: (N, 3) all surface vertices
        thicknesses: (N,) thickness values for all vertices
        region_indices: indices of vertices in this region
        percentile: percentile threshold for thin region
        cluster_eps: DBSCAN epsilon for clustering (mm)

    Returns:
        Dictionary with:
        - apertures: list of aperture dicts (one per contiguous cluster)
        - region_stats: overall statistics for the region
    """
    if len(region_indices) == 0:
        return {
            'apertures': [],
            'region_stats': {
                'n_vertices_total': 0,
                'n_vertices_thin': 0,
                'threshold_thickness_mm': 0.0,
                'mean_thickness_mm': 0.0,
                'min_thickness_mm': 0.0,
                'max_thickness_mm': 0.0
            }
        }

    # Extract region data
    region_verts = vertices[region_indices]
    region_thick = thicknesses[region_indices]

    # Find thin vertices
    thin_mask, threshold = find_thin_vertices(region_thick, percentile)
    thin_thick = region_thick[thin_mask]
    thin_verts = region_verts[thin_mask]

    # Find contiguous apertures using DBSCAN clustering
    apertures = fit_rectangular_aperture(region_verts, thin_mask, cluster_eps)

    # Add thickness statistics to each aperture
    for aperture in apertures:
        cluster_indices = aperture['_cluster_indices']
        cluster_thick = thin_thick[cluster_indices]
        aperture['mean_thickness_mm'] = float(cluster_thick.mean()) if len(cluster_thick) > 0 else 0.0
        aperture['min_thickness_mm'] = float(cluster_thick.min()) if len(cluster_thick) > 0 else 0.0
        aperture['max_thickness_mm'] = float(cluster_thick.max()) if len(cluster_thick) > 0 else 0.0
        # Remove internal key before returning (not needed for JSON serialization)
        del aperture['_cluster_indices']

    return {
        'apertures': apertures,
        'region_stats': {
            'n_vertices_total': len(region_indices),
            'n_vertices_thin': int(thin_mask.sum()),
            'threshold_thickness_mm': float(threshold),
            'mean_thickness_mm': float(thin_thick.mean()) if len(thin_thick) > 0 else 0.0,
            'min_thickness_mm': float(thin_thick.min()) if len(thin_thick) > 0 else 0.0,
            'max_thickness_mm': float(thin_thick.max()) if len(thin_thick) > 0 else 0.0
        }
    }


def analyze_all_regions(vertices: np.ndarray, thicknesses: np.ndarray,
                        regions: dict[str, np.ndarray], percentile: float = 25,
                        cluster_eps: float = 5.0,
                        quality_threshold: float = 0.8,
                        adjacency_max_dist: float = 15.0,
                        min_aperture_area: float = 0.0) -> dict:
    """
    Analyze all anatomical regions using improved graph-based cluster joining.

    For each of the 4 regions (temporal_left, temporal_right, occipital_left,
    occipital_right), finds the largest valid aperture that passes the quality
    threshold (≥80% of vertices inside rectangle must be thin).

    Args:
        vertices: (N, 3) all surface vertices
        thicknesses: (N,) thickness values for all vertices
        regions: dict mapping region name to vertex indices
        percentile: percentile threshold for thin region (default: 25)
        cluster_eps: DBSCAN epsilon for clustering (mm, default: 5.0)
        quality_threshold: Min fraction of rectangle vertices that must be thin (default: 0.8)
        adjacency_max_dist: Max distance for clusters to be adjacent (mm, default: 15.0)
        min_aperture_area: Minimum aperture area to keep (mm²)

    Returns:
        Dictionary with:
        - apertures: dict mapping aperture name to aperture data
        - region_stats: dict mapping region name to region statistics
    """
    all_apertures = {}
    all_stats = {}

    # Process all 4 regions
    region_names = ['temporal_left', 'temporal_right', 'occipital_left', 'occipital_right']

    for region_name in region_names:
        if region_name not in regions:
            continue

        region_indices = regions[region_name]
        if len(region_indices) == 0:
            continue

        region_verts = vertices[region_indices]
        region_thick = thicknesses[region_indices]

        # Use improved analysis with graph-based cluster joining
        aperture = analyze_region_improved(
            region_name=region_name,
            region_vertices=region_verts,
            region_thicknesses=region_thick,
            percentile=percentile,
            quality_threshold=quality_threshold,
            dbscan_eps=cluster_eps,
            adjacency_max_dist=adjacency_max_dist
        )

        # Compute region stats
        thin_threshold = np.percentile(region_thick, percentile)
        thin_mask = region_thick <= thin_threshold
        thin_thick = region_thick[thin_mask]

        all_stats[region_name] = {
            'n_vertices_total': len(region_indices),
            'n_vertices_thin': int(thin_mask.sum()),
            'threshold_thickness_mm': float(thin_threshold),
            'mean_thickness_mm': float(thin_thick.mean()) if len(thin_thick) > 0 else 0.0,
            'min_thickness_mm': float(thin_thick.min()) if len(thin_thick) > 0 else 0.0,
            'max_thickness_mm': float(thin_thick.max()) if len(thin_thick) > 0 else 0.0
        }

        # Add aperture if valid and meets minimum area
        if aperture and aperture['area_mm2'] >= min_aperture_area:
            all_apertures[region_name] = aperture

    return {
        'apertures': all_apertures,
        'region_stats': all_stats
    }


# =============================================================================
# Caching and Results I/O
# =============================================================================

def get_results_dir(specimen_id: str, results_dir: str = "results") -> Path:
    """Get results directory for a specimen, creating if needed."""
    path = Path(results_dir) / specimen_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_cache_path(specimen_id: str, results_dir: str = "results") -> Path:
    """Get path for cached thickness data."""
    return get_results_dir(specimen_id, results_dir) / "thickness_data.npz"


def cache_exists(specimen_id: str, results_dir: str = "results") -> bool:
    """Check if cached thickness data exists for a specimen."""
    return get_cache_path(specimen_id, results_dir).exists()


def save_thickness_cache(specimen_id: str, verts: np.ndarray, faces: np.ndarray,
                         thicknesses: np.ndarray, center: np.ndarray,
                         results_dir: str = "results") -> None:
    """
    Save thickness map to NPZ for reuse.

    Args:
        specimen_id: Specimen identifier (e.g., "A0001")
        verts: Surface mesh vertices
        faces: Surface mesh faces
        thicknesses: Per-vertex thickness values
        center: Brain cavity center
        results_dir: Base results directory
    """
    cache_path = get_cache_path(specimen_id, results_dir)
    np.savez_compressed(
        cache_path,
        vertices=verts,
        faces=faces,
        thicknesses=thicknesses,
        center=center
    )
    print(f"Saved thickness cache to {cache_path}")


def load_thickness_cache(specimen_id: str, results_dir: str = "results") -> dict | None:
    """
    Load cached thickness data if it exists.

    Returns:
        Dictionary with vertices, faces, thicknesses, center or None if not cached
    """
    cache_path = get_cache_path(specimen_id, results_dir)
    if not cache_path.exists():
        return None

    data = np.load(cache_path)
    result = {
        'vertices': data['vertices'],
        'faces': data['faces'],
        'thicknesses': data['thicknesses'],
        'center': data['center']
    }
    print(f"Loaded thickness cache from {cache_path}")
    return result


def save_aperture_results(specimen_id: str, apertures: dict, percentile: float,
                          results_dir: str = "results") -> None:
    """
    Save aperture analysis results to JSON.

    Args:
        specimen_id: Specimen identifier
        apertures: Dictionary mapping region name to aperture results
        percentile: Percentile threshold used
        results_dir: Base results directory
    """
    output_path = get_results_dir(specimen_id, results_dir) / "apertures.json"

    result = {
        'specimen_id': specimen_id,
        'timestamp': datetime.now().isoformat(),
        'percentile_threshold': percentile,
        'regions': apertures
    }

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Saved aperture results to {output_path}")


def load_aperture_results(specimen_id: str, results_dir: str = "results") -> dict | None:
    """Load aperture results if they exist."""
    output_path = get_results_dir(specimen_id, results_dir) / "apertures.json"
    if not output_path.exists():
        return None

    with open(output_path, 'r') as f:
        return json.load(f)


# =============================================================================
# Population Statistics
# =============================================================================

def compute_population_statistics(all_results: list[dict]) -> dict:
    """
    Aggregate aperture statistics across all specimens.

    Combines left/right regions into unified transducer types:
    - temporal: combines temporal_left and temporal_right
    - occipital: combines occipital_left and occipital_right

    For each specimen, we take the measurements from both sides to get
    population-level statistics for transducer design.

    Args:
        all_results: List of aperture result dictionaries (from load_aperture_results)

    Returns:
        Dictionary with aggregated transducer statistics
    """
    if not all_results:
        return {}

    # Map individual regions to transducer types
    region_to_transducer = {
        'temporal_left': 'temporal',
        'temporal_right': 'temporal',
        'occipital_left': 'occipital',
        'occipital_right': 'occipital',
    }

    # Collect measurements by transducer type (combining left/right)
    transducer_data = {
        'temporal': {'width': [], 'height': [], 'area': [], 'mean_thickness': [], 'min_thickness': []},
        'occipital': {'width': [], 'height': [], 'area': [], 'mean_thickness': [], 'min_thickness': []}
    }

    for result in all_results:
        for region_name, data in result.get('regions', {}).items():
            transducer_type = region_to_transducer.get(region_name)
            if transducer_type:
                transducer_data[transducer_type]['width'].append(data['width_mm'])
                transducer_data[transducer_type]['height'].append(data['height_mm'])
                transducer_data[transducer_type]['area'].append(data['area_mm2'])
                transducer_data[transducer_type]['mean_thickness'].append(data['mean_thickness_mm'])
                transducer_data[transducer_type]['min_thickness'].append(data['min_thickness_mm'])

    # Compute statistics for each transducer type
    stats = {
        'n_specimens': len(all_results),
        'n_measurements_per_transducer': {
            'temporal': len(transducer_data['temporal']['width']),
            'occipital': len(transducer_data['occipital']['width'])
        },
        'timestamp': datetime.now().isoformat(),
        'transducers': {}
    }

    for transducer_type, data in transducer_data.items():
        transducer_stats = {}
        for metric, values in data.items():
            if len(values) == 0:
                continue
            arr = np.array(values)
            transducer_stats[f'{metric}_mm'] = {
                'mean': float(arr.mean()),
                'std': float(arr.std()),
                'min': float(arr.min()),
                'max': float(arr.max()),
                'p5': float(np.percentile(arr, 5)),
                'p25': float(np.percentile(arr, 25)),
                'p50': float(np.percentile(arr, 50)),
                'p75': float(np.percentile(arr, 75)),
                'p95': float(np.percentile(arr, 95))
            }
        stats['transducers'][transducer_type] = transducer_stats

    return stats


def save_population_statistics(stats: dict, output_path: str = "results/population_summary.json") -> None:
    """Save population statistics to JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved population statistics to {output_path}")


# =============================================================================
# Visualization Helpers
# =============================================================================

def create_region_colors(vertices: np.ndarray, regions: dict[str, np.ndarray]) -> np.ndarray:
    """
    Create color array for region visualization.

    Args:
        vertices: (N, 3) all surface vertices
        regions: Dictionary mapping region name to vertex indices

    Returns:
        (N,) array with region labels (0=unassigned, 1-4 for the 4 regions)
    """
    colors = np.zeros(len(vertices), dtype=int)

    region_ids = {
        'temporal_left': 1,
        'temporal_right': 2,
        'occipital_left': 3,
        'occipital_right': 4,
        'occipital': 3  # Legacy support
    }

    for region_name, indices in regions.items():
        if region_name in region_ids:
            colors[indices] = region_ids[region_name]

    return colors


def get_thin_vertex_indices(vertices: np.ndarray, thicknesses: np.ndarray,
                            regions: dict[str, np.ndarray], percentile: float = 25) -> dict[str, np.ndarray]:
    """
    Get indices of thin vertices for each region.

    Returns:
        Dictionary mapping region name to array of thin vertex indices (global indices)
    """
    thin_indices = {}

    for region_name, region_idx in regions.items():
        if len(region_idx) == 0:
            thin_indices[region_name] = np.array([], dtype=int)
            continue

        region_thick = thicknesses[region_idx]
        threshold = np.percentile(region_thick, percentile)
        thin_mask = region_thick <= threshold
        thin_indices[region_name] = region_idx[thin_mask]

    return thin_indices
