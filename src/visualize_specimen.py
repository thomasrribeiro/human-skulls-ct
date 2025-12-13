"""
Visualize Cached Specimen Results

This script loads pre-computed thickness data and aperture analysis from the
results directory and launches an interactive 3D viewer.

Usage:
    # Visualize a specific specimen
    python visualize_specimen.py A0001

    # Or run from the specimen's results directory
    cd results/A0001 && python ../../visualize_specimen.py

    # Specify custom results directory
    python visualize_specimen.py A0001 --results-dir /path/to/results
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.aperture_analysis import segment_regions, load_thickness_cache, load_aperture_results


def visualize_specimen(specimen_id: str, results_dir: str = "results",
                       cmap: str = 'jet', clim: tuple = (3, 10)):
    """
    Load cached data and launch interactive 3D visualization.

    Args:
        specimen_id: Specimen identifier (e.g., "A0001")
        results_dir: Base results directory
        cmap: Colormap for thickness visualization
        clim: Color limits (min, max) in mm
    """
    print(f"Loading cached data for specimen: {specimen_id}")

    # Load thickness data
    cached = load_thickness_cache(specimen_id, results_dir)
    if cached is None:
        print(f"ERROR: No thickness data found for {specimen_id}")
        print(f"Expected file: {results_dir}/{specimen_id}/thickness_data.npz")
        sys.exit(1)

    verts = cached['vertices']
    faces = cached['faces']
    thicknesses = cached['thicknesses']
    center = cached['center']

    print(f"  Loaded {len(verts)} vertices, {len(faces)} faces")
    print(f"  Thickness range: {np.nanmin(thicknesses):.1f} - {np.nanmax(thicknesses):.1f} mm")

    # Load aperture results
    aperture_data = load_aperture_results(specimen_id, results_dir)
    if aperture_data is None:
        print(f"WARNING: No aperture data found for {specimen_id}")
        apertures = {}
    else:
        # Handle both old format (with 'regions' key) and new format (direct apertures dict)
        if 'regions' in aperture_data:
            apertures = aperture_data['regions']
        else:
            # Filter out metadata keys
            apertures = {k: v for k, v in aperture_data.items()
                        if k not in ['specimen_id', 'percentile_threshold']}

    print(f"  Found {len(apertures)} apertures")
    for ap_name, ap in apertures.items():
        if isinstance(ap, dict) and 'width_mm' in ap:
            print(f"    {ap_name}: {ap['width_mm']:.1f} x {ap['height_mm']:.1f} mm")

    # Compute regions for visualization
    print("\nComputing region segmentation...")
    regions = segment_regions(verts, center)

    # Launch visualization
    print("\nLaunching interactive 3D viewer...")
    print("Controls:")
    print("  - Checkboxes toggle layer visibility")
    print("  - Left-click + drag to rotate")
    print("  - Scroll to zoom")
    print("  - Right-click + drag to pan")

    visualize_with_toggles(
        verts=verts,
        faces=faces,
        thicknesses=thicknesses,
        center=center,
        regions=regions,
        apertures=apertures,
        specimen_id=specimen_id,
        cmap=cmap,
        clim=clim
    )


def visualize_with_toggles(verts: np.ndarray, faces: np.ndarray,
                           thicknesses: np.ndarray,
                           center: np.ndarray,
                           regions: dict,
                           apertures: dict,
                           specimen_id: str,
                           cmap: str = 'jet', clim: tuple = (3, 10)):
    """
    Launch interactive 3D visualization with toggle controls for layers.

    Features:
    - Checkbox to toggle thickness map visibility
    - Checkbox to toggle region boundaries
    - Checkbox to toggle aperture rectangles
    - 3D rectangles showing transducer aperture footprints
    """
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
        if not isinstance(ap_data, dict):
            continue

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
    plotter.add_title(f"Skull Thickness & Apertures - {specimen_id}", font_size=12)
    plotter.add_axes()

    # Set initial camera position (lateral view)
    plotter.camera_position = 'xz'

    plotter.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize cached specimen results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        'specimen_id', type=str, nargs='?', default=None,
        help='Specimen ID (e.g., A0001). If not provided, tries to detect from current directory.'
    )
    parser.add_argument(
        '--results-dir', type=str, default='results',
        help='Results directory (default: results)'
    )
    parser.add_argument(
        '--cmap', type=str, default='jet',
        help='Colormap for thickness visualization (default: jet)'
    )
    parser.add_argument(
        '--clim-min', type=float, default=3.0,
        help='Minimum thickness for colormap (default: 3.0 mm)'
    )
    parser.add_argument(
        '--clim-max', type=float, default=10.0,
        help='Maximum thickness for colormap (default: 10.0 mm)'
    )

    args = parser.parse_args()

    # Try to detect specimen ID from current directory if not provided
    if args.specimen_id is None:
        cwd = Path.cwd()
        # Check if current directory looks like a specimen results directory
        if (cwd / 'thickness_data.npz').exists():
            args.specimen_id = cwd.name
            args.results_dir = str(cwd.parent)
            print(f"Detected specimen: {args.specimen_id} (from current directory)")
        else:
            print("ERROR: No specimen ID provided and not in a results directory")
            print("Usage: python visualize_specimen.py A0001")
            print("   or: cd results/A0001 && python ../../visualize_specimen.py")
            sys.exit(1)

    visualize_specimen(
        specimen_id=args.specimen_id,
        results_dir=args.results_dir,
        cmap=args.cmap,
        clim=(args.clim_min, args.clim_max)
    )


if __name__ == "__main__":
    main()
