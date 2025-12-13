"""
Compute Population Statistics for Aperture Dimensions

This script analyzes all completed specimens in the results/ folder and computes
statistics for aperture dimensions across temporal and occipital regions.

Usage:
    python compute_statistics.py
    python compute_statistics.py --results-dir /path/to/results
"""

import json
import argparse
from pathlib import Path
import numpy as np


def load_all_apertures(results_dir: str = "results") -> list[dict]:
    """
    Load all aperture results from specimens that have apertures.json.

    Returns:
        List of aperture data dictionaries
    """
    results_path = Path(results_dir)
    all_data = []

    for specimen_dir in sorted(results_path.iterdir()):
        if not specimen_dir.is_dir():
            continue

        apertures_file = specimen_dir / "apertures.json"
        if not apertures_file.exists():
            print(f"  Skipping {specimen_dir.name} (no apertures.json)")
            continue

        with open(apertures_file, 'r') as f:
            data = json.load(f)
            all_data.append(data)

    return all_data


def compute_statistics(values: list[float]) -> dict:
    """Compute statistics for a list of values."""
    if not values:
        return {
            'n': 0,
            'mean': None,
            'std': None,
            'min': None,
            'max': None,
            'median': None,
            'q25': None,
            'q75': None
        }

    arr = np.array(values)
    return {
        'n': len(arr),
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'median': float(np.median(arr)),
        'q25': float(np.percentile(arr, 25)),
        'q75': float(np.percentile(arr, 75))
    }


def analyze_apertures(all_data: list[dict]) -> dict:
    """
    Analyze aperture dimensions across all specimens.

    Groups apertures by region type:
    - temporal_left
    - temporal_right
    - temporal_combined (both left and right)
    - occipital_0 (first occipital aperture)
    - occipital_1 (second occipital aperture)
    - occipital_combined (all occipital apertures)

    Returns:
        Dictionary with statistics for each region
    """
    # Collect values by region
    region_data = {
        'temporal_left': {'width': [], 'height': [], 'area': [], 'thickness': []},
        'temporal_right': {'width': [], 'height': [], 'area': [], 'thickness': []},
        'occipital_0': {'width': [], 'height': [], 'area': [], 'thickness': []},
        'occipital_1': {'width': [], 'height': [], 'area': [], 'thickness': []},
    }

    # Per-specimen data for combined analysis
    specimen_data = []

    for data in all_data:
        specimen_id = data.get('specimen_id', 'unknown')
        regions = data.get('regions', {})

        specimen_entry = {'specimen_id': specimen_id, 'apertures': {}}

        for region_name, aperture in regions.items():
            # Normalize region names (handle both formats)
            if region_name in region_data:
                region_data[region_name]['width'].append(aperture['width_mm'])
                region_data[region_name]['height'].append(aperture['height_mm'])
                region_data[region_name]['area'].append(aperture['area_mm2'])
                region_data[region_name]['thickness'].append(aperture['mean_thickness_mm'])

            specimen_entry['apertures'][region_name] = {
                'width_mm': aperture['width_mm'],
                'height_mm': aperture['height_mm'],
                'area_mm2': aperture['area_mm2'],
                'mean_thickness_mm': aperture['mean_thickness_mm']
            }

        specimen_data.append(specimen_entry)

    # Compute statistics for each region
    stats = {}
    for region_name, data in region_data.items():
        stats[region_name] = {
            'width_mm': compute_statistics(data['width']),
            'height_mm': compute_statistics(data['height']),
            'area_mm2': compute_statistics(data['area']),
            'mean_thickness_mm': compute_statistics(data['thickness'])
        }

    # Combined temporal statistics
    temporal_widths = region_data['temporal_left']['width'] + region_data['temporal_right']['width']
    temporal_heights = region_data['temporal_left']['height'] + region_data['temporal_right']['height']
    temporal_areas = region_data['temporal_left']['area'] + region_data['temporal_right']['area']
    temporal_thicknesses = region_data['temporal_left']['thickness'] + region_data['temporal_right']['thickness']

    stats['temporal_combined'] = {
        'width_mm': compute_statistics(temporal_widths),
        'height_mm': compute_statistics(temporal_heights),
        'area_mm2': compute_statistics(temporal_areas),
        'mean_thickness_mm': compute_statistics(temporal_thicknesses)
    }

    # Combined occipital statistics
    occipital_widths = region_data['occipital_0']['width'] + region_data['occipital_1']['width']
    occipital_heights = region_data['occipital_0']['height'] + region_data['occipital_1']['height']
    occipital_areas = region_data['occipital_0']['area'] + region_data['occipital_1']['area']
    occipital_thicknesses = region_data['occipital_0']['thickness'] + region_data['occipital_1']['thickness']

    stats['occipital_combined'] = {
        'width_mm': compute_statistics(occipital_widths),
        'height_mm': compute_statistics(occipital_heights),
        'area_mm2': compute_statistics(occipital_areas),
        'mean_thickness_mm': compute_statistics(occipital_thicknesses)
    }

    return stats, specimen_data


def print_statistics(stats: dict, specimen_data: list[dict]) -> None:
    """Print formatted statistics report."""
    print("\n" + "=" * 80)
    print("APERTURE DIMENSION STATISTICS")
    print("=" * 80)
    print(f"\nNumber of specimens analyzed: {len(specimen_data)}")
    print(f"Specimens: {', '.join([s['specimen_id'] for s in specimen_data])}")

    # Helper to format a stats dict
    def fmt_stat(s: dict, unit: str = "mm") -> str:
        if s['n'] == 0:
            return "No data"
        return (f"{s['mean']:.1f} ± {s['std']:.1f} {unit} "
                f"(range: {s['min']:.1f}-{s['max']:.1f}, median: {s['median']:.1f}, n={s['n']})")

    # Print by region
    region_labels = {
        'temporal_left': 'TEMPORAL LEFT',
        'temporal_right': 'TEMPORAL RIGHT',
        'temporal_combined': 'TEMPORAL (ALL)',
        'occipital_0': 'OCCIPITAL #1 (larger)',
        'occipital_1': 'OCCIPITAL #2 (smaller)',
        'occipital_combined': 'OCCIPITAL (ALL)'
    }

    for region_key in ['temporal_left', 'temporal_right', 'temporal_combined',
                       'occipital_0', 'occipital_1', 'occipital_combined']:
        region_stats = stats.get(region_key, {})
        label = region_labels.get(region_key, region_key)

        print(f"\n{'-' * 80}")
        print(f"{label}")
        print(f"{'-' * 80}")

        width = region_stats.get('width_mm', {})
        height = region_stats.get('height_mm', {})
        area = region_stats.get('area_mm2', {})
        thickness = region_stats.get('mean_thickness_mm', {})

        print(f"  Width:     {fmt_stat(width, 'mm')}")
        print(f"  Height:    {fmt_stat(height, 'mm')}")
        print(f"  Area:      {fmt_stat(area, 'mm²')}")
        print(f"  Thickness: {fmt_stat(thickness, 'mm')}")

    # Print individual specimen table
    print(f"\n{'=' * 80}")
    print("INDIVIDUAL SPECIMEN DATA")
    print("=" * 80)

    # Header
    print(f"\n{'Specimen':<10} | {'Region':<16} | {'Width':>8} | {'Height':>8} | {'Area':>10} | {'Thickness':>10}")
    print("-" * 80)

    for specimen in specimen_data:
        specimen_id = specimen['specimen_id']
        first_row = True

        for region_name in ['temporal_left', 'temporal_right', 'occipital_0', 'occipital_1']:
            aperture = specimen['apertures'].get(region_name)
            if aperture:
                spec_col = specimen_id if first_row else ""
                print(f"{spec_col:<10} | {region_name:<16} | "
                      f"{aperture['width_mm']:>7.1f} | {aperture['height_mm']:>7.1f} | "
                      f"{aperture['area_mm2']:>9.1f} | {aperture['mean_thickness_mm']:>9.1f}")
                first_row = False

        if not first_row:
            print("-" * 80)


def save_statistics(stats: dict, specimen_data: list[dict], output_path: str) -> None:
    """Save statistics to JSON file."""
    output = {
        'n_specimens': len(specimen_data),
        'specimens': [s['specimen_id'] for s in specimen_data],
        'region_statistics': stats,
        'specimen_data': specimen_data
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nStatistics saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute population statistics for aperture dimensions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--results-dir', type=str, default='results',
        help='Results directory containing specimen folders (default: results)'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output JSON file path (default: results/population_statistics.json)'
    )

    args = parser.parse_args()

    if args.output is None:
        args.output = f"{args.results_dir}/population_statistics.json"

    print("Loading aperture data from results folder...")
    all_data = load_all_apertures(args.results_dir)

    if not all_data:
        print("No aperture data found!")
        return

    print(f"Loaded data for {len(all_data)} specimens")

    print("\nComputing statistics...")
    stats, specimen_data = analyze_apertures(all_data)

    print_statistics(stats, specimen_data)
    save_statistics(stats, specimen_data, args.output)


if __name__ == "__main__":
    main()
