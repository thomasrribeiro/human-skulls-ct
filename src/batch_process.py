"""
Batch Processing Script for Skull Thickness and Aperture Analysis

This script processes all specimens in the data directory, computing:
1. Skull thickness maps (cached to avoid recomputation)
2. Optimal aperture dimensions for temporal and occipital regions
3. Population statistics across all specimens

Usage:
    # Process all specimens
    python batch_process.py

    # Force recompute all (ignore cache)
    python batch_process.py --force

    # Process specific specimen only
    python batch_process.py --specimen A0001

    # Force recompute specific specimen
    python batch_process.py --specimen A0001 --force
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import run_thickness_pipeline
from src.aperture_analysis import (
    segment_regions, analyze_all_regions,
    cache_exists, load_thickness_cache, save_thickness_cache,
    save_aperture_results, load_aperture_results,
    compute_population_statistics, save_population_statistics
)


def discover_specimens(data_dir: str = "data") -> list[tuple[str, str]]:
    """
    Find all NRRD specimen files in the data directory.

    Args:
        data_dir: Base data directory

    Returns:
        List of (specimen_id, nrrd_path) tuples, sorted by specimen_id
    """
    data_path = Path(data_dir)
    specimens = []

    # Search recursively for .nrrd files
    for nrrd_file in data_path.rglob("*.nrrd"):
        specimen_id = nrrd_file.stem  # e.g., "A0001"
        specimens.append((specimen_id, str(nrrd_file)))

    # Sort by specimen ID
    specimens.sort(key=lambda x: x[0])

    print(f"Found {len(specimens)} specimens in {data_dir}")
    return specimens


def process_single_specimen(specimen_id: str, nrrd_path: str,
                            results_dir: str = "results",
                            percentile: float = 25,
                            n_rays: int = 20000,
                            force_recompute: bool = False) -> dict:
    """
    Process a single specimen: thickness analysis + aperture computation.

    Args:
        specimen_id: Specimen identifier (e.g., "A0001")
        nrrd_path: Path to NRRD file
        results_dir: Base results directory
        percentile: Percentile threshold for thin region
        n_rays: Number of rays for thickness computation
        force_recompute: If True, recompute even if cached

    Returns:
        Aperture results dictionary
    """
    print(f"\n{'='*60}")
    print(f"Processing specimen: {specimen_id}")
    print(f"{'='*60}")

    # Check for cached thickness data
    if not force_recompute and cache_exists(specimen_id, results_dir):
        print("Loading cached thickness data...")
        cached = load_thickness_cache(specimen_id, results_dir)
        verts = cached['vertices']
        faces = cached['faces']
        vertex_thicknesses = cached['thicknesses']
        center = cached['center']
    else:
        print("Computing thickness map (this may take a few minutes)...")
        result = run_thickness_pipeline(nrrd_path, n_rays=n_rays)
        verts = result['verts']
        faces = result['faces']
        vertex_thicknesses = result['thicknesses']
        center = result['center']

        # Save to cache
        print("Saving thickness data to cache...")
        save_thickness_cache(specimen_id, verts, faces, vertex_thicknesses, center, results_dir)

    # Regional aperture analysis
    print("\nAnalyzing regional apertures...")
    print("Segmenting surface into anatomical regions...")
    regions = segment_regions(verts, center)

    print(f"\nComputing optimal apertures (thinnest {percentile}% per region)...")
    result = analyze_all_regions(verts, vertex_thicknesses, regions, percentile, cluster_eps=5.0)
    apertures = result['apertures']
    region_stats = result['region_stats']

    for ap_name, ap in apertures.items():
        print(f"  {ap_name}: {ap['width_mm']:.1f} x {ap['height_mm']:.1f} mm "
              f"(area: {ap['area_mm2']:.0f} mm², thickness: {ap['mean_thickness_mm']:.1f} mm)")

    # Save aperture results
    save_aperture_results(specimen_id, apertures, percentile, results_dir)

    return {
        'specimen_id': specimen_id,
        'percentile_threshold': percentile,
        'regions': apertures
    }


def process_all_specimens(data_dir: str = "data",
                          results_dir: str = "results",
                          percentile: float = 25,
                          n_rays: int = 20000,
                          force_recompute: bool = False) -> list[dict]:
    """
    Process all specimens in the data directory.

    Args:
        data_dir: Base data directory
        results_dir: Base results directory
        percentile: Percentile threshold for thin region
        n_rays: Number of rays for thickness computation
        force_recompute: If True, recompute all specimens

    Returns:
        List of aperture results for all specimens
    """
    specimens = discover_specimens(data_dir)

    if not specimens:
        print("No specimens found!")
        return []

    all_results = []
    total = len(specimens)
    start_time = datetime.now()

    for i, (specimen_id, nrrd_path) in enumerate(specimens, 1):
        print(f"\n[{i}/{total}] Processing {specimen_id}...")

        try:
            result = process_single_specimen(
                specimen_id, nrrd_path,
                results_dir=results_dir,
                percentile=percentile,
                n_rays=n_rays,
                force_recompute=force_recompute
            )
            all_results.append(result)
        except Exception as e:
            print(f"ERROR processing {specimen_id}: {e}")
            continue

        # Progress estimate
        elapsed = (datetime.now() - start_time).total_seconds()
        avg_time = elapsed / i
        remaining = avg_time * (total - i)
        print(f"Progress: {i}/{total} ({100*i/total:.0f}%) - "
              f"Est. remaining: {remaining/60:.1f} min")

    total_time = (datetime.now() - start_time).total_seconds()
    print(f"\n{'='*60}")
    print(f"Batch processing complete!")
    print(f"Processed {len(all_results)}/{total} specimens in {total_time/60:.1f} minutes")
    print(f"{'='*60}")

    return all_results


def load_all_results(results_dir: str = "results") -> list[dict]:
    """
    Load all existing aperture results from the results directory.

    Args:
        results_dir: Base results directory

    Returns:
        List of aperture results dictionaries
    """
    results_path = Path(results_dir)
    all_results = []

    for specimen_dir in sorted(results_path.iterdir()):
        if specimen_dir.is_dir():
            result = load_aperture_results(specimen_dir.name, results_dir)
            if result is not None:
                all_results.append(result)

    print(f"Loaded {len(all_results)} existing results from {results_dir}")
    return all_results


def print_population_summary(stats: dict) -> None:
    """Print a formatted summary of population statistics."""
    print(f"\n{'='*60}")
    print("TRANSDUCER APERTURE STATISTICS")
    print(f"{'='*60}")
    print(f"Number of specimens: {stats['n_specimens']}")

    n_measurements = stats.get('n_measurements_per_transducer', {})
    print(f"Measurements per transducer: temporal={n_measurements.get('temporal', 0)}, "
          f"occipital={n_measurements.get('occipital', 0)}")

    for transducer_name, transducer_stats in stats.get('transducers', {}).items():
        print(f"\n{transducer_name.upper()} TRANSDUCER:")
        print("-" * 50)

        width = transducer_stats.get('width_mm', {})
        height = transducer_stats.get('height_mm', {})
        area = transducer_stats.get('area_mm', {})
        thickness = transducer_stats.get('mean_thickness_mm', {})

        print(f"  Aperture Width:  {width.get('mean', 0):.1f} ± {width.get('std', 0):.1f} mm "
              f"(range: {width.get('min', 0):.1f} - {width.get('max', 0):.1f})")
        print(f"                   median: {width.get('p50', 0):.1f} mm, "
              f"IQR: [{width.get('p25', 0):.1f} - {width.get('p75', 0):.1f}]")
        print(f"  Aperture Height: {height.get('mean', 0):.1f} ± {height.get('std', 0):.1f} mm "
              f"(range: {height.get('min', 0):.1f} - {height.get('max', 0):.1f})")
        print(f"                   median: {height.get('p50', 0):.1f} mm, "
              f"IQR: [{height.get('p25', 0):.1f} - {height.get('p75', 0):.1f}]")
        print(f"  Aperture Area:   {area.get('mean', 0):.0f} ± {area.get('std', 0):.0f} mm² "
              f"(range: {area.get('min', 0):.0f} - {area.get('max', 0):.0f})")
        print(f"  Mean Thickness:  {thickness.get('mean', 0):.1f} ± {thickness.get('std', 0):.1f} mm "
              f"(range: {thickness.get('min', 0):.1f} - {thickness.get('max', 0):.1f})")


def main():
    """Main entry point for batch processing."""
    parser = argparse.ArgumentParser(
        description="Batch process skull thickness and aperture analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--data-dir', type=str, default='data',
        help='Data directory containing specimen NRRD files (default: data)'
    )
    parser.add_argument(
        '--results-dir', type=str, default='results',
        help='Results directory for output files (default: results)'
    )
    parser.add_argument(
        '--specimen', type=str, default=None,
        help='Process only this specimen ID (e.g., A0001)'
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Force recompute even if cached data exists'
    )
    parser.add_argument(
        '--percentile', type=float, default=25,
        help='Percentile threshold for thin region (default: 25)'
    )
    parser.add_argument(
        '--n-rays', type=int, default=20000,
        help='Number of rays for thickness computation (default: 20000)'
    )
    parser.add_argument(
        '--stats-only', action='store_true',
        help='Only compute population statistics from existing results'
    )

    args = parser.parse_args()

    # Stats only mode - just load and aggregate
    if args.stats_only:
        print("Loading existing results and computing statistics...")
        all_results = load_all_results(args.results_dir)
        if all_results:
            stats = compute_population_statistics(all_results)
            save_population_statistics(stats, f"{args.results_dir}/population_summary.json")
            print_population_summary(stats)
        else:
            print("No results found to aggregate!")
        return

    # Process single specimen
    if args.specimen:
        specimens = discover_specimens(args.data_dir)
        specimen_match = [(sid, path) for sid, path in specimens if sid == args.specimen]

        if not specimen_match:
            print(f"Specimen {args.specimen} not found in {args.data_dir}")
            sys.exit(1)

        specimen_id, nrrd_path = specimen_match[0]
        process_single_specimen(
            specimen_id, nrrd_path,
            results_dir=args.results_dir,
            percentile=args.percentile,
            n_rays=args.n_rays,
            force_recompute=args.force
        )
        return

    # Process all specimens
    all_results = process_all_specimens(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        percentile=args.percentile,
        n_rays=args.n_rays,
        force_recompute=args.force
    )

    # Compute and save population statistics
    if all_results:
        print("\nComputing population statistics...")
        stats = compute_population_statistics(all_results)
        save_population_statistics(stats, f"{args.results_dir}/population_summary.json")
        print_population_summary(stats)


if __name__ == "__main__":
    main()
