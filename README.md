# Human Skulls CT - Skull Thickness Analysis

Automated skull thickness measurement and transducer aperture analysis from CT data for transcranial ultrasound applications.

## Overview

This project analyzes skull CT scans to:
1. Compute per-vertex skull thickness using surface-normal ray tracing
2. Identify optimal rectangular apertures for ultrasound transducer placement
3. Generate population statistics for transducer design

## Aperture Analysis Algorithm

The algorithm finds the largest rectangular transducer footprint over thin bone regions:

1. **Clustering**: DBSCAN groups thin vertices (≤25th percentile thickness) into spatially contiguous patches (eps=5mm)

2. **Adjacency graph**: Connect clusters within 15mm of each other - defines which clusters can merge into a single transducer footprint

3. **Exhaustive search**: Try all subsets of adjacent clusters. Each combination must have ≥80% thin vertices to be valid.

4. **Rectangle fitting**: Project merged cluster to 2D via PCA → create occupancy grid → find largest inscribed rectangle → return biggest among all valid combinations

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install uv
uv pip install numpy pynrrd scipy scikit-image scikit-learn pyvista matplotlib
```

## Usage

### Process a single specimen
```bash
source .venv/bin/activate && uv run python src/batch_process.py --specimen A0001
```

### Process all specimens
```bash
source .venv/bin/activate && uv run python src/batch_process.py
```

### Visualize results
```bash
source .venv/bin/activate && uv run python src/visualize_specimen.py A0001
```

### Recompute statistics only
```bash
source .venv/bin/activate && uv run python src/batch_process.py --stats-only
```

## Output

Results are saved per specimen in `results/{specimen_id}/`:
- `thickness_data.npz`: Cached mesh vertices, faces, and thickness values
- `apertures.json`: Aperture dimensions and 3D coordinates for each region

Population statistics are aggregated in `results/population_summary.json`.

## Sample Results (n=20 skulls)

| Transducer | Width (mm) | Height (mm) | Bone Thickness (mm) |
|------------|------------|-------------|---------------------|
| Temporal   | 48 ± 16    | 25 ± 10     | 3.9 ± 1.2           |
| Occipital  | 38 ± 14    | 19 ± 7      | 5.7 ± 1.1           |

## Project Structure

```
human-skulls-ct/
├── main.py                      # Thickness computation pipeline
├── src/
│   ├── aperture_analysis.py     # Aperture detection algorithm
│   ├── batch_process.py         # Batch processing
│   └── visualize_specimen.py    # 3D visualization
├── data/                        # Input NRRD files
└── results/                     # Output data
```

## Data Source

Skull CT data obtained from the open-access dataset:
- Pinheiro et al. (2021). "An open-access dataset of human skulls from computed tomography scans." *Data in Brief*, 37, 107200. https://doi.org/10.1016/j.dib.2021.107200

## References

Thickness measurement methodology based on:
- Attali et al. 2023 - Ray-tracing approach for skull thickness estimation

## License

MIT License
