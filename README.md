# TVML: Time-Vertex Machine Learning for Sensor Placement

**Reference implementation of Time-Vertex Machine Learning (TVML) for selecting informative and nonredundant sensors in graph-structured time series.**


**Paper:** [Time-Vertex Machine Learning for Optimal Sensor Placement in Temporal Graph Signals: Applications in Structural Health Monitoring](https://doi.org/10.1016/j.ress.2025.112153)  
**Journal:** *Reliability Engineering & System Safety*, vol. 270, article 112153 (2026)  
**Authors:** Keivan Faghih Niresi, Jun Qing, Mengjie Zhao, and Olga Fink  
**Preprint:** [arXiv:2512.19309](https://arxiv.org/abs/2512.19309)

## Overview

Choosing sensor locations is important for structural health monitoring: redundant measurements increase costs, while poorly selected locations may miss useful structural information. TVML combines **time-series features** with **graph topology** to select representative sensors without requiring labels for the sensor-selection stage.

The selection pipeline has three steps:

1. **Extract temporal and spectral features:** For every sensor, calculate statistical descriptors (such as variance, skewness, and RMS) and FFT-based descriptors (including dominant frequency and low-frequency magnitudes).
2. **Group similar sensors:** Apply K-means to the feature vectors, using the desired number of sensors as the number of clusters.
3. **Select graph-representative sensors:** Compute a weighted topology score from degree, betweenness, eigenvector and harmonic centrality, and a graph spectral-localization feature; select the highest-scoring sensor from each cluster.

A simplified expression for the node score is

$$
\psi_i = \sum_{j=1}^{5} \alpha_j \widetilde{c}_{i,j},
$$

where $\widetilde{c}_{i,j}$ is a graph-topology descriptor, optionally standardized across nodes, and $\alpha_j$ controls its contribution. The default uses equal weights and feature standardization.

## Repository contents

```text
TVML/
├── README.md              # Project overview, installation, and usage
├── tvml.py                # TVML sensor selection implementation
├── TVML_Demo.ipynb        # Executable example with synthetic sensor data
├── requirements.txt      # Runtime and visualization dependencies
├── CITATION.cff           # Citation metadata for GitHub
├── .gitignore
└── tests/
    └── test_tvml.py      # Basic correctness and reproducibility tests
```

**Scope:** This release provides the TVML **sensor-selection method** and a self-contained synthetic demonstration.

## Installation

Python 3.10 or newer is recommended.

```bash
# Clone after the repository has been created
git clone https://github.com/EPFL-IMOS/TVML.git
cd TVML

python -m pip install -r requirements.txt
```

To run the Jupyter notebook, also install a notebook environment:

```bash
python -m pip install jupyterlab
jupyter lab TVML_Demo.ipynb
```

## Quick start

```python
import numpy as np
import networkx as nx
from tvml import TVML

# Each column contains the time series of a graph node.
rng = np.random.default_rng(42)
T, N = 500, 20
t = np.linspace(0, 10, T)
data = np.column_stack([
    np.sin(2 * np.pi * (0.5 + 0.1 * (i % 5)) * t + 0.2 * i)
    + 0.15 * rng.standard_normal(T)
    for i in range(N)
])

# Replace this example graph with the sensor graph for your application.
graph = nx.watts_strogatz_graph(N, k=4, p=0.2, seed=42)

selected_nodes, selected_columns = TVML(
    data=data,
    graph=graph,
    num_sensors=5,
    random_state=42,
    return_indices=True,
)

print("Selected node labels:", selected_nodes)
print("Selected data columns:", selected_columns)
```

For a visual example with a weighted graph, run [`TVML_Demo.ipynb`](TVML_Demo.ipynb).

## Inputs and configuration

| Parameter | Description | Default |
| --- | --- | --- |
| `data` | NumPy array of shape `(T, N)`: time samples × sensors | Required |
| `graph` | Undirected `networkx.Graph` with `N` sensor nodes | Required |
| `num_sensors` | Number of sensors to select (also the number of K-means clusters) | Required |
| `alpha` | Weights for degree, betweenness, eigenvector, harmonic, and localization features | `(0.2, 0.2, 0.2, 0.2, 0.2)` |
| `random_state` | K-means random seed | `42` |
| `nu` | Graph localization-kernel decay | `10.0` |
| `normalization_topofeature` | Standardize each topology feature across nodes | `True` |
| `return_indices` | Additionally return zero-based column indices | `False` |
| `betweenness_weight` | Edge attribute used as shortest-path cost in betweenness centrality | `"weight"` |

### Data and graph conventions

- Columns of `data` **must follow the insertion order of `graph.nodes()`**. Returned node labels use that same graph ordering; returned indices refer to the input data columns.
- Sensor signals must be finite. The graph must be undirected, with finite, strictly positive edge weights when weights are provided. Missing edge weights are treated as 1.
- The implementation uses min-max scaling across sensors for temporal/spectral features and, by default, standardization across nodes for topology features.

**Weighted-path convention:** The supplied reference implementation calls NetworkX betweenness centrality using the edge attribute `weight`, which NetworkX interprets as a **path length**. That convention is preserved here by default. If your edge weights instead represent **connection strengths** (larger weight means a stronger/shorter connection), use `betweenness_weight="distance"` to compute betweenness with inverse weights (`1 / weight`), consistent with the harmonic-centrality distance convention. This option can change the selected sensors.

```python
selected = TVML(data, graph, num_sensors=5, betweenness_weight="distance")
```

## Testing

Install `pytest` and run:

```bash
python -m pip install pytest
python -m pytest -q
```

## Citation

If you use this code or method in your research, please cite:

```bibtex
@article{faghihniresi2026tvml,
  title   = {Time-Vertex Machine Learning for Optimal Sensor Placement in Temporal Graph Signals: Applications in Structural Health Monitoring},
  author  = {Faghih Niresi, Keivan and Qing, Jun and Zhao, Mengjie and Fink, Olga},
  journal = {Reliability Engineering \& System Safety},
  volume  = {270},
  pages   = {112153},
  year    = {2026},
  doi     = {10.1016/j.ress.2025.112153}
}
```
