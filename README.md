[![MIT](https://custom-icon-badges.herokuapp.com/badge/license-MIT-8BB80A.svg?logo=law&logoColor=white)]()
[![Linux](https://custom-icon-badges.herokuapp.com/badge/Linux-F6CE18.svg?logo=Linux&logoColor=white)]()
[![Python](https://custom-icon-badges.herokuapp.com/badge/Python-3572A5.svg?logo=Python&logoColor=white)]()
<img src="https://img.shields.io/badge/-Ubuntu-6F52B5.svg?logo=ubuntu&style=flat">

# BERTopic Sample

Japanese document topic modeling and 2D visualization with BERTopic.
Uses SudachiPy for morphological analysis and UMAP for dimensionality reduction to cluster and plot Japanese text documents by topic.

## Prerequisites

- Python 3.12 or later
- [uv](https://docs.astral.sh/uv/) (Python package manager)

## Installation

```bash
git clone https://github.com/R-Tatara/bertopic-sample.git
cd bertopic-sample
uv sync
```

## Usage

```bash
uv run python main.py
```

- Detects topics from a set of Japanese documents using BERTopic
- Prints topic assignments to the console
- Displays a 2D scatter plot colored by topic

### Configuration

Key constants in `main.py`:

| Constant | Default | Description |
|---|---|---|
| `NUM_TOPICS` | `5` | Number of topics to extract |
| `HDBSCAN_MIN_CLUSTER_SIZE` | `5` | Minimum cluster size for HDBSCAN |
| `UMAP_N_NEIGHBORS` | `5` | Number of neighbors for UMAP |

## LICENSE

MIT
