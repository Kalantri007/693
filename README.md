# DFS vs BFS for Friend Recommendation Systems

This research project compares the performance and accuracy of Depth-First Search (DFS) and Breadth-First Search (BFS) algorithms for friend recommendation systems in social networks.

## Project Overview

In social networking platforms, friend recommendations are a critical feature for user engagement and network growth. This study implements and compares two fundamental graph traversal algorithms, DFS and BFS, for generating friend recommendations based on the Facebook social network dataset from SNAP (Stanford Network Analysis Project).

## Features

- Implementation of DFS and BFS algorithms for friend recommendations
- Performance benchmarking (execution time and memory usage)
- Quality assessment of recommendations
- Comprehensive visualization of results
- Automated testing suite

## Dataset

The project uses the Facebook social network dataset from SNAP, which contains anonymized connections between Facebook users. The dataset is automatically downloaded when running the project.

## Project Structure

```
.
├── data/                  # Data files
│   └── facebook_graph.gpickle  # Processed graph data (generated)
├── results/               # Results and visualizations
├── src/                   # Source code
│   ├── data_acquisition.py     # Script to download and process the dataset
│   ├── recommendation_algorithms.py  # DFS and BFS recommendation implementations
│   ├── benchmark.py            # Benchmarking and comparison code
│   └── main.py                 # Main script to run the experiment
├── tests/                 # Test files
│   └── test_recommendation_algorithms.py  # Unit tests for algorithms
└── requirements.txt       # Project dependencies
```

## Installation

Clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd dfs-bfs-recommendations
pip install -r requirements.txt
```

## Usage

Run the main experiment:

```bash
cd src
python main.py
```

### Command-line Arguments

- `--download`: Force re-download of the dataset
- `--num-users <N>`: Number of random users to test (default: 50)
- `--top-n <N>`: Number of recommendations to generate per user (default: 10)
- `--max-depth <N>`: Maximum depth for graph traversal (default: 2)
- `--output-dir <DIR>`: Directory to save results (default: ../results)

Example:

```bash
python main.py --num-users 100 --top-n 5 --max-depth 3
```

## Running Tests

```bash
cd tests
python -m unittest test_recommendation_algorithms.py
```

## Results and Analysis

After running the experiment, the results will be saved in the `results/` directory:

- `benchmark_results.csv`: Raw benchmark data for both algorithms
- `performance_comparison.png`: Visual comparison of execution time and memory usage
- `recommendation_similarity.png`: Histogram of recommendation similarities (Jaccard index)

A summary of the results will also be displayed in the console.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Stanford Network Analysis Project (SNAP) for the Facebook dataset
- NetworkX library for graph data structures and algorithms