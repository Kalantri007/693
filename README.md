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

## Benchmark Results

We conducted extensive benchmarks comparing DFS and BFS algorithms on a social network graph with 300 nodes and 760 edges, testing 50 random users. Here are the key findings:

### Performance

| Metric | DFS | BFS | Difference |
|--------|-----|-----|------------|
| Average Execution Time | 0.002688s | 0.000282s | BFS is 852.37% faster |
| Average Recommendations | 22.96 | 23.66 | BFS provides 3.05% more |
| Recommendation Similarity | - | - | 97.09% average overlap |

### Recommendation Quality

- Recommendation similarity (Jaccard index) ranged from 68.75% to 100%, with an average of 97.09%
- BFS typically provides slightly more recommendations than DFS
- Both algorithms produce highly similar recommendation sets, suggesting they find similar friend candidates

### Visual Comparisons

The visualizations in the `results` directory show:
- BFS consistently outperforms DFS in execution time
- BFS provides more consistent performance with less variance
- Recommendations have high similarity, with most users getting nearly identical suggestions from both algorithms

## Conclusions

1. **BFS is significantly faster** for friend recommendations in social networks, executing over 9 times faster than DFS in our tests
2. **BFS provides slightly more recommendations** on average, offering better coverage of potential friends
3. **Both algorithms produce highly similar results** with over 97% overlap, suggesting the fundamental structure of social network graphs leads to similar friend recommendations regardless of traversal method
4. **BFS offers more predictable performance** with lower variance in execution time

## When to Use Each Algorithm

**Use BFS when:**
- Performance is critical (it's significantly faster)
- You want more comprehensive recommendation sets
- You need consistent, predictable performance
- Social distance is an important factor in recommendations (BFS explores by social distance)

**Use DFS when:**
- Memory constraints are significant (though not measured effectively in our tests)
- You need to explore deep paths in the social network
- Implementation simplicity is preferred (recursive implementation can be simpler)

For social network friend recommendations specifically, **BFS is the recommended approach** as it:
1. Explores the graph by social distance, which aligns with how social relationships form
2. Provides better performance 
3. Delivers slightly more comprehensive recommendation sets
4. Maintains high similarity to DFS results while being faster

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

For a simpler example that doesn't require all dependencies:

```bash
python example.py
```

For comprehensive benchmarking:

```bash
python enhanced_benchmark.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Stanford Network Analysis Project (SNAP) for the Facebook dataset
- NetworkX library for graph data structures and algorithms