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

### Synthetic Network

We conducted benchmarks comparing DFS and BFS algorithms on a synthetic social network graph with 300 nodes and 760 edges, testing 50 random users. Here are the key findings:

| Metric | DFS | BFS | Difference |
|--------|-----|-----|------------|
| Average Execution Time | 0.002688s | 0.000282s | BFS is 852.37% faster |
| Average Recommendations | 22.96 | 23.66 | BFS provides 3.05% more |
| Recommendation Similarity | - | - | 97.09% average overlap |

### Real Facebook Network (SNAP Dataset)

We extended our analysis to use the real Stanford SNAP Facebook dataset, which contains 4,039 nodes (users) and 88,234 edges (friendships). This provided insights into how both algorithms perform in a real-world social network:

| Metric | DFS | BFS | Difference |
|--------|-----|-----|------------|
| Average Execution Time | 0.002656s | 0.002295s | BFS is 15.74% faster |
| Average Recommendations | 635.72 | 669.48 | BFS provides 5.31% more |
| Recommendation Similarity | - | - | 94.07% average overlap |
| Min Similarity | - | - | 20.98% |
| Max Similarity | - | - | 100.00% |

### Recommendation Quality

- In both synthetic and real networks, recommendation similarity (Jaccard index) is high, indicating that both algorithms find similar friend candidates
- In the real Facebook dataset, there's greater variability in recommendation similarity (±17.64%), with some users receiving quite different recommendation sets
- BFS consistently provides more recommendations in both synthetic and real networks
- The real Facebook dataset produces significantly more recommendations per user (~636-669) compared to our synthetic model (~23), reflecting the denser connectivity in real social networks

### Visual Comparisons

The visualizations in the `results` directory show:
- BFS consistently outperforms DFS in execution time across both synthetic and real datasets
- The performance advantage of BFS is much less pronounced in the real Facebook network (15.74% vs 852.37%)
- Both algorithms show higher variance in execution times on the real dataset, indicating more complex traversal patterns
- Recommendations have high similarity overall, but with some significant outliers in the real network

## Conclusions

1. **BFS is consistently faster** for friend recommendations in social networks, though the advantage is less dramatic in real-world networks
2. **BFS provides more recommendations** on average, offering better coverage of potential friends in both synthetic and real networks
3. **Both algorithms produce highly similar results** with over 94% overlap in real social networks, suggesting the fundamental structure of social networks leads to similar friend recommendations regardless of traversal method
4. **Real social networks exhibit different traversal characteristics** than synthetic models, with much denser recommendation sets and more variability in algorithm performance

## When to Use Each Algorithm

**Use BFS when:**
- Performance is critical (it's consistently faster)
- You want more comprehensive recommendation sets
- You need consistent, predictable performance
- Social distance is an important factor in recommendations (BFS explores by social distance)

**Use DFS when:**
- Memory constraints are significant
- You need to explore deep paths in the social network
- Implementation simplicity is preferred (recursive implementation can be simpler)

For social network friend recommendations specifically, **BFS remains the recommended approach** as it:
1. Explores the graph by social distance, which aligns with how social relationships form
2. Provides better performance across both synthetic and real networks
3. Delivers slightly more comprehensive recommendation sets
4. Maintains high similarity to DFS results while being faster

## Differences Between Synthetic and Real Networks

Our study revealed important differences between synthetic network models and real social networks:
1. **Performance gap**: The performance advantage of BFS is much less pronounced in real networks (15.74% vs 852.37%)
2. **Recommendation volume**: Real networks produce far more recommendations per user (~650 vs ~23)
3. **Recommendation variability**: Real networks show more variability in recommendation similarity between algorithms
4. **Network density**: Real social networks are much denser, with higher connectivity between users

## Dataset

The project uses the Facebook social network dataset from SNAP, which contains anonymized connections between Facebook users. The dataset is automatically downloaded when running the project.

- Nodes: 4,039 users
- Edges: 88,234 friendships

## Project Structure

```
.
├── data/                  # Data files
│   └── facebook_combined.txt.gz  # SNAP Facebook dataset
├── results/               # Results and visualizations
│   ├── synthetic/              # Results from synthetic network
│   └── facebook/               # Results from real Facebook network
├── src/                   # Source code
│   ├── data_acquisition.py     # Script to download and process the dataset
│   ├── recommendation_algorithms.py  # DFS and BFS recommendation implementations
│   ├── benchmark.py            # Benchmarking and comparison code
│   └── main.py                 # Main script to run the experiment
├── tests/                 # Test files
│   └── test_recommendation_algorithms.py  # Unit tests for algorithms
├── enhanced_benchmark.py       # Enhanced benchmark script with visualization
├── facebook_benchmark.py       # Benchmark script for real Facebook dataset
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

For comprehensive benchmarking on synthetic network:

```bash
python enhanced_benchmark.py
```

For benchmarking on the real Facebook dataset:

```bash
python facebook_benchmark.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Stanford Network Analysis Project (SNAP) for the Facebook dataset
- NetworkX library for graph data structures and algorithms