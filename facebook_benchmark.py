#!/usr/bin/env python3

"""
Benchmark script to compare DFS and BFS on the real SNAP Facebook dataset
"""

import os
import time
import random
import gzip
import urllib.request
from collections import defaultdict, deque
import csv
import numpy as np
from tqdm import tqdm

# Create results directory if it doesn't exist
os.makedirs('results/facebook', exist_ok=True)

class SimpleGraph:
    """A simple graph implementation that doesn't rely on NetworkX"""
    
    def __init__(self):
        """Initialize an empty graph"""
        self.adjacency_list = defaultdict(list)
    
    def add_edge(self, u, v):
        """Add an edge between vertices u and v"""
        self.adjacency_list[u].append(v)
        self.adjacency_list[v].append(u)
    
    def neighbors(self, vertex):
        """Return the neighbors of a vertex"""
        return self.adjacency_list[vertex]
    
    def nodes(self):
        """Return all nodes in the graph"""
        return list(self.adjacency_list.keys())
    
    def number_of_nodes(self):
        """Return the number of nodes in the graph"""
        return len(self.adjacency_list)
    
    def number_of_edges(self):
        """Return the number of edges in the graph"""
        return sum(len(neighbors) for neighbors in self.adjacency_list.values()) // 2

class DownloadProgressBar(tqdm):
    """Progress bar for downloads"""
    def update_to(self, block_num, block_size, total_size):
        if total_size is not None:
            self.total = total_size
        self.update(block_size)

def download_facebook_dataset(output_dir="data"):
    """
    Download the Facebook Social Network dataset from SNAP
    Args:
        output_dir: Directory to save the dataset
    Returns:
        Path to the dataset
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # URL for the Facebook dataset
    url = "https://snap.stanford.edu/data/facebook_combined.txt.gz"
    output_file = os.path.join(output_dir, "facebook_combined.txt.gz")
    
    # Download the dataset
    if not os.path.exists(output_file):
        print(f"Downloading Facebook dataset from {url}...")
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="Facebook Dataset") as t:
            urllib.request.urlretrieve(url, output_file, reporthook=t.update_to)
    else:
        print(f"Dataset already exists at {output_file}")
    
    return output_file

def load_facebook_graph(file_path):
    """
    Load the Facebook dataset as a graph
    Args:
        file_path: Path to the dataset file
    Returns:
        SimpleGraph instance
    """
    print("Loading Facebook graph...")
    graph = SimpleGraph()
    
    # Open and read the gzipped file
    with gzip.open(file_path, 'rt') as f:
        for line in f:
            # Skip comment lines
            if line.startswith('#'):
                continue
                
            # Parse the edge
            source, target = map(int, line.strip().split())
            graph.add_edge(source, target)
    
    print(f"Graph loaded with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    return graph

def measure_memory_usage():
    """
    Measure current memory usage
    Returns:
        Memory usage in MB
    """
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / 1024 / 1024  # Convert to MB
    except ImportError:
        # If psutil is not available, return a placeholder value
        return 0

def dfs_recommendations(graph, user_id, max_depth=2):
    """
    Get friend recommendations using Depth-First Search
    
    Args:
        graph: The social network graph
        user_id: The user to get recommendations for
        max_depth: Maximum depth for search
    
    Returns:
        Tuple of (recommendations, execution_time, memory_usage)
    """
    start_time = time.time()
    start_memory = measure_memory_usage()
    
    # Track candidates and their frequency
    candidates = defaultdict(int)
    visited = {user_id}  # Start with user as visited
    
    def dfs(node, current_depth):
        if current_depth > max_depth:
            return
        
        for neighbor in graph.neighbors(node):
            if current_depth == max_depth and neighbor not in visited:
                # Found a potential recommendation at desired depth
                candidates[neighbor] += 1
            
            if neighbor not in visited:
                visited.add(neighbor)
                dfs(neighbor, current_depth + 1)
    
    # Start DFS from the user
    dfs(user_id, 1)
    
    # Remove the user and their direct friends
    direct_friends = set(graph.neighbors(user_id))
    filtered_candidates = [c for c in candidates.keys() 
                          if c != user_id and c not in direct_friends]
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    end_memory = measure_memory_usage()
    memory_usage = end_memory - start_memory
    
    return filtered_candidates, execution_time, memory_usage

def bfs_recommendations(graph, user_id, max_depth=2):
    """
    Get friend recommendations using Breadth-First Search
    
    Args:
        graph: The social network graph
        user_id: The user to get recommendations for
        max_depth: Maximum depth for search
    
    Returns:
        Tuple of (recommendations, execution_time, memory_usage)
    """
    start_time = time.time()
    start_memory = measure_memory_usage()
    
    # Track candidates and their frequency
    candidates = defaultdict(int)
    visited = {user_id}
    queue = deque([(user_id, 0)])  # (node, depth)
    
    while queue:
        node, depth = queue.popleft()
        
        if depth >= max_depth:
            continue
            
        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))
                
                if depth == max_depth - 1:
                    # This is a friend-of-friend
                    candidates[neighbor] += 1
    
    # Remove the user and their direct friends
    direct_friends = set(graph.neighbors(user_id))
    filtered_candidates = [c for c in candidates.keys() 
                          if c != user_id and c not in direct_friends]
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    end_memory = measure_memory_usage()
    memory_usage = end_memory - start_memory
    
    return filtered_candidates, execution_time, memory_usage

def calculate_jaccard_similarity(set1, set2):
    """Calculate Jaccard similarity between two sets"""
    intersection = len(set(set1).intersection(set(set2)))
    union = len(set(set1).union(set(set2)))
    return intersection / union if union > 0 else 0

def run_benchmark(graph, num_users=100, max_depth=2):
    """
    Run benchmarks comparing DFS and BFS
    
    Args:
        graph: The social network graph
        num_users: Number of users to test
        max_depth: Maximum depth for search
    
    Returns:
        Dictionary with benchmark results
    """
    # Select random users for testing
    all_users = graph.nodes()
    if len(all_users) < num_users:
        num_users = len(all_users)
        
    test_users = random.sample(all_users, num_users)
    
    results = {
        'users': test_users,
        'dfs': {
            'times': [],
            'memory': [],
            'recommendation_counts': []
        },
        'bfs': {
            'times': [],
            'memory': [],
            'recommendation_counts': []
        },
        'similarity': []
    }
    
    print(f"Running benchmark on {num_users} users from the real Facebook dataset...")
    
    for i, user_id in enumerate(test_users):
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{num_users} users")
            
        # Get DFS recommendations
        dfs_recs, dfs_time, dfs_memory = dfs_recommendations(graph, user_id, max_depth)
        results['dfs']['times'].append(dfs_time)
        results['dfs']['memory'].append(dfs_memory)
        results['dfs']['recommendation_counts'].append(len(dfs_recs))
        
        # Get BFS recommendations
        bfs_recs, bfs_time, bfs_memory = bfs_recommendations(graph, user_id, max_depth)
        results['bfs']['times'].append(bfs_time)
        results['bfs']['memory'].append(bfs_memory)
        results['bfs']['recommendation_counts'].append(len(bfs_recs))
        
        # Calculate similarity
        similarity = calculate_jaccard_similarity(dfs_recs, bfs_recs)
        results['similarity'].append(similarity)
    
    # Calculate summary statistics
    results['summary'] = {
        'dfs': {
            'avg_time': np.mean(results['dfs']['times']),
            'std_time': np.std(results['dfs']['times']),
            'avg_memory': np.mean(results['dfs']['memory']),
            'std_memory': np.std(results['dfs']['memory']),
            'avg_recommendations': np.mean(results['dfs']['recommendation_counts']),
            'std_recommendations': np.std(results['dfs']['recommendation_counts'])
        },
        'bfs': {
            'avg_time': np.mean(results['bfs']['times']),
            'std_time': np.std(results['bfs']['times']),
            'avg_memory': np.mean(results['bfs']['memory']),
            'std_memory': np.std(results['bfs']['memory']),
            'avg_recommendations': np.mean(results['bfs']['recommendation_counts']),
            'std_recommendations': np.std(results['bfs']['recommendation_counts'])
        },
        'similarity': {
            'avg': np.mean(results['similarity']),
            'std': np.std(results['similarity']),
            'min': np.min(results['similarity']) if results['similarity'] else 0,
            'max': np.max(results['similarity']) if results['similarity'] else 0
        }
    }
    
    return results

def save_results_csv(results, filename="results/facebook/benchmark_results.csv"):
    """Save benchmark results to a CSV file"""
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['user_id', 'algorithm', 'execution_time', 'memory_usage', 'recommendation_count', 'similarity']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for i, user_id in enumerate(results['users']):
            # Write DFS results
            writer.writerow({
                'user_id': user_id,
                'algorithm': 'DFS',
                'execution_time': results['dfs']['times'][i],
                'memory_usage': results['dfs']['memory'][i],
                'recommendation_count': results['dfs']['recommendation_counts'][i],
                'similarity': results['similarity'][i]
            })
            
            # Write BFS results
            writer.writerow({
                'user_id': user_id,
                'algorithm': 'BFS',
                'execution_time': results['bfs']['times'][i],
                'memory_usage': results['bfs']['memory'][i],
                'recommendation_count': results['bfs']['recommendation_counts'][i],
                'similarity': results['similarity'][i]
            })
    
    print(f"Results saved to {filename}")

def save_summary_txt(results, filename="results/facebook/benchmark_summary.txt"):
    """Save benchmark summary to a text file"""
    with open(filename, 'w') as f:
        f.write("===== DFS vs BFS Benchmark Summary (Real Facebook Dataset) =====\n\n")
        
        # DFS summary
        f.write("DFS Algorithm:\n")
        f.write(f"  Average Execution Time: {results['summary']['dfs']['avg_time']:.6f} seconds (±{results['summary']['dfs']['std_time']:.6f})\n")
        f.write(f"  Average Memory Usage: {results['summary']['dfs']['avg_memory']:.2f} MB (±{results['summary']['dfs']['std_memory']:.2f})\n")
        f.write(f"  Average Recommendations: {results['summary']['dfs']['avg_recommendations']:.2f} (±{results['summary']['dfs']['std_recommendations']:.2f})\n\n")
        
        # BFS summary
        f.write("BFS Algorithm:\n")
        f.write(f"  Average Execution Time: {results['summary']['bfs']['avg_time']:.6f} seconds (±{results['summary']['bfs']['std_time']:.6f})\n")
        f.write(f"  Average Memory Usage: {results['summary']['bfs']['avg_memory']:.2f} MB (±{results['summary']['bfs']['std_memory']:.2f})\n")
        f.write(f"  Average Recommendations: {results['summary']['bfs']['avg_recommendations']:.2f} (±{results['summary']['bfs']['std_recommendations']:.2f})\n\n")
        
        # Similarity summary
        f.write("Recommendation Similarity:\n")
        f.write(f"  Average Jaccard Similarity: {results['summary']['similarity']['avg']:.4f} (±{results['summary']['similarity']['std']:.4f})\n")
        f.write(f"  Min Similarity: {results['summary']['similarity']['min']:.4f}\n")
        f.write(f"  Max Similarity: {results['summary']['similarity']['max']:.4f}\n\n")
        
        # Performance comparison
        time_diff_percent = ((results['summary']['dfs']['avg_time'] - results['summary']['bfs']['avg_time']) / 
                          results['summary']['bfs']['avg_time']) * 100
        
        memory_diff_percent = ((results['summary']['dfs']['avg_memory'] - results['summary']['bfs']['avg_memory']) / 
                           results['summary']['bfs']['avg_memory']) * 100 if results['summary']['bfs']['avg_memory'] != 0 else 0
        
        time_winner = "DFS" if time_diff_percent < 0 else "BFS"
        memory_winner = "DFS" if memory_diff_percent < 0 else "BFS"
        
        f.write("Performance Comparison:\n")
        f.write(f"  Execution Time: {time_winner} is faster by {abs(time_diff_percent):.2f}%\n")
        
        if results['summary']['bfs']['avg_memory'] != 0 or results['summary']['dfs']['avg_memory'] != 0:
            f.write(f"  Memory Usage: {memory_winner} uses less memory by {abs(memory_diff_percent):.2f}%\n\n")
        else:
            f.write("  Memory Usage: Could not be accurately measured\n\n")
        
        # Recommendations
        f.write("Conclusions from Real Facebook Dataset:\n")
        if time_diff_percent < 0:
            f.write("  - For the real-world Facebook network, DFS appears to be faster than BFS, which differs from our synthetic graph.\n")
            f.write("  - This suggests that the structure of real social networks may favor DFS traversal in some cases.\n")
        else:
            f.write("  - For the real-world Facebook network, BFS is faster than DFS, confirming our findings from the synthetic graph.\n")
            f.write("  - This reinforces BFS as the preferred algorithm for friend recommendations in social networks.\n")
            
        if results['summary']['bfs']['avg_recommendations'] > results['summary']['dfs']['avg_recommendations']:
            f.write("  - BFS provides more recommendations on average, offering better coverage of potential friends.\n")
        else:
            f.write("  - In this dataset, DFS provides more recommendations on average, possibly because it explores deeper paths.\n")
            
        f.write(f"  - The high similarity ({results['summary']['similarity']['avg']:.2%}) between DFS and BFS recommendations suggests\n")
        f.write("    that both algorithms identify similar potential friends in real-world social networks.\n")
    
    print(f"Summary saved to {filename}")

def create_visualizations(results):
    """
    Create visualizations of benchmark results
    """
    # Create simple ASCII visualizations
    with open("results/facebook/ascii_charts.txt", "w") as f:
        # Execution time comparison
        f.write("===== Execution Time Comparison (Real Facebook Dataset) =====\n\n")
        f.write(f"DFS Mean: {results['summary']['dfs']['avg_time']:.6f}s\n")
        dfs_bar = "#" * int(results['summary']['dfs']['avg_time'] * 1000)
        f.write(f"DFS: {dfs_bar}\n\n")
        
        f.write(f"BFS Mean: {results['summary']['bfs']['avg_time']:.6f}s\n")
        bfs_bar = "#" * int(results['summary']['bfs']['avg_time'] * 1000)
        f.write(f"BFS: {bfs_bar}\n\n")
        
        # Memory usage comparison
        f.write("===== Memory Usage Comparison =====\n\n")
        f.write(f"DFS Mean: {results['summary']['dfs']['avg_memory']:.2f}MB\n")
        dfs_bar = "#" * int(results['summary']['dfs']['avg_memory'] * 10)
        f.write(f"DFS: {dfs_bar}\n\n")
        
        f.write(f"BFS Mean: {results['summary']['bfs']['avg_memory']:.2f}MB\n")
        bfs_bar = "#" * int(results['summary']['bfs']['avg_memory'] * 10)
        f.write(f"BFS: {bfs_bar}\n\n")
        
        # Recommendation count comparison
        f.write("===== Recommendation Count Comparison =====\n\n")
        f.write(f"DFS Mean: {results['summary']['dfs']['avg_recommendations']:.2f}\n")
        dfs_bar = "#" * int(results['summary']['dfs']['avg_recommendations'])
        f.write(f"DFS: {dfs_bar}\n\n")
        
        f.write(f"BFS Mean: {results['summary']['bfs']['avg_recommendations']:.2f}\n")
        bfs_bar = "#" * int(results['summary']['bfs']['avg_recommendations'])
        f.write(f"BFS: {bfs_bar}\n\n")
        
        # Similarity distribution
        f.write("===== Similarity Distribution =====\n\n")
        similarity_bins = {
            "0.0-0.2": 0,
            "0.2-0.4": 0,
            "0.4-0.6": 0,
            "0.6-0.8": 0,
            "0.8-1.0": 0
        }
        
        for sim in results['similarity']:
            if sim < 0.2:
                similarity_bins["0.0-0.2"] += 1
            elif sim < 0.4:
                similarity_bins["0.2-0.4"] += 1
            elif sim < 0.6:
                similarity_bins["0.4-0.6"] += 1
            elif sim < 0.8:
                similarity_bins["0.6-0.8"] += 1
            else:
                similarity_bins["0.8-1.0"] += 1
        
        for bin_range, count in similarity_bins.items():
            bar = "#" * count
            f.write(f"{bin_range}: {bar} ({count})\n")
    
    print("ASCII visualizations saved to results/facebook/ascii_charts.txt")
    
    try:
        # Try to create matplotlib visualizations if available
        import matplotlib.pyplot as plt
        
        # Execution time comparison
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.boxplot([results['dfs']['times'], results['bfs']['times']], labels=['DFS', 'BFS'])
        plt.title('Execution Time Comparison\n(Real Facebook Dataset)')
        plt.ylabel('Time (seconds)')
        
        # Memory usage comparison
        plt.subplot(1, 2, 2)
        plt.boxplot([results['dfs']['memory'], results['bfs']['memory']], labels=['DFS', 'BFS'])
        plt.title('Memory Usage Comparison\n(Real Facebook Dataset)')
        plt.ylabel('Memory (MB)')
        
        plt.tight_layout()
        plt.savefig("results/facebook/performance_comparison.png")
        print("Performance comparison plot saved to results/facebook/performance_comparison.png")
        
        # Recommendation similarity histogram
        plt.figure(figsize=(8, 6))
        plt.hist(results['similarity'], bins=10, alpha=0.7)
        plt.title('Recommendation Similarity Distribution\n(Real Facebook Dataset)')
        plt.xlabel('Similarity Score')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.savefig("results/facebook/recommendation_similarity.png")
        print("Recommendation similarity plot saved to results/facebook/recommendation_similarity.png")
        
        # Recommendation count comparison
        plt.figure(figsize=(8, 6))
        plt.boxplot([results['dfs']['recommendation_counts'], results['bfs']['recommendation_counts']], labels=['DFS', 'BFS'])
        plt.title('Number of Recommendations Comparison\n(Real Facebook Dataset)')
        plt.ylabel('Number of Recommendations')
        plt.savefig("results/facebook/recommendation_count.png")
        print("Recommendation count plot saved to results/facebook/recommendation_count.png")
        
    except ImportError:
        print("Matplotlib not available. Only ASCII visualizations were created.")

def main():
    """Run benchmark on the real Facebook dataset"""
    print("DFS vs BFS for Friend Recommendation - Real Facebook Dataset")
    print("==========================================================")
    
    # Download the dataset
    dataset_path = download_facebook_dataset()
    
    # Load the Facebook graph
    graph = load_facebook_graph(dataset_path)
    
    # Run benchmark on a subset of users
    results = run_benchmark(graph, num_users=50, max_depth=2)
    
    # Save results
    save_results_csv(results)
    save_summary_txt(results)
    create_visualizations(results)
    
    # Print summary
    print("\nBenchmark Summary (Real Facebook Dataset):")
    print(f"DFS average time: {results['summary']['dfs']['avg_time']:.6f} seconds")
    print(f"BFS average time: {results['summary']['bfs']['avg_time']:.6f} seconds")
    print(f"Average recommendation similarity: {results['summary']['similarity']['avg']:.4f}")
    
    # Compare with synthetic graph results
    print("\nComparing with synthetic graph results:")
    try:
        with open("results/benchmark_summary.txt", "r") as f:
            content = f.read()
            if "BFS is faster by" in content:
                print("In our synthetic graph benchmark, BFS was faster.")
            elif "DFS is faster by" in content:
                print("In our synthetic graph benchmark, DFS was faster.")
                
        if results['summary']['bfs']['avg_time'] < results['summary']['dfs']['avg_time']:
            print("In the real Facebook dataset, BFS is also faster.")
        else:
            print("In the real Facebook dataset, DFS is faster, which differs from our synthetic graph.")
            print("This indicates that real-world social network structures may have different traversal properties.")
    except:
        print("Could not compare with synthetic graph results.")
    
    print("\nBenchmark with real Facebook dataset completed successfully!")

if __name__ == "__main__":
    main()