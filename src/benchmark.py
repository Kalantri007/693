#!/usr/bin/env python3

import os
import time
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_acquisition import create_graph_from_dataset
from recommendation_algorithms import DFSRecommender, BFSRecommender, measure_performance

class RecommendationBenchmark:
    """Benchmark class for comparing friend recommendation algorithms"""
    
    def __init__(self, graph=None):
        """
        Initialize the benchmark with a social network graph
        Args:
            graph: NetworkX graph representing the social network (optional)
        """
        self.graph = graph if graph else self._load_graph()
        self.results = {}
    
    def _load_graph(self):
        """
        Load or create the social network graph
        Returns:
            NetworkX graph
        """
        graph_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "facebook_graph.gpickle")
        
        if os.path.exists(graph_path):
            print(f"Loading graph from {graph_path}")
            return nx.read_gpickle(graph_path)
        else:
            print("Graph file not found. Downloading and creating the graph...")
            return create_graph_from_dataset()
        
    def run_benchmark(self, user_ids=None, top_n=10, max_depth=2, num_users=50):
        """
        Run benchmark tests on both DFS and BFS algorithms
        Args:
            user_ids: List of user IDs to test (if None, random users will be selected)
            top_n: Number of recommendations to return
            max_depth: Maximum depth for search
            num_users: Number of random users to test if user_ids is None
        Returns:
            Dictionary with benchmark results
        """
        if user_ids is None:
            # Select random users for testing
            all_users = list(self.graph.nodes())
            user_ids = np.random.choice(all_users, min(num_users, len(all_users)), replace=False)
        
        print(f"Running benchmark on {len(user_ids)} users...")
        
        dfs_recommender = DFSRecommender(self.graph)
        bfs_recommender = BFSRecommender(self.graph)
        
        dfs_times = []
        dfs_memory = []
        dfs_num_recommendations = []
        
        bfs_times = []
        bfs_memory = []
        bfs_num_recommendations = []
        
        # Keep track of common recommendations between DFS and BFS
        similarity_scores = []
        
        # Run the benchmark for each user
        for user_id in user_ids:
            print(f"Processing user {user_id}...")
            
            # Measure DFS performance
            dfs_result = measure_performance(dfs_recommender, user_id, top_n, max_depth)
            dfs_times.append(dfs_result['execution_time'])
            dfs_memory.append(dfs_result['memory_usage'])
            dfs_num_recommendations.append(len(dfs_result['recommendations']))
            
            # Measure BFS performance
            bfs_result = measure_performance(bfs_recommender, user_id, top_n, max_depth)
            bfs_times.append(bfs_result['execution_time'])
            bfs_memory.append(bfs_result['memory_usage'])
            bfs_num_recommendations.append(len(bfs_result['recommendations']))
            
            # Calculate similarity between DFS and BFS recommendations
            dfs_rec_set = set([rec[0] for rec in dfs_result['recommendations']])
            bfs_rec_set = set([rec[0] for rec in bfs_result['recommendations']])
            
            if len(dfs_rec_set) > 0 and len(bfs_rec_set) > 0:
                jaccard_similarity = len(dfs_rec_set.intersection(bfs_rec_set)) / len(dfs_rec_set.union(bfs_rec_set))
                similarity_scores.append(jaccard_similarity)
        
        # Compile the results
        self.results = {
            'dfs': {
                'execution_time': {
                    'mean': np.mean(dfs_times),
                    'std': np.std(dfs_times),
                    'values': dfs_times
                },
                'memory_usage': {
                    'mean': np.mean(dfs_memory),
                    'std': np.std(dfs_memory),
                    'values': dfs_memory
                },
                'num_recommendations': {
                    'mean': np.mean(dfs_num_recommendations),
                    'std': np.std(dfs_num_recommendations),
                    'values': dfs_num_recommendations
                }
            },
            'bfs': {
                'execution_time': {
                    'mean': np.mean(bfs_times),
                    'std': np.std(bfs_times),
                    'values': bfs_times
                },
                'memory_usage': {
                    'mean': np.mean(bfs_memory),
                    'std': np.std(bfs_memory),
                    'values': bfs_memory
                },
                'num_recommendations': {
                    'mean': np.mean(bfs_num_recommendations),
                    'std': np.std(bfs_num_recommendations),
                    'values': bfs_num_recommendations
                }
            },
            'similarity': {
                'mean': np.mean(similarity_scores) if similarity_scores else 0,
                'std': np.std(similarity_scores) if similarity_scores else 0,
                'values': similarity_scores
            }
        }
        
        return self.results
    
    def print_summary(self):
        """
        Print a summary of the benchmark results
        """
        if not self.results:
            print("No benchmark results available. Run run_benchmark() first.")
            return
        
        print("\n============= BENCHMARK SUMMARY =============")
        print("\nDFS Algorithm:")
        print(f"  Average Execution Time: {self.results['dfs']['execution_time']['mean']:.6f} seconds (±{self.results['dfs']['execution_time']['std']:.6f})")
        print(f"  Average Memory Usage: {self.results['dfs']['memory_usage']['mean']:.2f} MB (±{self.results['dfs']['memory_usage']['std']:.2f})")
        print(f"  Average Number of Recommendations: {self.results['dfs']['num_recommendations']['mean']:.2f} (±{self.results['dfs']['num_recommendations']['std']:.2f})")
        
        print("\nBFS Algorithm:")
        print(f"  Average Execution Time: {self.results['bfs']['execution_time']['mean']:.6f} seconds (±{self.results['bfs']['execution_time']['std']:.6f})")
        print(f"  Average Memory Usage: {self.results['bfs']['memory_usage']['mean']:.2f} MB (±{self.results['bfs']['memory_usage']['std']:.2f})")
        print(f"  Average Number of Recommendations: {self.results['bfs']['num_recommendations']['mean']:.2f} (±{self.results['bfs']['num_recommendations']['std']:.2f})")
        
        print("\nRecommendation Similarity:")
        print(f"  Average Jaccard Similarity: {self.results['similarity']['mean']:.4f} (±{self.results['similarity']['std']:.4f})")
        
        # Determine which algorithm performed better
        time_diff_percent = ((self.results['dfs']['execution_time']['mean'] - self.results['bfs']['execution_time']['mean']) / 
                            self.results['bfs']['execution_time']['mean']) * 100
        
        memory_diff_percent = ((self.results['dfs']['memory_usage']['mean'] - self.results['bfs']['memory_usage']['mean']) / 
                               self.results['bfs']['memory_usage']['mean']) * 100
        
        time_winner = "DFS" if time_diff_percent < 0 else "BFS"
        memory_winner = "DFS" if memory_diff_percent < 0 else "BFS"
        
        print("\nPerformance Comparison:")
        print(f"  Execution Time: {time_winner} is faster by {abs(time_diff_percent):.2f}%")
        print(f"  Memory Usage: {memory_winner} uses less memory by {abs(memory_diff_percent):.2f}%")
        print("=============================================")
    
    def save_results(self, output_dir="../results"):
        """
        Save the benchmark results to CSV files
        Args:
            output_dir: Directory to save the results
        """
        if not self.results:
            print("No benchmark results available. Run run_benchmark() first.")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create DataFrames for the results
        dfs_df = pd.DataFrame({
            'user_idx': range(len(self.results['dfs']['execution_time']['values'])),
            'execution_time': self.results['dfs']['execution_time']['values'],
            'memory_usage': self.results['dfs']['memory_usage']['values'],
            'num_recommendations': self.results['dfs']['num_recommendations']['values'],
            'algorithm': 'DFS'
        })
        
        bfs_df = pd.DataFrame({
            'user_idx': range(len(self.results['bfs']['execution_time']['values'])),
            'execution_time': self.results['bfs']['execution_time']['values'],
            'memory_usage': self.results['bfs']['memory_usage']['values'],
            'num_recommendations': self.results['bfs']['num_recommendations']['values'],
            'algorithm': 'BFS'
        })
        
        # Combine the results
        results_df = pd.concat([dfs_df, bfs_df], ignore_index=True)
        
        # Save to CSV
        results_df.to_csv(os.path.join(output_dir, "benchmark_results.csv"), index=False)
        print(f"Results saved to {os.path.join(output_dir, 'benchmark_results.csv')}")
        
    def visualize_results(self, output_dir="../results"):
        """
        Create visualizations of the benchmark results
        Args:
            output_dir: Directory to save the visualizations
        """
        if not self.results:
            print("No benchmark results available. Run run_benchmark() first.")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Execution Time Comparison
        plt.figure(figsize=(12, 6))
        
        # Boxplot for execution time
        plt.subplot(1, 2, 1)
        dfs_times = self.results['dfs']['execution_time']['values']
        bfs_times = self.results['bfs']['execution_time']['values']
        plt.boxplot([dfs_times, bfs_times], labels=['DFS', 'BFS'])
        plt.title('Execution Time Comparison')
        plt.ylabel('Time (seconds)')
        
        # Boxplot for memory usage
        plt.subplot(1, 2, 2)
        dfs_memory = self.results['dfs']['memory_usage']['values']
        bfs_memory = self.results['bfs']['memory_usage']['values']
        plt.boxplot([dfs_memory, bfs_memory], labels=['DFS', 'BFS'])
        plt.title('Memory Usage Comparison')
        plt.ylabel('Memory (MB)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "performance_comparison.png"))
        print(f"Performance comparison plot saved to {os.path.join(output_dir, 'performance_comparison.png')}")
        
        # Recommendation Similarity Histogram
        plt.figure(figsize=(10, 6))
        plt.hist(self.results['similarity']['values'], bins=10, alpha=0.7)
        plt.title('Recommendation Similarity Distribution (Jaccard Similarity)')
        plt.xlabel('Similarity Score')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "recommendation_similarity.png"))
        print(f"Recommendation similarity plot saved to {os.path.join(output_dir, 'recommendation_similarity.png')}")

if __name__ == "__main__":
    # Run a sample benchmark
    benchmark = RecommendationBenchmark()
    benchmark.run_benchmark(num_users=10)  # Use 10 random users for testing
    benchmark.print_summary()
    benchmark.save_results()
    benchmark.visualize_results()