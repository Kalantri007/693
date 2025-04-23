#!/usr/bin/env python3

import os
import argparse
import time
from simple_benchmark import SimpleRecommendationBenchmark
from data_acquisition import create_graph_from_dataset

def main():
    """
    Simple script to run the friend recommendation benchmark
    """
    parser = argparse.ArgumentParser(
        description="Compare DFS and BFS algorithms for friend recommendations in social networks"
    )
    
    parser.add_argument(
        "--download", 
        action="store_true", 
        help="Force download of the dataset even if it exists locally"
    )
    parser.add_argument(
        "--num-users", 
        type=int, 
        default=5, 
        help="Number of random users to test recommendations for"
    )
    parser.add_argument(
        "--top-n", 
        type=int, 
        default=10, 
        help="Number of recommendations to generate for each user"
    )
    parser.add_argument(
        "--max-depth", 
        type=int, 
        default=2, 
        help="Maximum depth for graph traversal (2 means friends of friends)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="../results", 
        help="Directory to save results"
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    start_time = time.time()
    
    print("============================================")
    print("DFS vs BFS for Friend Recommendations Study")
    print("============================================")
    
    # Step 1: Download and process the dataset
    print("\n[Step 1] Downloading and processing dataset...")
    if args.download:
        # Force re-download of the dataset
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "facebook_graph.gpickle")
        if os.path.exists(data_path):
            os.remove(data_path)
    
    graph = create_graph_from_dataset()
    
    # Step 2: Run the benchmark
    print("\n[Step 2] Running benchmark...")
    benchmark = SimpleRecommendationBenchmark(graph)
    benchmark.run_benchmark(
        num_users=args.num_users,
        top_n=args.top_n,
        max_depth=args.max_depth
    )
    
    # Step 3: Print and save the results
    print("\n[Step 3] Analyzing results...")
    benchmark.print_summary()
    benchmark.save_results(args.output_dir)
    
    # Calculate total execution time
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    
    print("\n============================================")
    print(f"Results saved to {args.output_dir}")
    print("Study completed successfully!")
    print("============================================")

if __name__ == "__main__":
    main()