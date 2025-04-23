#!/usr/bin/env python3

import os
import urllib.request
import zipfile
import networkx as nx
import pandas as pd
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    """Progress bar for downloads"""
    def update_to(self, block_num, block_size, total_size):
        if total_size is not None:
            self.total = total_size
        self.update(block_size)

def download_facebook_dataset(output_dir="../data"):
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
    Load the Facebook dataset as a NetworkX graph
    Args:
        file_path: Path to the dataset file
    Returns:
        NetworkX graph
    """
    print("Loading Facebook graph...")
    # Load the graph from the edge list
    graph = nx.read_edgelist(file_path)
    
    print(f"Graph loaded with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    return graph

def create_graph_from_dataset():
    """
    Download the Facebook dataset and create a NetworkX graph
    Returns:
        NetworkX graph
    """
    data_path = download_facebook_dataset()
    graph = load_facebook_graph(data_path)
    return graph

if __name__ == "__main__":
    # Download and load the dataset
    graph = create_graph_from_dataset()
    
    # Print some basic statistics
    print("\nGraph Statistics:")
    print(f"Number of nodes: {graph.number_of_nodes()}")
    print(f"Number of edges: {graph.number_of_edges()}")
    print(f"Average degree: {sum(dict(graph.degree()).values()) / graph.number_of_nodes():.2f}")
    print(f"Is connected: {nx.is_connected(graph)}")
    
    # Save the graph for later use
    nx.write_gpickle(graph, "../data/facebook_graph.gpickle")
    print("Graph saved to ../data/facebook_graph.gpickle")