#!/usr/bin/env python3

"""
A simplified example demonstrating DFS vs BFS for friend recommendations
This script uses minimal dependencies to avoid environment issues
"""

import time
from collections import defaultdict, deque

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

def create_sample_graph():
    """Create a sample social network graph for testing"""
    # Create a graph representing a small social network
    graph = SimpleGraph()
    
    # Add edges representing friendships
    edges = [
        (1, 2), (1, 3), (1, 4),
        (2, 5), (2, 6),
        (3, 7), (3, 8),
        (4, 9),
        (5, 10), (5, 11),
        (7, 11), (7, 12),
        (10, 11), (10, 12),
        (11, 13), (12, 14),
        (13, 15), (14, 15)
    ]
    
    for u, v in edges:
        graph.add_edge(u, v)
    
    return graph

def dfs_recommendations(graph, user_id, max_depth=2):
    """
    Get friend recommendations using Depth-First Search
    
    Args:
        graph: The social network graph
        user_id: The user to get recommendations for
        max_depth: Maximum depth for search
    
    Returns:
        List of recommended friend IDs
    """
    start_time = time.time()
    
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
    
    return filtered_candidates, execution_time

def bfs_recommendations(graph, user_id, max_depth=2):
    """
    Get friend recommendations using Breadth-First Search
    
    Args:
        graph: The social network graph
        user_id: The user to get recommendations for
        max_depth: Maximum depth for search
    
    Returns:
        List of recommended friend IDs
    """
    start_time = time.time()
    
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
    
    return filtered_candidates, execution_time

def main():
    """Run a simple comparison of DFS and BFS for friend recommendations"""
    print("DFS vs BFS for Friend Recommendation Systems")
    print("===========================================")
    
    # Create a sample social network
    graph = create_sample_graph()
    print(f"Created graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    
    # Test with different users
    test_users = [1, 2, 5, 10]
    
    for user_id in test_users:
        print(f"\nRecommendations for user {user_id}:")
        
        # Get DFS recommendations
        dfs_recs, dfs_time = dfs_recommendations(graph, user_id)
        print(f"DFS recommends: {dfs_recs}")
        print(f"DFS execution time: {dfs_time:.6f} seconds")
        
        # Get BFS recommendations
        bfs_recs, bfs_time = bfs_recommendations(graph, user_id)
        print(f"BFS recommends: {bfs_recs}")
        print(f"BFS execution time: {bfs_time:.6f} seconds")
        
        # Compare the recommendations
        common_recs = set(dfs_recs).intersection(set(bfs_recs))
        all_recs = set(dfs_recs).union(set(bfs_recs))
        similarity = len(common_recs) / len(all_recs) if all_recs else 0
        
        print(f"Recommendation similarity: {similarity:.4f}")
        print(f"Common recommendations: {list(common_recs)}")
        
        # Determine which algorithm was faster
        if dfs_time < bfs_time:
            print(f"DFS was faster by {((bfs_time - dfs_time) / bfs_time) * 100:.2f}%")
        else:
            print(f"BFS was faster by {((dfs_time - bfs_time) / dfs_time) * 100:.2f}%")
    
    print("\nConclusion:")
    print("""
    This example demonstrates the application of DFS and BFS algorithms to generate 
    friend recommendations in social networks. While both algorithms explore the graph 
    to find potential connections, they do so in different ways:
    
    - DFS explores paths deeply before backtracking, which can be efficient for 
      finding long paths but may not always provide the most relevant recommendations.
    
    - BFS explores neighbors at each level before moving deeper, which generally 
      provides more relevant recommendations for social networks since closer 
      connections tend to be more meaningful.
    
    The choice between DFS and BFS depends on the specific requirements of the 
    application and the structure of the social network graph.
    """)

if __name__ == "__main__":
    main()