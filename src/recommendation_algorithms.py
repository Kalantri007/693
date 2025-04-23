#!/usr/bin/env python3

import networkx as nx
import time
import psutil
from collections import defaultdict, deque
from memory_profiler import memory_usage

class FriendRecommender:
    """Base class for friend recommendation algorithms"""
    
    def __init__(self, graph):
        """
        Initialize the recommender with a social network graph
        Args:
            graph: NetworkX graph representing the social network
        """
        self.graph = graph
        
    def recommend(self, user_id, top_n=10):
        """
        Abstract method to be implemented by subclasses
        Args:
            user_id: ID of the user to get recommendations for
            top_n: Number of recommendations to return
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def _get_common_neighbors_score(self, user_id, candidate):
        """
        Calculate a score based on the number of common neighbors
        Args:
            user_id: ID of the target user
            candidate: ID of the candidate friend
        Returns:
            Score representing the strength of the recommendation
        """
        user_neighbors = set(self.graph.neighbors(user_id))
        candidate_neighbors = set(self.graph.neighbors(candidate))
        return len(user_neighbors.intersection(candidate_neighbors))
    
    def _filter_candidates(self, user_id, candidates):
        """
        Filter candidates to remove already connected users
        Args:
            user_id: ID of the target user
            candidates: List of candidate IDs
        Returns:
            Filtered list of candidates
        """
        # Remove the user themselves and users they are already connected to
        user_neighbors = set(self.graph.neighbors(user_id))
        return [c for c in candidates if c != user_id and c not in user_neighbors]

class DFSRecommender(FriendRecommender):
    """Friend recommendation based on Depth-First Search"""
    
    def recommend(self, user_id, top_n=10, max_depth=2):
        """
        Use DFS to find friends of friends as recommendations
        Args:
            user_id: ID of the user to get recommendations for
            top_n: Number of recommendations to return
            max_depth: Maximum depth for DFS (2 means friends of friends)
        Returns:
            List of (candidate_id, score) tuples
        """
        start_time = time.time()
        
        # Track candidates and their frequency
        candidates = defaultdict(int)
        visited = set([user_id])  # Start with user as visited
        
        def dfs(node, current_depth):
            if current_depth > max_depth:
                return
            
            for neighbor in self.graph.neighbors(node):
                if current_depth == max_depth and neighbor not in visited:
                    # Found a potential recommendation at desired depth
                    candidates[neighbor] += 1
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    dfs(neighbor, current_depth + 1)
        
        # Start DFS from the user
        dfs(user_id, 1)
        
        # Remove inappropriate candidates and sort by frequency
        filtered_candidates = self._filter_candidates(user_id, candidates.keys())
        
        # Score candidates based on common neighbors
        recommendations = [(candidate, self._get_common_neighbors_score(user_id, candidate)) 
                           for candidate in filtered_candidates]
        
        # Sort by score in descending order
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        return recommendations[:top_n], execution_time

class BFSRecommender(FriendRecommender):
    """Friend recommendation based on Breadth-First Search"""
    
    def recommend(self, user_id, top_n=10, max_depth=2):
        """
        Use BFS to find friends of friends as recommendations
        Args:
            user_id: ID of the user to get recommendations for
            top_n: Number of recommendations to return
            max_depth: Maximum depth for BFS (2 means friends of friends)
        Returns:
            List of (candidate_id, score) tuples
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
                
            for neighbor in self.graph.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
                    
                    if depth == max_depth - 1:
                        # This is a friend-of-friend
                        candidates[neighbor] += 1
        
        # Remove inappropriate candidates
        filtered_candidates = self._filter_candidates(user_id, candidates.keys())
        
        # Score candidates based on common neighbors
        recommendations = [(candidate, self._get_common_neighbors_score(user_id, candidate)) 
                           for candidate in filtered_candidates]
        
        # Sort by score in descending order
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        return recommendations[:top_n], execution_time

def profile_memory_usage(func, *args, **kwargs):
    """
    Profile the memory usage of a function
    Args:
        func: Function to profile
        *args, **kwargs: Arguments to pass to the function
    Returns:
        (result, max_memory)
    """
    # Get the peak memory usage during function execution
    mem_usage, result = memory_usage(
        (func, args, kwargs),
        retval=True,
        interval=0.1,
        timeout=None,
        max_iterations=1
    )
    
    # Return the function result and the max memory usage
    return result, max(mem_usage)

def measure_performance(recommender, user_id, top_n=10, max_depth=2):
    """
    Measure the performance of a recommendation algorithm
    Args:
        recommender: Recommender instance
        user_id: ID of the target user
        top_n: Number of recommendations to return
        max_depth: Maximum depth for search
    Returns:
        Dictionary with performance metrics
    """
    # Measure execution time directly in the recommend method
    func = recommender.recommend
    args = (user_id, top_n, max_depth)
    
    # Measure memory usage
    (recommendations, exec_time), max_memory = profile_memory_usage(func, *args)
    
    return {
        'recommendations': recommendations,
        'execution_time': exec_time,
        'memory_usage': max_memory
    }

if __name__ == "__main__":
    # Simple test with a test graph
    G = nx.Graph()
    # Create a simple social network
    edges = [
        (1, 2), (1, 3), (1, 4),
        (2, 5), (2, 6),
        (3, 7), (3, 8),
        (4, 9),
        (5, 10), (5, 11),
        (7, 11), (7, 12),
        (10, 11), (10, 12)
    ]
    G.add_edges_from(edges)
    
    # Test both recommenders
    dfs_recommender = DFSRecommender(G)
    bfs_recommender = BFSRecommender(G)
    
    # Get recommendations for user 1
    dfs_result = measure_performance(dfs_recommender, '1')
    bfs_result = measure_performance(bfs_recommender, '1')
    
    print("DFS Recommendations:", dfs_result['recommendations'])
    print(f"DFS Execution Time: {dfs_result['execution_time']:.6f} seconds")
    print(f"DFS Memory Usage: {dfs_result['memory_usage']:.2f} MB")
    
    print("\nBFS Recommendations:", bfs_result['recommendations'])
    print(f"BFS Execution Time: {bfs_result['execution_time']:.6f} seconds")
    print(f"BFS Memory Usage: {bfs_result['memory_usage']:.2f} MB")