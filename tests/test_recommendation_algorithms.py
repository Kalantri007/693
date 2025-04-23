#!/usr/bin/env python3

import os
import sys
import unittest
import networkx as nx
import numpy as np

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.recommendation_algorithms import DFSRecommender, BFSRecommender

class TestRecommendationAlgorithms(unittest.TestCase):
    """Test cases for the recommendation algorithms"""
    
    def setUp(self):
        """Set up a test graph for the test cases"""
        # Create a simple graph for testing
        self.graph = nx.Graph()
        
        # Create a simple social network with known structure
        edges = [
            ('1', '2'), ('1', '3'), ('1', '4'),
            ('2', '5'), ('2', '6'),
            ('3', '7'), ('3', '8'),
            ('4', '9'),
            ('5', '10'), ('5', '11'),
            ('7', '11'), ('7', '12'),
            ('10', '11'), ('10', '12')
        ]
        self.graph.add_edges_from(edges)
        
        # Create instances of the recommenders
        self.dfs_recommender = DFSRecommender(self.graph)
        self.bfs_recommender = BFSRecommender(self.graph)
    
    def test_dfs_basic_functionality(self):
        """Test that DFS produces recommendations for a user"""
        recommendations, _ = self.dfs_recommender.recommend('1', top_n=5)
        
        # Check that we got recommendations
        self.assertTrue(len(recommendations) > 0, "DFS should return recommendations")
        
        # Check format of recommendations
        self.assertEqual(len(recommendations[0]), 2, "Recommendations should be (user_id, score) pairs")
        
        # Check that the recommendations are sorted by score
        scores = [score for _, score in recommendations]
        self.assertEqual(scores, sorted(scores, reverse=True), "Recommendations should be sorted by score")
    
    def test_bfs_basic_functionality(self):
        """Test that BFS produces recommendations for a user"""
        recommendations, _ = self.bfs_recommender.recommend('1', top_n=5)
        
        # Check that we got recommendations
        self.assertTrue(len(recommendations) > 0, "BFS should return recommendations")
        
        # Check format of recommendations
        self.assertEqual(len(recommendations[0]), 2, "Recommendations should be (user_id, score) pairs")
        
        # Check that the recommendations are sorted by score
        scores = [score for _, score in recommendations]
        self.assertEqual(scores, sorted(scores, reverse=True), "Recommendations should be sorted by score")
    
    def test_no_self_recommendation(self):
        """Test that users are not recommended to themselves"""
        dfs_recommendations, _ = self.dfs_recommender.recommend('1')
        bfs_recommendations, _ = self.bfs_recommender.recommend('1')
        
        # Check that the user is not in their own recommendations
        dfs_users = [user for user, _ in dfs_recommendations]
        bfs_users = [user for user, _ in bfs_recommendations]
        
        self.assertNotIn('1', dfs_users, "User should not be recommended to themselves in DFS")
        self.assertNotIn('1', bfs_users, "User should not be recommended to themselves in BFS")
    
    def test_no_existing_friends(self):
        """Test that existing friends are not recommended"""
        dfs_recommendations, _ = self.dfs_recommender.recommend('1')
        bfs_recommendations, _ = self.bfs_recommender.recommend('1')
        
        # Get the friends of user '1'
        friends = list(self.graph.neighbors('1'))
        
        # Check that none of the friends are in the recommendations
        dfs_users = [user for user, _ in dfs_recommendations]
        bfs_users = [user for user, _ in bfs_recommendations]
        
        for friend in friends:
            self.assertNotIn(friend, dfs_users, f"Friend {friend} should not be in DFS recommendations")
            self.assertNotIn(friend, bfs_users, f"Friend {friend} should not be in BFS recommendations")
    
    def test_recommendation_depth(self):
        """Test that recommendations respect the max_depth parameter"""
        # Set max_depth to 1 to only get direct friends (which should be filtered out)
        dfs_recommendations, _ = self.dfs_recommender.recommend('1', max_depth=1)
        bfs_recommendations, _ = self.bfs_recommender.recommend('1', max_depth=1)
        
        # With max_depth=1, we should get no recommendations since direct friends are filtered
        self.assertEqual(len(dfs_recommendations), 0, "DFS with max_depth=1 should return no recommendations")
        self.assertEqual(len(bfs_recommendations), 0, "BFS with max_depth=1 should return no recommendations")
        
        # With max_depth=2, we should get friends of friends
        dfs_recommendations, _ = self.dfs_recommender.recommend('1', max_depth=2)
        bfs_recommendations, _ = self.bfs_recommender.recommend('1', max_depth=2)
        
        # We expect to get some recommendations now
        self.assertTrue(len(dfs_recommendations) > 0, "DFS with max_depth=2 should return recommendations")
        self.assertTrue(len(bfs_recommendations) > 0, "BFS with max_depth=2 should return recommendations")
    
    def test_top_n_limit(self):
        """Test that the number of recommendations respects the top_n parameter"""
        top_n = 3
        dfs_recommendations, _ = self.dfs_recommender.recommend('1', top_n=top_n)
        bfs_recommendations, _ = self.bfs_recommender.recommend('1', top_n=top_n)
        
        self.assertLessEqual(len(dfs_recommendations), top_n, "DFS should respect top_n limit")
        self.assertLessEqual(len(bfs_recommendations), top_n, "BFS should respect top_n limit")
    
    def test_common_neighbors_score(self):
        """Test that the scoring function based on common neighbors works correctly"""
        # For this test, we need to check a specific case
        # In our test graph, nodes '10' and '11' have common neighbors: '5'
        # So the score for recommending '10' to '11' should be 1
        
        # Create a recommender instance for testing
        recommender = DFSRecommender(self.graph)
        
        # Check the common neighbors score
        score = recommender._get_common_neighbors_score('10', '11')
        self.assertEqual(score, 1, "The common neighbors score for nodes '10' and '11' should be 1")

if __name__ == '__main__':
    unittest.main()