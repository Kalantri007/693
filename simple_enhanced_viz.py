#!/usr/bin/env python3

"""
Simplified enhanced visualization script for DFS vs BFS comparison
Using only matplotlib to avoid dependency issues
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create results directory if it doesn't exist
os.makedirs('results/enhanced', exist_ok=True)

def load_results(file_path):
    """Load benchmark results from CSV file"""
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None

def create_execution_time_comparison(synthetic_df, facebook_df, output_path):
    """Create enhanced execution time comparison visualization"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Prepare data
    dfs_synth = synthetic_df[synthetic_df['algorithm'] == 'DFS']['execution_time']
    bfs_synth = synthetic_df[synthetic_df['algorithm'] == 'BFS']['execution_time']
    
    dfs_fb = facebook_df[facebook_df['algorithm'] == 'DFS']['execution_time']
    bfs_fb = facebook_df[facebook_df['algorithm'] == 'BFS']['execution_time']
    
    # Colors
    colors = ['#3498db', '#e74c3c']  # Blue for BFS, Red for DFS
    
    # Synthetic network plot - simple boxplot
    axes[0].set_title('Execution Time: Synthetic Network', fontsize=14)
    axes[0].boxplot([dfs_synth, bfs_synth], patch_artist=True,
                    boxprops=dict(facecolor=colors[1], color='black'),
                    medianprops=dict(color='black'))
    axes[0].set_yscale('log')
    axes[0].set_ylabel('Time (seconds, log scale)', fontsize=12)
    axes[0].set_xticklabels(['DFS', 'BFS'])
    axes[0].grid(True, alpha=0.3)
    
    # Add means
    dfs_mean = dfs_synth.mean()
    bfs_mean = bfs_synth.mean()
    axes[0].axhline(y=dfs_mean, color='black', linestyle='--', alpha=0.7)
    axes[0].axhline(y=bfs_mean, color='black', linestyle='--', alpha=0.7)
    
    # Annotate with speedup percentage
    speedup = ((dfs_mean - bfs_mean) / bfs_mean) * 100
    axes[0].text(0.5, 0.95, f"BFS is {speedup:.1f}% faster", 
             transform=axes[0].transAxes, 
             ha='center', va='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Facebook network plot
    axes[1].set_title('Execution Time: Real Facebook Network', fontsize=14)
    axes[1].boxplot([dfs_fb, bfs_fb], patch_artist=True,
                    boxprops=dict(facecolor=colors[0], color='black'),
                    medianprops=dict(color='black'))
    axes[1].set_yscale('log')
    axes[1].set_ylabel('Time (seconds, log scale)', fontsize=12)
    axes[1].set_xticklabels(['DFS', 'BFS'])
    axes[1].grid(True, alpha=0.3)
    
    # Add means
    dfs_mean = dfs_fb.mean()
    bfs_mean = bfs_fb.mean()
    axes[1].axhline(y=dfs_mean, color='black', linestyle='--', alpha=0.7)
    axes[1].axhline(y=bfs_mean, color='black', linestyle='--', alpha=0.7)
    
    # Annotate with speedup percentage
    speedup = ((dfs_mean - bfs_mean) / bfs_mean) * 100
    axes[1].text(0.5, 0.95, f"BFS is {speedup:.1f}% faster", 
             transform=axes[1].transAxes, 
             ha='center', va='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Execution time comparison saved to {output_path}")

def create_recommendation_count_comparison(synthetic_df, facebook_df, output_path):
    """Create enhanced recommendation count comparison visualization"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Prepare data
    dfs_synth = synthetic_df[synthetic_df['algorithm'] == 'DFS']['recommendation_count']
    bfs_synth = synthetic_df[synthetic_df['algorithm'] == 'BFS']['recommendation_count']
    
    dfs_fb = facebook_df[facebook_df['algorithm'] == 'DFS']['recommendation_count']
    bfs_fb = facebook_df[facebook_df['algorithm'] == 'BFS']['recommendation_count']
    
    # Colors
    colors = ['#3498db', '#e74c3c']  # Blue for BFS, Red for DFS
    
    # Synthetic network plot
    axes[0].set_title('Recommendation Count: Synthetic Network', fontsize=14)
    bar_width = 0.35
    x = np.array([0, 1])
    axes[0].bar(x - bar_width/2, [dfs_synth.mean(), bfs_synth.mean()], bar_width, 
              label='Mean Count', color=[colors[1], colors[0]])
    
    # Add error bars
    axes[0].errorbar(x - bar_width/2, [dfs_synth.mean(), bfs_synth.mean()], 
                   yerr=[dfs_synth.std(), bfs_synth.std()], 
                   fmt='none', capsize=5, color='black', alpha=0.7)
    
    axes[0].set_xticks(x - bar_width/2)
    axes[0].set_xticklabels(['DFS', 'BFS'])
    axes[0].set_ylabel('Average Number of Recommendations', fontsize=12)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Annotate bars
    for i, mean in enumerate([dfs_synth.mean(), bfs_synth.mean()]):
        axes[0].text(i - bar_width/2, mean + 0.5, f"{mean:.1f}", ha='center', fontsize=10)
    
    # Facebook network plot
    axes[1].set_title('Recommendation Count: Real Facebook Network', fontsize=14)
    axes[1].bar(x - bar_width/2, [dfs_fb.mean(), bfs_fb.mean()], bar_width,
              label='Mean Count', color=[colors[1], colors[0]])
    
    # Add error bars
    axes[1].errorbar(x - bar_width/2, [dfs_fb.mean(), bfs_fb.mean()], 
                   yerr=[dfs_fb.std(), bfs_fb.std()], 
                   fmt='none', capsize=5, color='black', alpha=0.7)
    
    axes[1].set_xticks(x - bar_width/2)
    axes[1].set_xticklabels(['DFS', 'BFS'])
    axes[1].set_ylabel('Average Number of Recommendations', fontsize=12)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Annotate bars
    for i, mean in enumerate([dfs_fb.mean(), bfs_fb.mean()]):
        axes[1].text(i - bar_width/2, mean + 30, f"{mean:.1f}", ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Recommendation count comparison saved to {output_path}")

def create_similarity_distribution(synthetic_df, facebook_df, output_path):
    """Create enhanced similarity distribution visualization"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Prepare data
    sim_synth = synthetic_df[synthetic_df['algorithm'] == 'DFS']['similarity'].dropna()
    sim_fb = facebook_df[facebook_df['algorithm'] == 'DFS']['similarity'].dropna()
    
    # Colors
    colors = ['#3498db', '#e74c3c']  # Blue for BFS, Red for DFS
    
    # Synthetic network plot
    axes[0].set_title('Recommendation Similarity: Synthetic Network', fontsize=14)
    axes[0].hist(sim_synth, bins=10, alpha=0.7, color=colors[0], edgecolor='black')
    axes[0].set_xlabel('Jaccard Similarity Score', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # Add mean line
    axes[0].axvline(x=sim_synth.mean(), color='red', linestyle='--', linewidth=2)
    axes[0].text(sim_synth.mean() - 0.02, axes[0].get_ylim()[1] * 0.9, 
             f"Mean: {sim_synth.mean():.3f}", 
             rotation=90, va='top', ha='right', color='red',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Facebook network plot
    axes[1].set_title('Recommendation Similarity: Real Facebook Network', fontsize=14)
    axes[1].hist(sim_fb, bins=10, alpha=0.7, color=colors[1], edgecolor='black')
    axes[1].set_xlabel('Jaccard Similarity Score', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    # Add mean line
    axes[1].axvline(x=sim_fb.mean(), color='blue', linestyle='--', linewidth=2)
    axes[1].text(sim_fb.mean() + 0.02, axes[1].get_ylim()[1] * 0.9, 
             f"Mean: {sim_fb.mean():.3f}", 
             rotation=90, va='top', ha='left', color='blue',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Similarity distribution saved to {output_path}")

def create_combined_performance_visualization(synthetic_df, facebook_df, output_path):
    """Create a comprehensive performance visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Prepare data
    dfs_synth_time = synthetic_df[synthetic_df['algorithm'] == 'DFS']['execution_time']
    bfs_synth_time = synthetic_df[synthetic_df['algorithm'] == 'BFS']['execution_time']
    dfs_fb_time = facebook_df[facebook_df['algorithm'] == 'DFS']['execution_time']
    bfs_fb_time = facebook_df[facebook_df['algorithm'] == 'BFS']['execution_time']
    
    dfs_synth_recs = synthetic_df[synthetic_df['algorithm'] == 'DFS']['recommendation_count']
    bfs_synth_recs = synthetic_df[synthetic_df['algorithm'] == 'BFS']['recommendation_count']
    dfs_fb_recs = facebook_df[facebook_df['algorithm'] == 'DFS']['recommendation_count']
    bfs_fb_recs = facebook_df[facebook_df['algorithm'] == 'BFS']['recommendation_count']
    
    sim_synth = synthetic_df[synthetic_df['algorithm'] == 'DFS']['similarity'].dropna()
    sim_fb = facebook_df[facebook_df['algorithm'] == 'DFS']['similarity'].dropna()
    
    # Colors
    colors = ['#3498db', '#e74c3c']  # Blue for BFS, Red for DFS
    
    # Top left: Execution time comparison (log scale)
    axes[0, 0].set_title('Execution Time Comparison', fontsize=14)
    
    # Position for grouped bar chart
    bar_width = 0.35
    index = np.array([0, 1])
    
    # Plot execution times
    axes[0, 0].bar(index - bar_width/2, [dfs_synth_time.mean(), dfs_fb_time.mean()], 
                 bar_width, label='DFS', color=colors[1], alpha=0.7)
    axes[0, 0].bar(index + bar_width/2, [bfs_synth_time.mean(), bfs_fb_time.mean()], 
                 bar_width, label='BFS', color=colors[0], alpha=0.7)
    
    # Set labels and log scale
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_xticks(index)
    axes[0, 0].set_xticklabels(['Synthetic Network', 'Facebook Network'])
    axes[0, 0].set_ylabel('Execution Time (seconds, log scale)', fontsize=12)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Annotate with speedup percentage
    synth_speedup = ((dfs_synth_time.mean() - bfs_synth_time.mean()) / bfs_synth_time.mean()) * 100
    fb_speedup = ((dfs_fb_time.mean() - bfs_fb_time.mean()) / bfs_fb_time.mean()) * 100
    
    axes[0, 0].text(0, dfs_synth_time.mean(), f"{dfs_synth_time.mean():.6f}s", 
                  ha='center', va='bottom', fontsize=10)
    axes[0, 0].text(0, bfs_synth_time.mean(), f"{bfs_synth_time.mean():.6f}s", 
                  ha='center', va='bottom', fontsize=10)
    axes[0, 0].text(1, dfs_fb_time.mean(), f"{dfs_fb_time.mean():.6f}s", 
                  ha='center', va='bottom', fontsize=10)
    axes[0, 0].text(1, bfs_fb_time.mean(), f"{bfs_fb_time.mean():.6f}s", 
                  ha='center', va='bottom', fontsize=10)
    
    # Top right: Recommendation count comparison
    axes[0, 1].set_title('Recommendation Count Comparison', fontsize=14)
    
    # Plot recommendation counts
    axes[0, 1].bar(index - bar_width/2, [dfs_synth_recs.mean(), dfs_fb_recs.mean()], 
                 bar_width, label='DFS', color=colors[1], alpha=0.7)
    axes[0, 1].bar(index + bar_width/2, [bfs_synth_recs.mean(), bfs_fb_recs.mean()], 
                 bar_width, label='BFS', color=colors[0], alpha=0.7)
    
    # Set labels
    axes[0, 1].set_xticks(index)
    axes[0, 1].set_xticklabels(['Synthetic Network', 'Facebook Network'])
    axes[0, 1].set_ylabel('Number of Recommendations', fontsize=12)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Add data point labels
    axes[0, 1].text(0 - bar_width/2, dfs_synth_recs.mean(), f"{dfs_synth_recs.mean():.1f}", 
                  ha='center', va='bottom', fontsize=10)
    axes[0, 1].text(0 + bar_width/2, bfs_synth_recs.mean(), f"{bfs_synth_recs.mean():.1f}", 
                  ha='center', va='bottom', fontsize=10)
    axes[0, 1].text(1 - bar_width/2, dfs_fb_recs.mean(), f"{dfs_fb_recs.mean():.1f}", 
                  ha='center', va='bottom', fontsize=10)
    axes[0, 1].text(1 + bar_width/2, bfs_fb_recs.mean(), f"{bfs_fb_recs.mean():.1f}", 
                  ha='center', va='bottom', fontsize=10)
    
    # Bottom left: Similarity distribution as box plot
    axes[1, 0].set_title('Recommendation Similarity Distribution', fontsize=14)
    
    bp = axes[1, 0].boxplot([sim_synth, sim_fb], patch_artist=True)
    
    # Change box colors
    bp['boxes'][0].set(facecolor=colors[0], alpha=0.7)
    bp['boxes'][1].set(facecolor=colors[1], alpha=0.7)
    
    # Set labels
    axes[1, 0].set_xticklabels(['Synthetic Network', 'Facebook Network'])
    axes[1, 0].set_ylabel('Jaccard Similarity Score', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Add mean lines
    axes[1, 0].axhline(y=sim_synth.mean(), color='blue', linestyle='--', linewidth=1, 
                     xmin=0.05, xmax=0.25)
    axes[1, 0].axhline(y=sim_fb.mean(), color='red', linestyle='--', linewidth=1,
                     xmin=0.75, xmax=0.95)
    
    # Add mean annotations
    axes[1, 0].text(0.9, sim_synth.mean(), f"Mean: {sim_synth.mean():.3f}", 
                  va='center', ha='center', fontsize=10,
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    axes[1, 0].text(1.9, sim_fb.mean(), f"Mean: {sim_fb.mean():.3f}", 
                  va='center', ha='center', fontsize=10,
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Bottom right: Performance summary with speedup % and recommendation difference %
    axes[1, 1].axis('off')  # Turn off axis for text summary
    
    # Calculate key metrics
    synth_rec_diff = ((bfs_synth_recs.mean() - dfs_synth_recs.mean()) / dfs_synth_recs.mean()) * 100
    fb_rec_diff = ((bfs_fb_recs.mean() - dfs_fb_recs.mean()) / dfs_fb_recs.mean()) * 100
    
    summary_text = (
        "PERFORMANCE SUMMARY\n"
        "===================\n\n"
        "Synthetic Network:\n"
        f"• BFS is {abs(synth_speedup):.1f}% {'faster' if synth_speedup > 0 else 'slower'} than DFS\n"
        f"• BFS provides {abs(synth_rec_diff):.1f}% {'more' if synth_rec_diff > 0 else 'fewer'} recommendations\n"
        f"• Mean recommendation similarity: {sim_synth.mean():.3f}\n\n"
        "Real Facebook Network:\n"
        f"• BFS is {abs(fb_speedup):.1f}% {'faster' if fb_speedup > 0 else 'slower'} than DFS\n"
        f"• BFS provides {abs(fb_rec_diff):.1f}% {'more' if fb_rec_diff > 0 else 'fewer'} recommendations\n"
        f"• Mean recommendation similarity: {sim_fb.mean():.3f}\n\n"
        "Key Finding:\n"
        "• The performance advantage of BFS is much more pronounced\n"
        "  in synthetic networks compared to real social networks\n"
        "• Both algorithms produce highly similar recommendations\n"
        "• Real networks generate significantly more recommendations\n"
        f"  ({dfs_fb_recs.mean():.1f} vs {dfs_synth_recs.mean():.1f} for DFS)\n"
    )
    
    axes[1, 1].text(0.05, 0.95, summary_text, 
                  transform=axes[1, 1].transAxes, 
                  fontsize=12, 
                  va='top', ha='left',
                  fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Combined performance visualization saved to {output_path}")

def main():
    """Main function to create enhanced visualizations"""
    # Load data
    synthetic_df = load_results('results/benchmark_results.csv')
    facebook_df = load_results('results/facebook/benchmark_results.csv')
    
    if synthetic_df is None or facebook_df is None:
        print("Error: Could not load benchmark results")
        return
    
    # Create enhanced visualizations
    create_execution_time_comparison(
        synthetic_df, facebook_df, 'results/enhanced/execution_time_comparison.png')
    
    create_recommendation_count_comparison(
        synthetic_df, facebook_df, 'results/enhanced/recommendation_count_comparison.png')
    
    create_similarity_distribution(
        synthetic_df, facebook_df, 'results/enhanced/similarity_distribution.png')
    
    create_combined_performance_visualization(
        synthetic_df, facebook_df, 'results/enhanced/combined_performance.png')
    
    print("All enhanced visualizations created successfully!")

if __name__ == "__main__":
    main()