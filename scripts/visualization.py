import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import cv2
import glob
from utils import calculate_fish_density, calculate_minimum_distances

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize DeepFish analysis results')
    parser.add_argument('--results-dir', type=str, required=True, help='Directory containing results JSON files')
    parser.add_argument('--output-dir', type=str, default='visualizations', help='Output directory for visualizations')
    parser.add_argument('--original-images', type=str, default='', help='Directory containing original images (optional)')
    return parser.parse_args()

def load_results(results_dir):
    """
    Load all results JSON files from directory
    """
    result_files = glob.glob(os.path.join(results_dir, '*_results.json'))
    
    if not result_files:
        raise ValueError(f"No result files found in {results_dir}")
    
    results = []
    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                result_data = json.load(f)
                results.append(result_data)
        except Exception as e:
            print(f"Error loading {result_file}: {e}")
    
    print(f"Loaded {len(results)} result files")
    return results

def plot_fish_count_histogram(results, output_dir):
    """
    Plot histogram of fish counts
    """
    counts = [r['fish_count'] for r in results]
    
    plt.figure(figsize=(10, 6))
    sns.histplot(counts, kde=True)
    plt.title('Distribution of Fish Counts')
    plt.xlabel('Number of Fish')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'fish_count_histogram.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved fish count histogram to {output_path}")

def plot_distance_boxplot(results, output_dir):
    """
    Plot boxplot of fish distances
    """
    all_distances = []
    for result in results:
        distances = [d['distance'] for d in result['distance_data']]
        if distances:
            all_distances.extend(distances)
    
    if not all_distances:
        print("No distance data available for boxplot")
        return
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(y=all_distances)
    plt.title('Distribution of Fish Distances')
    plt.ylabel('Distance (meters)')
    plt.grid(True, alpha=0.3)
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'fish_distance_boxplot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved fish distance boxplot to {output_path}")

def plot_density_heatmap(results, output_dir, original_images_dir=''):
    """
    Plot combined density heatmap
    """
    if not results:
        print("No results available for density heatmap")
        return
    
    # Find image dimensions from first image with available original
    sample_image = None
    if original_images_dir:
        for result in results:
            img_name = os.path.basename(result['image'])
            img_path = os.path.join(original_images_dir, img_name)
            if os.path.exists(img_path):
                sample_image = cv2.imread(img_path)
                break
    
    if sample_image is None:
        # Use standard dimensions if no image is available
        print("No original images found, using default dimensions")
        image_shape = (720, 1280)
    else:
        image_shape = sample_image.shape[:2]
    
    # Calculate combined density
    cell_size = 20
    combined_density = np.zeros((image_shape[0] // cell_size, image_shape[1] // cell_size), dtype=np.float32)
    
    for result in results:
        detections = [d['box'] for d in result['distance_data']]
        if detections:
            density = calculate_fish_density(detections, image_shape, cell_size)
            combined_density += density
    
    # Normalize combined density
    if np.max(combined_density) > 0:
        combined_density = combined_density / np.max(combined_density)
    
    # Plot heatmap
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(combined_density, cmap='inferno', cbar_kws={'label': 'Normalized Density'})
    plt.title('Combined Fish Density Heatmap')
    plt.xlabel('X Position (grid cells)')
    plt.ylabel('Y Position (grid cells)')
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'fish_density_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved fish density heatmap to {output_path}")

def plot_average_distances_over_time(results, output_dir):
    """
    Plot average distances over time
    """
    timestamps = []
    avg_distances = []
    
    for result in results:
        if 'timestamp' in result and result['distance_data']:
            timestamps.append(result['timestamp'])
            avg_distance = np.mean([d['distance'] for d in result['distance_data']])
            avg_distances.append(avg_distance)
    
    if not timestamps:
        print("No timestamp data available for time series plot")
        return
    
    # Sort by timestamp
    sorted_data = sorted(zip(timestamps, avg_distances))
    timestamps, avg_distances = zip(*sorted_data)
    
    # Convert timestamps to relative time index for better display
    time_indices = np.arange(len(timestamps))
    
    plt.figure(figsize=(12, 6))
    plt.plot(time_indices, avg_distances, marker='o', linestyle='-')
    plt.title('Average Fish Distance Over Time')
    plt.xlabel('Time Index')
    plt.ylabel('Average Distance (meters)')
    plt.grid(True, alpha=0.3)
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'avg_distance_time_series.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved average distance time series to {output_path}")

def generate_summary_statistics(results, output_dir):
    """
    Generate summary statistics and save to file
    """
    if not results:
        return
    
    # Calculate statistics
    all_counts = [r['fish_count'] for r in results]
    all_distances = []
    
    for result in results:
        distances = [d['distance'] for d in result['distance_data']]
        if distances:
            all_distances.extend(distances)
    
    # Compute statistics
    stats = {
        'total_images_analyzed': len(results),
        'total_fish_detected': sum(all_counts),
        'fish_count_statistics': {
            'min': min(all_counts),
            'max': max(all_counts),
            'mean': np.mean(all_counts),
            'median': np.median(all_counts),
            'std_dev': np.std(all_counts)
        }
    }
    
    if all_distances:
        stats['distance_statistics'] = {
            'min': min(all_distances),
            'max': max(all_distances),
            'mean': np.mean(all_distances),
            'median': np.median(all_distances),
            'std_dev': np.std(all_distances)
        }
    
    # Save statistics to file
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'summary_statistics.json')
    
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=4)
    
    print(f"Saved summary statistics to {output_path}")

def main():
    args = parse_args()
    
    # Load results
    results = load_results(args.results_dir)
    
    # Create visualizations
    plot_fish_count_histogram(results, args.output_dir)
    plot_distance_boxplot(results, args.output_dir)
    plot_density_heatmap(results, args.output_dir, args.original_images)
    plot_average_distances_over_time(results, args.output_dir)
    
    # Generate summary statistics
    generate_summary_statistics(results, args.output_dir)
    
    print(f"All visualizations saved to {args.output_dir}")

if __name__ == "__main__":
    main()
