import cv2
import numpy as np
from matplotlib.cm import get_cmap
import json
import os
import datetime

def create_heatmap(image, detections, pixel_to_meter=0.01):
    """
    Create a heatmap visualization of fish detections
    
    Args:
        image: Input image (RGB)
        detections: List of bounding boxes in xyxy format
        pixel_to_meter: Conversion ratio from pixels to meters
        
    Returns:
        Annotated image with heatmap overlay
    """
    height, width, _ = image.shape
    heatmap = np.zeros((height, width), dtype=np.float32)
    
    # Update heatmap based on detections
    for box in detections:
        x_min, y_min, x_max, y_max = map(int, box)
        heatmap[y_min:y_max, x_min:x_max] += 1
    
    # Normalize and colorize heatmap
    heatmap = cv2.normalize(heatmap, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cmap = get_cmap('inferno')
    heatmap_color = cmap(heatmap)[:, :, :3]
    heatmap_color = (heatmap_color * 255).astype(np.uint8)
    heatmap_overlay = cv2.addWeighted(image, 0.6, heatmap_color, 0.4, 0)
    
    # Annotate fish count
    fish_count = len(detections)
    cv2.putText(heatmap_overlay, f'Fish Count: {fish_count}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Draw bounding boxes, distance lines, and measurements
    distance_data = []
    for box in detections:
        x_min, y_min, x_max, y_max = map(int, box)
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        distance = y_min * pixel_to_meter
        
        # Save distance data
        distance_data.append({
            'box': [x_min, y_min, x_max, y_max],
            'center': [center_x, center_y],
            'distance': float(distance)
        })
        
        # Draw bounding box
        cv2.rectangle(heatmap_overlay, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        # Draw dotted line to top
        for y in range(0, y_min, 10):  # Step of 10 pixels for dots
            cv2.line(heatmap_overlay, (center_x, y), (center_x, y + 5), (255, 255, 255), 2)
        
        # Draw distance text along the line
        text_x = center_x + 5
        text_y = y_min // 2  # Midpoint of the line
        cv2.putText(heatmap_overlay, f'{distance:.2f}m', (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return heatmap_overlay, fish_count, distance_data

def save_results(output_dir, image_name, fish_count, distance_data):
    """
    Save analysis results to JSON file
    
    Args:
        output_dir: Directory to save results
        image_name: Name of the analyzed image
        fish_count: Number of fish detected
        distance_data: List of distance measurements
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create results data structure
    results = {
        'image': image_name,
        'timestamp': datetime.datetime.now().isoformat(),
        'fish_count': fish_count,
        'average_distance': np.mean([d['distance'] for d in distance_data]) if distance_data else 0,
        'distance_data': distance_data
    }
    
    # Save to JSON file
    base_name = os.path.splitext(os.path.basename(image_name))[0]
    output_file = os.path.join(output_dir, f"{base_name}_results.json")
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    return output_file

def calculate_fish_density(detections, image_shape, cell_size=50):
    """
    Calculate fish density in a grid pattern
    
    Args:
        detections: List of bounding boxes in xyxy format
        image_shape: Tuple of (height, width) of the image
        cell_size: Size of each cell in the grid (pixels)
        
    Returns:
        2D numpy array with density values
    """
    height, width = image_shape[:2]
    grid_h, grid_w = height // cell_size, width // cell_size
    
    # Initialize grid
    density_grid = np.zeros((grid_h, grid_w), dtype=np.float32)
    
    # Count fish in each grid cell
    for box in detections:
        x_min, y_min, x_max, y_max = map(int, box)
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        
        grid_x = min(center_x // cell_size, grid_w - 1)
        grid_y = min(center_y // cell_size, grid_h - 1)
        
        density_grid[grid_y, grid_x] += 1
    
    # Normalize the density grid
    if np.max(density_grid) > 0:
        density_grid = density_grid / np.max(density_grid)
    
    return density_grid

def calculate_minimum_distances(detections):
    """
    Calculate minimum distances between fish
    
    Args:
        detections: List of bounding boxes in xyxy format
        
    Returns:
        List of minimum distances for each fish
    """
    if len(detections) <= 1:
        return []
    
    # Calculate centers of each bounding box
    centers = []
    for box in detections:
        x_min, y_min, x_max, y_max = map(int, box)
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        centers.append((center_x, center_y))
    
    # Calculate minimum distance for each fish
    min_distances = []
    for i, center1 in enumerate(centers):
        distances = []
        for j, center2 in enumerate(centers):
            if i != j:
                dist = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                distances.append(dist)
        if distances:
            min_distances.append(min(distances))
    
    return min_distances
