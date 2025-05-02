import argparse
import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import json
import datetime
from utils import create_heatmap, save_results

def parse_args():
    parser = argparse.ArgumentParser(description='Batch process images with DeepFish')
    parser.add_argument('--input-dir', type=str, required=True, help='Directory containing input images')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save output images and results')
    parser.add_argument('--model', type=str, default='models/best.pt', help='Path to YOLOv9 model')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.4, help='IoU threshold')
    parser.add_argument('--pixel-to-meter', type=float, default=0.01, help='Pixel to meter conversion ratio')
    parser.add_argument('--file-ext', type=str, default='jpg,jpeg,png', help='File extensions to process (comma separated)')
    parser.add_argument('--save-json', action='store_true', help='Save results to JSON files')
    return parser.parse_args()

def process_image(image_path, model, conf_thresh, iou_thresh, pixel_to_meter):
    """
    Process a single image with the model
    
    Args:
        image_path: Path to input image
        model: Loaded YOLO model
        conf_thresh: Confidence threshold
        iou_thresh: IoU threshold
        pixel_to_meter: Pixel to meter conversion ratio
        
    Returns:
        Processed image, fish count, and detection data
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run inference
    results = model.predict(source=image_rgb, conf=conf_thresh, iou=iou_thresh)
    
    # Extract bounding boxes
    detections = results[0].boxes.xyxy.cpu().numpy()
    
    # Create heatmap and annotations
    heatmap_overlay, fish_count, distance_data = create_heatmap(image_rgb, detections, pixel_to_meter)
    
    return heatmap_overlay, fish_count, distance_data

def batch_process(input_dir, output_dir, model_path, conf_thresh, iou_thresh, pixel_to_meter, file_extensions, save_json):
    """
    Process all images in a directory
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save output images
        model_path: Path to YOLOv9 model
        conf_thresh: Confidence threshold
        iou_thresh: IoU threshold
        pixel_to_meter: Pixel to meter conversion ratio
        file_extensions: List of file extensions to process
        save_json: Whether to save results to JSON files
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    if save_json:
        os.makedirs(os.path.join(output_dir, 'json'), exist_ok=True)
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    # Get list of image files
    image_files = []
    for ext in file_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, f"*.{ext}")))
    
    if not image_files:
        print(f"No images with extensions {file_extensions} found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    all_results = []
    
    for image_path in tqdm(image_files, desc="Processing images"):
        try:
            # Process image
            heatmap_overlay, fish_count, distance_data = process_image(
                image_path, model, conf_thresh, iou_thresh, pixel_to_meter
            )
            
            # Save output image
            image_name = os.path.basename(image_path)
            output_image_path = os.path.join(output_dir, 'images', image_name)
            cv2.imwrite(output_image_path, cv2.cvtColor(heatmap_overlay, cv2.COLOR_RGB2BGR))
            
            # Save JSON results if requested
            if save_json:
                results_dir = os.path.join(output_dir, 'json')
                json_path = save_results(results_dir, image_name, fish_count, distance_data)
                
                # Add to all results for summary
                with open(json_path, 'r') as f:
                    result_data = json.load(f)
                    all_results.append(result_data)
        
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    # Generate summary
    if save_json and all_results:
        summary = {
            'total_images': len(image_files),
            'successful_detections': len(all_results),
            'total_fish_detected': sum(r['fish_count'] for r in all_results),
            'average_fish_per_image': sum(r['fish_count'] for r in all_results) / len(all_results) if all_results else 0,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        with open(os.path.join(output_dir, 'batch_summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)
    
    print(f"Batch processing complete. Results saved to {output_dir}")

def main():
    args = parse_args()
    
    # Split file extensions
    file_extensions = args.file_ext.split(',')
    
    # Run batch processing
    batch_process(
        args.input_dir,
        args.output_dir,
        args.model,
        args.conf,
        args.iou,
        args.pixel_to_meter,
        file_extensions,
        args.save_json
    )

if __name__ == "__main__":
    main()
