import argparse
import cv2
import numpy as np
from ultralytics import YOLO
from matplotlib.cm import get_cmap
import os

def parse_args():
    parser = argparse.ArgumentParser(description='DeepFish Detection and Analysis')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model', type=str, default='models/best.pt', help='Path to YOLOv9 model')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.4, help='IoU threshold')
    parser.add_argument('--pixel-to-meter', type=float, default=0.01, help='Pixel to meter conversion ratio')
    parser.add_argument('--output', type=str, default='output.jpg', help='Path to output image')
    return parser.parse_args()

def detect_and_analyze(image_path, model_path, conf_thresh, iou_thresh, pixel_to_meter, output_path):
    # Load model
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    # Load image
    print(f"Processing image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run inference
    results = model.predict(source=image_rgb, conf=conf_thresh, iou=iou_thresh)
    
    # Extract bounding boxes
    detections = results[0].boxes.xyxy.cpu().numpy()
    fish_count = len(detections)
    print(f"Detected {fish_count} fish")
    
    # Generate heatmap
    height, width, _ = image.shape
    heatmap = np.zeros((height, width), dtype=np.float32)
    
    # Update heatmap and calculate distances
    for box in detections:
        x_min, y_min, x_max, y_max = map(int, box)
        heatmap[y_min:y_max, x_min:x_max] += 1
    
    # Normalize and colorize heatmap
    heatmap = cv2.normalize(heatmap, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cmap = get_cmap('inferno')
    heatmap_color = cmap(heatmap)[:, :, :3]
    heatmap_color = (heatmap_color * 255).astype(np.uint8)
    heatmap_overlay = cv2.addWeighted(image_rgb, 0.6, heatmap_color, 0.4, 0)
    
    # Annotate fish count
    cv2.putText(heatmap_overlay, f'Fish Count: {fish_count}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Draw bounding boxes, distance lines, and measurements
    for box in detections:
        x_min, y_min, x_max, y_max = map(int, box)
        center_x = (x_min + x_max) // 2
        distance = y_min * pixel_to_meter
        
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
    
    # Save output
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(heatmap_overlay, cv2.COLOR_RGB2BGR))
    print(f"Output saved to {output_path}")
    
    # Display output (if in interactive environment)
    try:
        cv2.imshow("DeepFish Detection", cv2.cvtColor(heatmap_overlay, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        pass
    
    return {
        "fish_count": fish_count,
        "detections": detections.tolist(),
        "output_path": output_path
    }

def main():
    args = parse_args()
    detect_and_analyze(
        args.image, 
        args.model, 
        args.conf, 
        args.iou, 
        args.pixel_to_meter,
        args.output
    )

if __name__ == "__main__":
    main()
