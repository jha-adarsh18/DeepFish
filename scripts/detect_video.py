import argparse
import cv2
import numpy as np
from ultralytics import YOLO
from matplotlib.cm import get_cmap
import time
import os

def parse_args():
    parser = argparse.ArgumentParser(description='DeepFish Video Detection and Analysis')
    parser.add_argument('--source', type=str, default='0', help='Video source (0 for webcam, or path to video file)')
    parser.add_argument('--model', type=str, default='models/best.pt', help='Path to YOLOv9 model')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.4, help='IoU threshold')
    parser.add_argument('--pixel-to-meter', type=float, default=0.01, help='Pixel to meter conversion ratio')
    parser.add_argument('--output', type=str, default='', help='Path to output video file (optional)')
    parser.add_argument('--show', action='store_true', help='Show video during processing')
    return parser.parse_args()

def process_video(source, model_path, conf_thresh, iou_thresh, pixel_to_meter, output_path, show_video):
    # Load model
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    # Initialize video source
    try:
        # Check if source is a webcam index
        if source.isdigit():
            cap = cv2.VideoCapture(int(source))
        else:
            cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {source}")
    except Exception as e:
        print(f"Error opening video source: {e}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize video writer if output path is specified
    out = None
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process video frames
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Convert frame to RGB for YOLO
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run inference
        results = model.predict(source=frame_rgb, conf=conf_thresh, iou=iou_thresh)
        
        # Extract bounding boxes
        detections = results[0].boxes.xyxy.cpu().numpy()
        fish_count = len(detections)
        
        # Generate heatmap
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
        heatmap_overlay = cv2.addWeighted(frame_rgb, 0.6, heatmap_color, 0.4, 0)
        
        # Annotate fish count and FPS
        current_time = time.time()
        elapsed_time = current_time - start_time
        current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        cv2.putText(heatmap_overlay, f'Fish Count: {fish_count}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(heatmap_overlay, f'FPS: {current_fps:.1f}', (10, 70),
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
        
        # Convert back to BGR for OpenCV
        output_frame = cv2.cvtColor(heatmap_overlay, cv2.COLOR_RGB2BGR)
        
        # Write frame to output video if configured
        if out:
            out.write(output_frame)
        
        # Display frame if configured
        if show_video:
            cv2.imshow('DeepFish Detection', output_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                break
    
    # Release resources
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    
    # Print summary
    total_time = time.time() - start_time
    print(f"Processed {frame_count} frames in {total_time:.2f} seconds")
    print(f"Average FPS: {frame_count / total_time:.2f}")
    if output_path:
        print(f"Output saved to {output_path}")

def main():
    args = parse_args()
    process_video(
        args.source,
        args.model,
        args.conf,
        args.iou,
        args.pixel_to_meter,
        args.output,
        args.show
    )

if __name__ == "__main__":
    main()
