import argparse
import os
import yaml
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLOv9 for fish detection')
    parser.add_argument('--data', type=str, default='data/fishcounting.yaml', help='Dataset configuration file')
    parser.add_argument('--cfg', type=str, default='models/yolov9s.yaml', help='Model configuration file')
    parser.add_argument('--weights', type=str, default='', help='Initial weights path')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train for')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size for training')
    parser.add_argument('--device', type=str, default='', help='Device to train on (cpu, 0, 0,1,2,3)')
    parser.add_argument('--workers', type=int, default=8, help='Number of worker threads')
    parser.add_argument('--project', type=str, default='runs/train', help='Project name')
    parser.add_argument('--name', type=str, default='exp', help='Experiment name')
    parser.add_argument('--exist-ok', action='store_true', help='Overwrite existing experiment')
    parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint')
    return parser.parse_args()

def validate_dataset(data_yaml):
    """
    Validate the dataset YAML configuration
    """
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Dataset configuration not found: {data_yaml}")
    
    try:
        with open(data_yaml, 'r') as f:
            data_config = yaml.safe_load(f)
        
        required_keys = ['path', 'train', 'val', 'names']
        for key in required_keys:
            if key not in data_config:
                raise ValueError(f"Missing required key in dataset config: {key}")
        
        # Check that the directories exist
        for split in ['train', 'val']:
            split_path = data_config[split]
            if not os.path.exists(os.path.join(data_config['path'], split_path)):
                print(f"Warning: {split} path does not exist: {os.path.join(data_config['path'], split_path)}")
        
        # Validate class names
        if not isinstance(data_config['names'], dict) and not isinstance(data_config['names'], list):
            raise ValueError("Class names must be a dictionary or list")
        
        print(f"Dataset validation passed. Found {len(data_config['names'])} classes.")
        
    except Exception as e:
        print(f"Error validating dataset: {e}")
        raise

def train(args):
    """
    Train the YOLOv9 model
    """
    # Validate dataset configuration
    validate_dataset(args.data)
    
    # Load base model
    if args.weights:
        print(f"Loading weights from {args.weights}")
        model = YOLO(args.weights)
    else:
        print(f"Creating new model from {args.cfg}")
        model = YOLO(args.cfg)
    
    # Train the model
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        resume=args.resume
    )
    
    # Validate the trained model
    metrics = model.val()
    
    print(f"Training complete. Model saved to {os.path.join(args.project, args.name)}")
    print(f"Validation metrics: mAP@0.5={metrics.box.map50:.4f}, mAP@0.5:0.95={metrics.box.map:.4f}")
    
    return results

def main():
    args = parse_args()
    train(args)

if __name__ == "__main__":
    main()
