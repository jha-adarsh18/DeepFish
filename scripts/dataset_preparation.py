import os
import argparse
import glob
import shutil
import random
import cv2
import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET
import json

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare Caltech Fish Counting Dataset for YOLOv9 Training')
    parser.add_argument('--source', type=str, required=True, help='Source directory containing raw dataset')
    parser.add_argument('--dest', type=str, required=True, help='Destination directory for YOLO formatted dataset')
    parser.add_argument('--split', type=float, default=0.8, help='Train/validation split ratio')
    parser.add_argument('--convert-annotations', action='store_true', help='Convert annotations to YOLO format')
    return parser.parse_args()

def create_dataset_structure(dest_dir):
    """
    Create YOLO dataset directory structure
    """
    # Create main directories
    os.makedirs(dest_dir, exist_ok=True)
    
    # Create subdirectories
    for subdir in ['images/train', 'images/val', 'images/test', 'labels/train', 'labels/val', 'labels/test']:
        os.makedirs(os.path.join(dest_dir, subdir), exist_ok=True)
    
    print(f"Created directory structure in {dest_dir}")

def convert_to_yolo_format(annotation_path, image_width, image_height):
    """
    Convert annotation to YOLO format
    
    Args:
        annotation_path: Path to XML annotation file
        image_width: Width of the image
        image_height: Height of the image
        
    Returns:
        List of YOLO formatted annotations [class_id, x_center, y_center, width, height]
    """
    try:
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        
        yolo_annotations = []
        
        for obj in root.findall('./object'):
            # Get class_id (assuming fish is class 0)
            class_id = 0
            
            # Get bounding box coordinates
            bbox = obj.find('bndbox')
            if bbox is None:
                continue
                
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            # Convert to YOLO format (normalized coordinates)
            x_center = (xmin + xmax) / 2 / image_width
            y_center = (ymin + ymax) / 2 / image_height
            width = (xmax - xmin) / image_width
            height = (ymax - ymin) / image_height
            
            # Add annotation
            yolo_annotations.append([class_id, x_center, y_center, width, height])
        
        return yolo_annotations
    
    except Exception as e:
        print(f"Error processing {annotation_path}: {e}")
        return []

def process_dataset(source_dir, dest_dir, split_ratio, convert_annotations):
    """
    Process dataset from source directory to YOLO format
    """
    # Get all image files
    image_extensions = ['jpg', 'jpeg', 'png']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(source_dir, f"**/*.{ext}"), recursive=True))
    
    if not image_files:
        raise ValueError(f"No image files found in {source_dir}")
    
    print(f"Found {len(image_files)} images in {source_dir}")
    
    # Shuffle and split data
    random.shuffle(image_files)
    split_idx = int(len(image_files) * split_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    print(f"Split dataset: {len(train_files)} train, {len(val_files)} validation")
    
    # Process images and annotations
    for subset_files, subset_name in [(train_files, 'train'), (val_files, 'val')]:
        for img_path in tqdm(subset_files, desc=f"Processing {subset_name} set"):
            # Determine destination paths
            img_filename = os.path.basename(img_path)
            img_dest = os.path.join(dest_dir, 'images', subset_name, img_filename)
            
            # Copy image
            shutil.copy2(img_path, img_dest)
            
            # Process annotations if required
            if convert_annotations:
                # Determine annotation path (assuming same basename with .xml extension)
                annotation_path = os.path.splitext(img_path)[0] + '.xml'
                
                if os.path.exists(annotation_path):
                    # Get image dimensions
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Warning: Could not read image {img_path}")
                        continue
                    
                    height, width = img.shape[:2]
                    
                    # Convert annotations
                    yolo_annotations = convert_to_yolo_format(annotation_path, width, height)
                    
                    # Save YOLO format annotations
                    if yolo_annotations:
                        label_filename = os.path.splitext(img_filename)[0] + '.txt'
                        label_dest = os.path.join(dest_dir, 'labels', subset_name, label_filename)
                        
                        with open(label_dest, 'w') as f:
                            for ann in yolo_annotations:
                                f.write(f"{ann[0]} {ann[1]} {ann[2]} {ann[3]} {ann[4]}\n")
    
    print(f"Dataset processing complete. Saved to {dest_dir}")

def main():
    args = parse_args()
    
    # Create dataset structure
    create_dataset_structure(args.dest)
    
    # Process dataset
    process_dataset(args.source, args.dest, args.split, args.convert_annotations)
    
    # Create dataset.yaml file
    dataset_yaml = {
        'path': os.path.abspath(args.dest),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': {
            0: 'fish'
        },
        'nc': 1
    }
    
    with open(os.path.join(args.dest, 'dataset.yaml'), 'w') as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)
    
    print(f"Created dataset.yaml in {args.dest}")

if __name__ == "__main__":
    main()
