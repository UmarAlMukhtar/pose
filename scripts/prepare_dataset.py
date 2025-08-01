"""
Data Preparation Script for YOLO11 Pose Estimation

This script prepares the dataset for training by:
1. Organizing images and labels
2. Splitting data into train/val/test sets
3. Creating YOLO dataset configuration
4. Validating annotations
"""

import os
import shutil
import random
import json
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2
import numpy as np


class DatasetPreparer:
    def __init__(self, frames_dir, annotations_dir, output_dir, train_ratio=0.7, val_ratio=0.2):
        """
        Initialize Dataset Preparer
        
        Args:
            frames_dir (str): Directory containing extracted frames
            annotations_dir (str): Directory containing YOLO format annotations
            output_dir (str): Output directory for organized dataset
            train_ratio (float): Ratio for training set
            val_ratio (float): Ratio for validation set (test = 1 - train - val)
        """
        self.frames_dir = Path(frames_dir)
        self.annotations_dir = Path(annotations_dir)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1 - train_ratio - val_ratio
        
        # YOLO11 pose keypoints (17 keypoints for human pose)
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # Important keypoints for throwing detection
        self.throwing_keypoints = [
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip'
        ]
    
    def find_image_annotation_pairs(self):
        """Find matching image-annotation pairs"""
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(self.frames_dir.glob(f'**/*{ext}'))
            image_files.extend(self.frames_dir.glob(f'**/*{ext.upper()}'))
        
        # Find corresponding annotation files
        pairs = []
        missing_annotations = []
        
        for img_path in image_files:
            # Look for corresponding .txt annotation file
            annotation_name = img_path.stem + '.txt'
            annotation_path = self.annotations_dir / annotation_name
            
            if annotation_path.exists():
                pairs.append((img_path, annotation_path))
            else:
                missing_annotations.append(img_path)
        
        print(f"Found {len(pairs)} image-annotation pairs")
        if missing_annotations:
            print(f"Warning: {len(missing_annotations)} images without annotations")
        
        return pairs, missing_annotations
    
    def validate_annotations(self, pairs):
        """Validate YOLO format annotations"""
        valid_pairs = []
        invalid_pairs = []
        
        print("Validating annotations...")
        
        for img_path, ann_path in tqdm(pairs):
            try:
                # Read image to get dimensions
                img = cv2.imread(str(img_path))
                if img is None:
                    invalid_pairs.append((img_path, ann_path, "Cannot read image"))
                    continue
                
                img_height, img_width = img.shape[:2]
                
                # Read annotation
                with open(ann_path, 'r') as f:
                    lines = f.readlines()
                
                valid_annotation = True
                for line in lines:
                    parts = line.strip().split()
                    
                    # YOLO pose format: class_id x_center y_center width height x1 y1 v1 x2 y2 v2 ... x17 y17 v17
                    # Where xi, yi are keypoint coordinates and vi is visibility (0=not visible, 1=visible, 2=occluded)
                    if len(parts) != 56:  # 1 class_id + 4 bbox + 51 keypoints (17 * 3)
                        valid_annotation = False
                        break
                    
                    # Validate class_id
                    try:
                        class_id = int(parts[0])
                        if class_id != 0:  # Assuming person class is 0
                            valid_annotation = False
                            break
                    except ValueError:
                        valid_annotation = False
                        break
                    
                    # Validate bbox coordinates (should be normalized 0-1)
                    try:
                        x_center, y_center, width, height = map(float, parts[1:5])
                        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                               0 <= width <= 1 and 0 <= height <= 1):
                            valid_annotation = False
                            break
                    except ValueError:
                        valid_annotation = False
                        break
                    
                    # Validate keypoints
                    try:
                        keypoints = list(map(float, parts[5:]))
                        for i in range(0, len(keypoints), 3):
                            x, y, v = keypoints[i], keypoints[i+1], keypoints[i+2]
                            if not (0 <= x <= 1 and 0 <= y <= 1 and v in [0, 1, 2]):
                                valid_annotation = False
                                break
                    except (ValueError, IndexError):
                        valid_annotation = False
                        break
                
                if valid_annotation:
                    valid_pairs.append((img_path, ann_path))
                else:
                    invalid_pairs.append((img_path, ann_path, "Invalid annotation format"))
                    
            except Exception as e:
                invalid_pairs.append((img_path, ann_path, str(e)))
        
        print(f"Valid pairs: {len(valid_pairs)}")
        print(f"Invalid pairs: {len(invalid_pairs)}")
        
        if invalid_pairs:
            # Save invalid pairs report
            invalid_report_path = self.output_dir / 'invalid_annotations_report.txt'
            with open(invalid_report_path, 'w') as f:
                f.write("Invalid Image-Annotation Pairs Report\n")
                f.write("=" * 50 + "\n\n")
                for img_path, ann_path, error in invalid_pairs:
                    f.write(f"Image: {img_path}\n")
                    f.write(f"Annotation: {ann_path}\n")
                    f.write(f"Error: {error}\n")
                    f.write("-" * 30 + "\n")
            
            print(f"Invalid pairs report saved to: {invalid_report_path}")
        
        return valid_pairs
    
    def split_dataset(self, pairs):
        """Split dataset into train/val/test sets"""
        print(f"Splitting dataset: {self.train_ratio:.1%} train, {self.val_ratio:.1%} val, {self.test_ratio:.1%} test")
        
        # First split: separate test set
        train_val_pairs, test_pairs = train_test_split(
            pairs, test_size=self.test_ratio, random_state=42
        )
        
        # Second split: separate train and validation
        val_size = self.val_ratio / (self.train_ratio + self.val_ratio)
        train_pairs, val_pairs = train_test_split(
            train_val_pairs, test_size=val_size, random_state=42
        )
        
        print(f"Train: {len(train_pairs)} samples")
        print(f"Validation: {len(val_pairs)} samples")
        print(f"Test: {len(test_pairs)} samples")
        
        return train_pairs, val_pairs, test_pairs
    
    def copy_files(self, pairs, split_name):
        """Copy files to appropriate directories"""
        images_dir = self.output_dir / split_name / 'images'
        labels_dir = self.output_dir / split_name / 'labels'
        
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Copying {split_name} files...")
        
        for img_path, ann_path in tqdm(pairs):
            # Copy image
            img_dest = images_dir / img_path.name
            shutil.copy2(img_path, img_dest)
            
            # Copy annotation
            ann_dest = labels_dir / ann_path.name
            shutil.copy2(ann_path, ann_dest)
    
    def create_dataset_config(self):
        """Create YOLO dataset configuration file"""
        config = {
            'path': str(self.output_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'names': {0: 'person'},
            'nc': 1,  # number of classes
            'kpt_shape': [17, 3],  # 17 keypoints, each with x, y, visibility
        }
        
        config_path = self.output_dir / 'dataset.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"Dataset configuration saved to: {config_path}")
        return config_path
    
    def analyze_throwing_poses(self, pairs):
        """Analyze poses to identify throwing characteristics"""
        print("Analyzing throwing poses...")
        
        throwing_stats = {
            'total_samples': len(pairs),
            'pose_characteristics': {},
            'keypoint_visibility': {kp: 0 for kp in self.keypoint_names}
        }
        
        for img_path, ann_path in tqdm(pairs):
            with open(ann_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 56:  # Valid pose annotation
                    keypoints = list(map(float, parts[5:]))
                    
                    # Extract key throwing-related keypoints
                    for i, kp_name in enumerate(self.keypoint_names):
                        x, y, v = keypoints[i*3], keypoints[i*3+1], keypoints[i*3+2]
                        if v > 0:  # Visible keypoint
                            throwing_stats['keypoint_visibility'][kp_name] += 1
        
        # Calculate visibility percentages
        for kp_name in throwing_stats['keypoint_visibility']:
            count = throwing_stats['keypoint_visibility'][kp_name]
            throwing_stats['keypoint_visibility'][kp_name] = {
                'count': count,
                'percentage': (count / throwing_stats['total_samples']) * 100
            }
        
        # Save analysis
        analysis_path = self.output_dir / 'pose_analysis.json'
        with open(analysis_path, 'w') as f:
            json.dump(throwing_stats, f, indent=2)
        
        print(f"Pose analysis saved to: {analysis_path}")
        
        # Print key insights
        print("\nKey Pose Analysis Insights:")
        for kp_name in self.throwing_keypoints:
            stats = throwing_stats['keypoint_visibility'][kp_name]
            print(f"  {kp_name}: {stats['percentage']:.1f}% visible")
    
    def create_training_instructions(self):
        """Create detailed training instructions"""
        instructions = f"""# YOLO11 Pose Estimation Training Instructions

## Dataset Overview
- Training samples: Check train/images/ directory
- Validation samples: Check val/images/ directory  
- Test samples: Check test/images/ directory

## Key Files
- `dataset.yaml`: Main dataset configuration file
- `pose_analysis.json`: Analysis of pose characteristics
- `invalid_annotations_report.txt`: Issues found during validation

## Training Command
```bash
# Basic training
yolo pose train data={self.output_dir.absolute()}/dataset.yaml model=yolo11n-pose.pt epochs=100 imgsz=640

# Advanced training with custom parameters
yolo pose train data={self.output_dir.absolute()}/dataset.yaml model=yolo11n-pose.pt epochs=200 imgsz=640 batch=16 lr0=0.01 lrf=0.01 mosaic=1.0 mixup=0.1

# Resume training
yolo pose train resume=path/to/last.pt
```

## Model Sizes Available
- yolo11n-pose.pt: Nano (fastest, least accurate)
- yolo11s-pose.pt: Small
- yolo11m-pose.pt: Medium  
- yolo11l-pose.pt: Large
- yolo11x-pose.pt: Extra Large (slowest, most accurate)

## Key Hyperparameters for Pose Estimation
- `epochs`: Number of training epochs (start with 100-200)
- `imgsz`: Input image size (640 is standard)
- `batch`: Batch size (adjust based on GPU memory)
- `lr0`: Initial learning rate (0.01 is typical)
- `lrf`: Final learning rate (lr0 * lrf)
- `mosaic`: Mosaic augmentation probability
- `mixup`: Mixup augmentation probability
- `kobj`: Keypoint objectness loss weight
- `pose`: Pose loss weight

## Monitoring Training
Use Weights & Biases for monitoring:
```bash
pip install wandb
wandb login
# Training will automatically log to W&B
```

## Evaluation
```bash
# Evaluate on validation set
yolo pose val model=path/to/best.pt data={self.output_dir.absolute()}/dataset.yaml

# Evaluate on test set
yolo pose val model=path/to/best.pt data={self.output_dir.absolute()}/dataset.yaml split=test
```

## Inference
```bash
# Single image
yolo pose predict model=path/to/best.pt source=path/to/image.jpg

# Video
yolo pose predict model=path/to/best.pt source=path/to/video.mp4

# Webcam
yolo pose predict model=path/to/best.pt source=0
```

## Tips for Better Performance
1. **Data Quality**: Ensure accurate keypoint annotations
2. **Augmentation**: Use appropriate augmentations for your use case
3. **Class Balance**: Ensure balanced representation of throwing poses
4. **Validation**: Monitor validation metrics to avoid overfitting
5. **Keypoint Quality**: Focus on key throwing-related keypoints
6. **Multi-scale Training**: Use different image sizes during training

## Troubleshooting
- **GPU Memory Issues**: Reduce batch size or image size
- **Poor Performance**: Check annotation quality and increase dataset size
- **Overfitting**: Reduce model complexity or increase augmentation
- **Slow Training**: Use smaller model or reduce image size

## Expected Training Time
- Small dataset (< 1000 images): 1-3 hours
- Medium dataset (1000-5000 images): 3-12 hours  
- Large dataset (> 5000 images): 12+ hours

Times vary significantly based on GPU and dataset complexity.
"""
        
        instructions_path = self.output_dir / 'TRAINING_INSTRUCTIONS.md'
        with open(instructions_path, 'w') as f:
            f.write(instructions)
        
        print(f"Training instructions saved to: {instructions_path}")
    
    def prepare_dataset(self):
        """Main method to prepare the dataset"""
        print("Starting dataset preparation...")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find image-annotation pairs
        pairs, missing = self.find_image_annotation_pairs()
        
        if not pairs:
            print("No valid image-annotation pairs found!")
            return
        
        # Validate annotations
        valid_pairs = self.validate_annotations(pairs)
        
        if not valid_pairs:
            print("No valid annotations found!")
            return
        
        # Split dataset
        train_pairs, val_pairs, test_pairs = self.split_dataset(valid_pairs)
        
        # Copy files to appropriate directories
        self.copy_files(train_pairs, 'train')
        self.copy_files(val_pairs, 'val')
        if test_pairs:
            self.copy_files(test_pairs, 'test')
        
        # Create dataset configuration
        self.create_dataset_config()
        
        # Analyze poses
        self.analyze_throwing_poses(valid_pairs)
        
        # Create training instructions
        self.create_training_instructions()
        
        print(f"\nDataset preparation complete!")
        print(f"Dataset location: {self.output_dir}")
        print(f"Ready for training with YOLO11!")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare dataset for YOLO11 pose estimation training')
    parser.add_argument('--frames_dir', default='data/frames',
                       help='Directory containing extracted frames')
    parser.add_argument('--annotations_dir', default='data/annotations',
                       help='Directory containing YOLO format annotations')
    parser.add_argument('--output_dir', default='data',
                       help='Output directory for organized dataset')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Ratio for training set')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                       help='Ratio for validation set')
    
    args = parser.parse_args()
    
    # Create dataset preparer
    preparer = DatasetPreparer(
        frames_dir=args.frames_dir,
        annotations_dir=args.annotations_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio
    )
    
    # Prepare dataset
    preparer.prepare_dataset()


if __name__ == "__main__":
    main()
