"""
Model Evaluation and Analysis Script

This script provides comprehensive evaluation of the trained pose estimation model,
including performance metrics, visualization, and analysis of throwing detection accuracy.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import yaml
from ultralytics import YOLO
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import argparse
from tqdm import tqdm


class PoseEvaluator:
    def __init__(self, model_path, dataset_config_path):
        """
        Initialize the Pose Evaluator
        
        Args:
            model_path (str): Path to trained model
            dataset_config_path (str): Path to dataset configuration
        """
        self.model_path = Path(model_path)
        self.dataset_config_path = Path(dataset_config_path)
        
        # Load model
        self.model = YOLO(str(self.model_path))
        
        # Load dataset config
        with open(self.dataset_config_path, 'r') as f:
            self.dataset_config = yaml.safe_load(f)
        
        self.data_dir = Path(self.dataset_config['path'])
        self.results_dir = self.data_dir.parent / 'results' / 'evaluation'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Keypoint information
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # Key keypoints for throwing detection
        self.throwing_keypoints = [
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist'
        ]
    
    def run_comprehensive_evaluation(self):
        """Run comprehensive evaluation on all splits"""
        print("Running comprehensive model evaluation...")
        
        results = {}
        
        # Evaluate on each split
        for split in ['train', 'val', 'test']:
            split_dir = self.data_dir / split
            if split_dir.exists() and (split_dir / 'images').exists():
                print(f"\nEvaluating on {split} set...")
                split_results = self.evaluate_split(split)
                results[split] = split_results
        
        # Save comprehensive results
        results_path = self.results_dir / 'comprehensive_evaluation.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Comprehensive evaluation results saved to: {results_path}")
        
        # Generate summary report
        self.generate_evaluation_report(results)
        
        return results
    
    def evaluate_split(self, split='val'):
        """Evaluate model on a specific data split"""
        # Run YOLO evaluation
        yolo_results = self.model.val(
            data=str(self.dataset_config_path),
            split=split,
            imgsz=640,
            batch=16,
            conf=0.25,
            iou=0.6,
            save_json=True,
            save_hybrid=True,
            plots=True,
            verbose=False
        )
        
        # Extract key metrics
        metrics = {
            'mAP50_pose': yolo_results.results_dict.get('metrics/mAP50(P)', 0),
            'mAP50_95_pose': yolo_results.results_dict.get('metrics/mAP50-95(P)', 0),
            'precision_pose': yolo_results.results_dict.get('metrics/precision(P)', 0),
            'recall_pose': yolo_results.results_dict.get('metrics/recall(P)', 0),
        }
        
        # Custom pose analysis
        custom_metrics = self.analyze_pose_performance(split)
        metrics.update(custom_metrics)
        
        return metrics
    
    def analyze_pose_performance(self, split='val'):
        """Analyze pose estimation performance in detail"""
        split_dir = self.data_dir / split
        images_dir = split_dir / 'images'
        labels_dir = split_dir / 'labels'
        
        if not images_dir.exists():
            return {}
        
        image_files = list(images_dir.glob('*.*'))
        
        keypoint_accuracies = []
        throwing_pose_detections = []
        visibility_stats = {kp: {'detected': 0, 'total': 0} for kp in self.keypoint_names}
        
        print(f"Analyzing {len(image_files)} images...")
        
        for img_path in tqdm(image_files):
            # Load ground truth
            label_path = labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue
            
            # Get predictions
            results = self.model(str(img_path), verbose=False)
            
            if not results or not results[0].keypoints:
                continue
            
            # Load ground truth keypoints
            with open(label_path, 'r') as f:
                gt_lines = f.readlines()
            
            # Analyze each detection
            for result in results:
                if result.keypoints is not None:
                    pred_keypoints = result.keypoints.xy[0].cpu().numpy()  # First detection
                    pred_conf = result.keypoints.conf[0].cpu().numpy() if result.keypoints.conf is not None else None
                    
                    # Compare with ground truth
                    for gt_line in gt_lines:
                        gt_parts = gt_line.strip().split()
                        if len(gt_parts) >= 56:  # Valid pose annotation
                            gt_keypoints = np.array(gt_parts[5:], dtype=float).reshape(-1, 3)
                            
                            # Calculate keypoint accuracy
                            kp_accuracy = self.calculate_keypoint_accuracy(pred_keypoints, gt_keypoints[:, :2])
                            keypoint_accuracies.append(kp_accuracy)
                            
                            # Analyze throwing pose characteristics
                            throwing_score = self.analyze_throwing_pose(pred_keypoints)
                            throwing_pose_detections.append(throwing_score)
                            
                            # Update visibility stats
                            for i, kp_name in enumerate(self.keypoint_names):
                                if i < len(gt_keypoints):
                                    visibility_stats[kp_name]['total'] += 1
                                    if gt_keypoints[i, 2] > 0:  # Visible in ground truth
                                        if pred_conf is None or (i < len(pred_conf) and pred_conf[i] > 0.5):
                                            visibility_stats[kp_name]['detected'] += 1
        
        # Calculate summary metrics
        avg_keypoint_accuracy = np.mean(keypoint_accuracies) if keypoint_accuracies else 0
        avg_throwing_score = np.mean(throwing_pose_detections) if throwing_pose_detections else 0
        
        # Calculate detection rates
        detection_rates = {}
        for kp_name, stats in visibility_stats.items():
            if stats['total'] > 0:
                detection_rates[kp_name] = stats['detected'] / stats['total']
            else:
                detection_rates[kp_name] = 0
        
        return {
            'avg_keypoint_accuracy': avg_keypoint_accuracy,
            'avg_throwing_score': avg_throwing_score,
            'keypoint_detection_rates': detection_rates,
            'total_analyzed_images': len(image_files),
            'valid_detections': len(keypoint_accuracies)
        }
    
    def calculate_keypoint_accuracy(self, pred_keypoints, gt_keypoints, threshold=0.05):
        """Calculate keypoint accuracy using normalized distance threshold"""
        if len(pred_keypoints) != len(gt_keypoints):
            return 0
        
        # Calculate normalized distances
        distances = np.linalg.norm(pred_keypoints - gt_keypoints, axis=1)
        
        # Count accurate keypoints (within threshold)
        accurate_keypoints = np.sum(distances < threshold)
        
        return accurate_keypoints / len(pred_keypoints)
    
    def analyze_throwing_pose(self, keypoints):
        """Analyze if the pose indicates throwing motion"""
        if len(keypoints) < 17:
            return 0
        
        # Get key throwing-related keypoints
        shoulder_indices = [5, 6]  # left_shoulder, right_shoulder
        elbow_indices = [7, 8]     # left_elbow, right_elbow
        wrist_indices = [9, 10]    # left_wrist, right_wrist
        
        throwing_indicators = []
        
        # Check arm elevation (wrists above shoulders)
        for i, (shoulder_idx, wrist_idx) in enumerate(zip(shoulder_indices, wrist_indices)):
            if shoulder_idx < len(keypoints) and wrist_idx < len(keypoints):
                if keypoints[wrist_idx, 1] < keypoints[shoulder_idx, 1]:  # Y coordinate (lower is higher)
                    throwing_indicators.append(1)
                else:
                    throwing_indicators.append(0)
        
        # Check arm extension (elbow-wrist distance)
        for i, (elbow_idx, wrist_idx) in enumerate(zip(elbow_indices, wrist_indices)):
            if elbow_idx < len(keypoints) and wrist_idx < len(keypoints):
                distance = np.linalg.norm(keypoints[elbow_idx] - keypoints[wrist_idx])
                if distance > 0.1:  # Threshold for extended arm
                    throwing_indicators.append(1)
                else:
                    throwing_indicators.append(0)
        
        return np.mean(throwing_indicators) if throwing_indicators else 0
    
    def visualize_predictions(self, split='val', num_samples=10):
        """Visualize model predictions on sample images"""
        split_dir = self.data_dir / split / 'images'
        if not split_dir.exists():
            print(f"Split directory {split_dir} does not exist")
            return
        
        image_files = list(split_dir.glob('*.*'))
        if not image_files:
            print(f"No images found in {split_dir}")
            return
        
        # Select random samples
        import random
        samples = random.sample(image_files, min(num_samples, len(image_files)))
        
        vis_dir = self.results_dir / 'visualizations'
        vis_dir.mkdir(exist_ok=True)
        
        print(f"Creating visualizations for {len(samples)} images...")
        
        for i, img_path in enumerate(samples):
            # Load image
            img = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Get predictions
            results = self.model(str(img_path), verbose=False)
            
            # Create visualization
            plt.figure(figsize=(12, 8))
            plt.imshow(img_rgb)
            plt.title(f'Pose Detection: {img_path.name}')
            plt.axis('off')
            
            # Draw keypoints if detected
            if results and results[0].keypoints is not None:
                keypoints = results[0].keypoints.xy[0].cpu().numpy()
                conf = results[0].keypoints.conf[0].cpu().numpy() if results[0].keypoints.conf is not None else None
                
                # Draw keypoints
                for j, (x, y) in enumerate(keypoints):
                    if conf is None or conf[j] > 0.5:
                        color = 'red' if j in [5, 6, 7, 8, 9, 10] else 'blue'  # Highlight arm keypoints
                        plt.scatter(x, y, c=color, s=50, alpha=0.8)
                        plt.text(x, y-10, self.keypoint_names[j] if j < len(self.keypoint_names) else str(j), 
                                fontsize=8, ha='center')
                
                # Draw skeleton connections
                self.draw_skeleton(plt, keypoints, conf)
            
            # Save visualization
            vis_path = vis_dir / f'prediction_{i:03d}_{img_path.stem}.png'
            plt.savefig(vis_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"Visualizations saved to: {vis_dir}")
    
    def draw_skeleton(self, plt, keypoints, conf=None, threshold=0.5):
        """Draw skeleton connections between keypoints"""
        # Define skeleton connections (COCO format)
        skeleton = [
            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
            [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
            [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
            [2, 4], [3, 5], [4, 6], [5, 7]
        ]
        
        for connection in skeleton:
            kp1_idx, kp2_idx = connection[0] - 1, connection[1] - 1  # Convert to 0-based indexing
            
            if (kp1_idx < len(keypoints) and kp2_idx < len(keypoints)):
                if conf is None or (conf[kp1_idx] > threshold and conf[kp2_idx] > threshold):
                    x1, y1 = keypoints[kp1_idx]
                    x2, y2 = keypoints[kp2_idx]
                    plt.plot([x1, x2], [y1, y2], 'g-', alpha=0.6, linewidth=2)
    
    def create_performance_plots(self, results):
        """Create performance visualization plots"""
        plots_dir = self.results_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        # 1. mAP comparison across splits
        splits = list(results.keys())
        map50_scores = [results[split].get('mAP50_pose', 0) for split in splits]
        map50_95_scores = [results[split].get('mAP50_95_pose', 0) for split in splits]
        
        plt.figure(figsize=(10, 6))
        x = np.arange(len(splits))
        width = 0.35
        
        plt.bar(x - width/2, map50_scores, width, label='mAP@50', alpha=0.8)
        plt.bar(x + width/2, map50_95_scores, width, label='mAP@50:95', alpha=0.8)
        
        plt.xlabel('Dataset Split')
        plt.ylabel('mAP Score')
        plt.title('Model Performance Across Dataset Splits')
        plt.xticks(x, splits)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'map_comparison.png', dpi=150)
        plt.close()
        
        # 2. Keypoint detection rates heatmap
        if 'val' in results and 'keypoint_detection_rates' in results['val']:
            detection_rates = results['val']['keypoint_detection_rates']
            
            # Create heatmap data
            keypoint_names = list(detection_rates.keys())
            rates = list(detection_rates.values())
            
            plt.figure(figsize=(12, 8))
            
            # Reshape for heatmap
            rates_matrix = np.array(rates).reshape(1, -1)
            
            sns.heatmap(rates_matrix, 
                       xticklabels=keypoint_names,
                       yticklabels=['Detection Rate'],
                       annot=True, 
                       fmt='.3f',
                       cmap='YlOrRd',
                       vmin=0, vmax=1)
            
            plt.title('Keypoint Detection Rates')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(plots_dir / 'keypoint_detection_rates.png', dpi=150)
            plt.close()
        
        print(f"Performance plots saved to: {plots_dir}")
    
    def generate_evaluation_report(self, results):
        """Generate a comprehensive evaluation report"""
        report_lines = []
        report_lines.append("# Pose Estimation Model Evaluation Report\n")
        report_lines.append(f"**Model:** {self.model_path.name}")
        report_lines.append(f"**Dataset:** {self.dataset_config_path}")
        report_lines.append(f"**Evaluation Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Overall Performance Summary
        report_lines.append("## Overall Performance Summary\n")
        
        for split, metrics in results.items():
            report_lines.append(f"### {split.upper()} Set Results")
            report_lines.append(f"- **mAP@50 (Pose):** {metrics.get('mAP50_pose', 0):.4f}")
            report_lines.append(f"- **mAP@50:95 (Pose):** {metrics.get('mAP50_95_pose', 0):.4f}")
            report_lines.append(f"- **Precision (Pose):** {metrics.get('precision_pose', 0):.4f}")
            report_lines.append(f"- **Recall (Pose):** {metrics.get('recall_pose', 0):.4f}")
            
            if 'avg_keypoint_accuracy' in metrics:
                report_lines.append(f"- **Average Keypoint Accuracy:** {metrics['avg_keypoint_accuracy']:.4f}")
                report_lines.append(f"- **Average Throwing Score:** {metrics['avg_throwing_score']:.4f}")
                report_lines.append(f"- **Images Analyzed:** {metrics.get('total_analyzed_images', 0)}")
                report_lines.append(f"- **Valid Detections:** {metrics.get('valid_detections', 0)}")
            
            report_lines.append("")
        
        # Keypoint Analysis
        if 'val' in results and 'keypoint_detection_rates' in results['val']:
            report_lines.append("## Keypoint Detection Analysis\n")
            detection_rates = results['val']['keypoint_detection_rates']
            
            # Sort by detection rate
            sorted_rates = sorted(detection_rates.items(), key=lambda x: x[1], reverse=True)
            
            report_lines.append("### Detection Rates by Keypoint")
            for kp_name, rate in sorted_rates:
                status = "✅" if rate > 0.8 else "⚠️" if rate > 0.6 else "❌"
                report_lines.append(f"- **{kp_name}:** {rate:.3f} {status}")
            
            report_lines.append("")
            
            # Analyze throwing-specific keypoints
            throwing_rates = {kp: detection_rates.get(kp, 0) for kp in self.throwing_keypoints}
            avg_throwing_rate = np.mean(list(throwing_rates.values()))
            
            report_lines.append("### Throwing-Related Keypoints Performance")
            report_lines.append(f"- **Average Detection Rate:** {avg_throwing_rate:.3f}")
            for kp_name, rate in throwing_rates.items():
                status = "✅" if rate > 0.8 else "⚠️" if rate > 0.6 else "❌"
                report_lines.append(f"  - **{kp_name}:** {rate:.3f} {status}")
        
        # Recommendations
        report_lines.append("\n## Recommendations\n")
        
        val_metrics = results.get('val', {})
        map50 = val_metrics.get('mAP50_pose', 0)
        
        if map50 < 0.5:
            report_lines.append("🔴 **Poor Performance Detected**")
            report_lines.append("- Consider increasing training epochs")
            report_lines.append("- Check annotation quality")
            report_lines.append("- Add more training data")
            report_lines.append("- Try data augmentation strategies")
        elif map50 < 0.7:
            report_lines.append("🟡 **Moderate Performance**")
            report_lines.append("- Fine-tune hyperparameters")
            report_lines.append("- Consider using a larger model")
            report_lines.append("- Improve data quality and diversity")
        else:
            report_lines.append("🟢 **Good Performance**")
            report_lines.append("- Model performs well for deployment")
            report_lines.append("- Consider optimizing for inference speed")
            report_lines.append("- Test on real-world scenarios")
        
        if 'keypoint_detection_rates' in val_metrics:
            detection_rates = val_metrics['keypoint_detection_rates']
            low_detection_kps = [kp for kp, rate in detection_rates.items() if rate < 0.6]
            
            if low_detection_kps:
                report_lines.append(f"\n🔍 **Low Detection Rate Keypoints:** {', '.join(low_detection_kps)}")
                report_lines.append("- Review annotation quality for these keypoints")
                report_lines.append("- Consider adding more diverse poses")
                report_lines.append("- Check for occlusion patterns")
        
        # Save report
        report_path = self.results_dir / 'evaluation_report.md'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Evaluation report saved to: {report_path}")
        
        # Also create plots
        self.create_performance_plots(results)


def main():
    parser = argparse.ArgumentParser(description='Evaluate YOLO11 pose estimation model')
    parser.add_argument('--model', required=True,
                       help='Path to trained model (.pt file)')
    parser.add_argument('--config', default='data/dataset.yaml',
                       help='Path to dataset configuration file')
    parser.add_argument('--split', default='val', choices=['train', 'val', 'test'],
                       help='Dataset split to evaluate')
    parser.add_argument('--visualize', action='store_true',
                       help='Create prediction visualizations')
    parser.add_argument('--num_vis', type=int, default=10,
                       help='Number of images to visualize')
    parser.add_argument('--comprehensive', action='store_true',
                       help='Run comprehensive evaluation on all splits')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = PoseEvaluator(args.model, args.config)
    
    if args.comprehensive:
        # Run comprehensive evaluation
        results = evaluator.run_comprehensive_evaluation()
    else:
        # Evaluate specific split
        print(f"Evaluating model on {args.split} split...")
        results = evaluator.evaluate_split(args.split)
        
        print("\nEvaluation Results:")
        for key, value in results.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for subkey, subvalue in value.items():
                    print(f"  {subkey}: {subvalue:.4f}" if isinstance(subvalue, float) else f"  {subkey}: {subvalue}")
            else:
                print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    
    # Create visualizations if requested
    if args.visualize:
        evaluator.visualize_predictions(args.split, args.num_vis)
    
    print(f"\nEvaluation complete! Results saved to: {evaluator.results_dir}")


if __name__ == "__main__":
    main()
