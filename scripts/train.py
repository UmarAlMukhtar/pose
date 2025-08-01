"""
YOLO11 Pose Estimation Training Script

This script trains a YOLO11 pose estimation model to detect people throwing waste.
It includes advanced configuration options and monitoring capabilities.
"""

import os
import yaml
import argparse
from pathlib import Path
import torch
from ultralytics import YOLO
import wandb
from datetime import datetime
import json


class PoseTrainer:
    def __init__(self, config_path, model_size='n', project_name='pose-estimation'):
        """
        Initialize the Pose Trainer
        
        Args:
            config_path (str): Path to dataset.yaml configuration file
            model_size (str): YOLO model size ('n', 's', 'm', 'l', 'x')
            project_name (str): Project name for logging
        """
        self.config_path = Path(config_path)
        self.model_size = model_size
        self.project_name = project_name
        
        # Load dataset configuration
        with open(self.config_path, 'r') as f:
            self.dataset_config = yaml.safe_load(f)
        
        # Set up directories
        self.data_dir = Path(self.dataset_config['path'])
        self.models_dir = self.data_dir.parent / 'models'
        self.results_dir = self.data_dir.parent / 'results'
        
        # Create directories
        self.models_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        # Model selection
        self.model_path = f'yolo11{model_size}-pose.pt'
        
        # Training configuration
        self.training_config = {
            'epochs': 200,
            'imgsz': 640,
            'batch': 16,
            'lr0': 0.01,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 0.05,
            'cls': 0.5,
            'kobj': 1.0,
            'pose': 12.0,
            'mosaic': 1.0,
            'mixup': 0.1,
            'copy_paste': 0.1,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
        }
    
    def setup_wandb(self, run_name=None):
        """Setup Weights & Biases logging"""
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"yolo11{self.model_size}_pose_{timestamp}"
        
        wandb.init(
            project=self.project_name,
            name=run_name,
            config=self.training_config,
            tags=[f"yolo11{self.model_size}", "pose-estimation", "waste-throwing"]
        )
        
        return run_name
    
    def optimize_hyperparameters(self):
        """Hyperparameter optimization for pose estimation"""
        print("Starting hyperparameter optimization...")
        
        # Define hyperparameter search space
        hyperparameters = {
            'lr0': [0.001, 0.01, 0.1],
            'lrf': [0.001, 0.01, 0.1],
            'momentum': [0.9, 0.937, 0.95],
            'weight_decay': [0.0001, 0.0005, 0.001],
            'pose': [8.0, 12.0, 16.0],
            'kobj': [0.5, 1.0, 1.5],
            'mosaic': [0.5, 1.0],
            'mixup': [0.0, 0.1, 0.2]
        }
        
        best_score = 0
        best_config = self.training_config.copy()
        
        # Simple grid search (you can implement more sophisticated methods)
        import itertools
        
        # Limit combinations for practical purposes
        key_params = ['lr0', 'pose', 'kobj']
        param_combinations = []
        
        for params in itertools.product(*[hyperparameters[key] for key in key_params]):
            param_dict = dict(zip(key_params, params))
            param_combinations.append(param_dict)
        
        print(f"Testing {len(param_combinations)} hyperparameter combinations...")
        
        for i, params in enumerate(param_combinations[:5]):  # Limit to 5 for demo
            print(f"\nTesting combination {i+1}/{min(5, len(param_combinations))}: {params}")
            
            # Update config
            test_config = self.training_config.copy()
            test_config.update(params)
            
            # Train with reduced epochs for testing
            test_config['epochs'] = 20
            
            model = YOLO(self.model_path)
            results = model.train(
                data=str(self.config_path),
                epochs=test_config['epochs'],
                imgsz=test_config['imgsz'],
                batch=test_config['batch'],
                project=str(self.results_dir),
                name=f'hyperopt_{i}',
                **{k: v for k, v in test_config.items() if k not in ['epochs', 'imgsz', 'batch']}
            )
            
            # Get validation score (mAP)
            val_score = results.results_dict.get('metrics/mAP50(P)', 0)
            
            if val_score > best_score:
                best_score = val_score
                best_config = test_config.copy()
                print(f"New best score: {best_score:.4f}")
        
        print(f"\nBest hyperparameters found with score {best_score:.4f}:")
        for key, value in best_config.items():
            if key in hyperparameters:
                print(f"  {key}: {value}")
        
        # Update training config
        self.training_config = best_config
        
        # Save best config
        config_save_path = self.results_dir / 'best_hyperparameters.json'
        with open(config_save_path, 'w') as f:
            json.dump(best_config, f, indent=2)
        
        print(f"Best configuration saved to: {config_save_path}")
    
    def train_model(self, use_wandb=True, optimize_hyperparams=False, resume=None):
        """Train the pose estimation model"""
        print(f"Starting training with YOLO11{self.model_size} pose estimation...")
        print(f"Dataset: {self.config_path}")
        print(f"GPU available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Setup logging
        run_name = None
        if use_wandb:
            try:
                run_name = self.setup_wandb()
                print(f"W&B logging enabled. Run: {run_name}")
            except Exception as e:
                print(f"W&B setup failed: {e}")
                use_wandb = False
        
        # Optimize hyperparameters if requested
        if optimize_hyperparams and not resume:
            self.optimize_hyperparameters()
        
        # Load model
        if resume:
            print(f"Resuming training from: {resume}")
            model = YOLO(resume)
        else:
            print(f"Loading pretrained model: {self.model_path}")
            model = YOLO(self.model_path)
        
        # Start training
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"yolo11{self.model_size}_pose_{timestamp}"
        
        try:
            results = model.train(
                data=str(self.config_path),
                epochs=self.training_config['epochs'],
                imgsz=self.training_config['imgsz'],
                batch=self.training_config['batch'],
                project=str(self.results_dir),
                name=experiment_name,
                save=True,
                save_period=10,  # Save checkpoint every 10 epochs
                cache=True,
                device=0 if torch.cuda.is_available() else 'cpu',
                workers=4,
                exist_ok=True,
                pretrained=True,
                optimizer='AdamW',
                verbose=True,
                seed=42,
                deterministic=True,
                single_cls=False,
                rect=False,
                cos_lr=True,
                close_mosaic=10,  # Close mosaic augmentation in last 10 epochs
                resume=resume is not None,
                amp=True,  # Automatic Mixed Precision
                fraction=1.0,
                profile=False,
                # Pose-specific parameters
                **{k: v for k, v in self.training_config.items() 
                   if k not in ['epochs', 'imgsz', 'batch']}
            )
            
            print("Training completed successfully!")
            
            # Save model to models directory
            best_model_path = self.results_dir / experiment_name / 'weights' / 'best.pt'
            if best_model_path.exists():
                final_model_path = self.models_dir / f'best_yolo11{self.model_size}_pose_{timestamp}.pt'
                import shutil
                shutil.copy2(best_model_path, final_model_path)
                print(f"Best model saved to: {final_model_path}")
            
            # Log final results
            if use_wandb:
                wandb.log({
                    "final_mAP50": results.results_dict.get('metrics/mAP50(P)', 0),
                    "final_mAP50-95": results.results_dict.get('metrics/mAP50-95(P)', 0),
                    "training_time": results.results_dict.get('train/time', 0)
                })
                wandb.finish()
            
            return results
            
        except Exception as e:
            print(f"Training failed: {e}")
            if use_wandb:
                wandb.finish()
            raise
    
    def evaluate_model(self, model_path, split='val'):
        """Evaluate the trained model"""
        print(f"Evaluating model: {model_path}")
        print(f"Split: {split}")
        
        model = YOLO(model_path)
        
        # Run evaluation
        results = model.val(
            data=str(self.config_path),
            split=split,
            imgsz=640,
            batch=16,
            save_json=True,
            save_hybrid=True,
            conf=0.25,
            iou=0.6,
            max_det=300,
            half=True,
            device=0 if torch.cuda.is_available() else 'cpu',
            dnn=False,
            plots=True,
            verbose=True
        )
        
        # Print key metrics
        print("\nEvaluation Results:")
        print(f"mAP50(P): {results.results_dict.get('metrics/mAP50(P)', 0):.4f}")
        print(f"mAP50-95(P): {results.results_dict.get('metrics/mAP50-95(P)', 0):.4f}")
        
        return results
    
    def create_training_summary(self):
        """Create a summary of the training process"""
        summary = {
            'dataset': {
                'path': str(self.config_path),
                'train_images': len(list((self.data_dir / 'train' / 'images').glob('*.*'))),
                'val_images': len(list((self.data_dir / 'val' / 'images').glob('*.*'))),
                'test_images': len(list((self.data_dir / 'test' / 'images').glob('*.*'))) if (self.data_dir / 'test').exists() else 0,
            },
            'model': {
                'size': self.model_size,
                'pretrained_model': self.model_path,
            },
            'training_config': self.training_config,
            'system_info': {
                'gpu_available': torch.cuda.is_available(),
                'gpu_name': torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                'torch_version': torch.__version__,
            }
        }
        
        summary_path = self.results_dir / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Training summary saved to: {summary_path}")
        return summary


def main():
    parser = argparse.ArgumentParser(description='Train YOLO11 pose estimation model')
    parser.add_argument('--config', default='data/dataset.yaml',
                       help='Path to dataset configuration file')
    parser.add_argument('--model_size', default='n', choices=['n', 's', 'm', 'l', 'x'],
                       help='YOLO model size')
    parser.add_argument('--project', default='waste-throwing-pose',
                       help='Project name for logging')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Input image size')
    parser.add_argument('--no_wandb', action='store_true',
                       help='Disable Weights & Biases logging')
    parser.add_argument('--optimize_hyperparams', action='store_true',
                       help='Perform hyperparameter optimization')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume training from checkpoint')
    parser.add_argument('--eval_only', type=str, default=None,
                       help='Only evaluate the specified model')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = PoseTrainer(
        config_path=args.config,
        model_size=args.model_size,
        project_name=args.project
    )
    
    # Update training config with command line args
    trainer.training_config['epochs'] = args.epochs
    trainer.training_config['batch'] = args.batch
    trainer.training_config['imgsz'] = args.imgsz
    
    if args.eval_only:
        # Only run evaluation
        trainer.evaluate_model(args.eval_only)
    else:
        # Create training summary
        trainer.create_training_summary()
        
        # Train model
        results = trainer.train_model(
            use_wandb=not args.no_wandb,
            optimize_hyperparams=args.optimize_hyperparams,
            resume=args.resume
        )
        
        print(f"\nTraining completed!")
        print(f"Results saved to: {trainer.results_dir}")
        print(f"Models saved to: {trainer.models_dir}")


if __name__ == "__main__":
    main()
