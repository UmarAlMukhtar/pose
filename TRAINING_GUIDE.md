# YOLO11 Pose Estimation Training Guide

## Complete Step-by-Step Process

### 1. Environment Setup
- Install Python 3.8+
- Install required packages (see requirements.txt)
- Verify GPU availability

### 2. Data Preparation

#### Video Collection
- Collect videos of people throwing waste
- Minimum 5-10 minutes of diverse footage
- Various angles, lighting conditions, and people
- Good quality videos (720p minimum recommended)

#### Frame Extraction
```python
# Extract frames at 2 FPS
python scripts/extract_frames.py --video_dir data/videos --output_dir data/frames --fps_target 2
```

#### Annotation Process
1. **Tool Selection**: Use CVAT (recommended) or similar
2. **Annotation Format**: YOLO pose format (17 keypoints)
3. **Key Focus Areas**:
   - Accurate shoulder, elbow, wrist positions
   - Proper visibility marking (0=not visible, 1=visible, 2=occluded)
   - Consistent annotation quality

#### Dataset Organization
```python
# Prepare YOLO dataset structure
python scripts/prepare_dataset.py --frames_dir data/frames --annotations_dir data/annotations
```

### 3. Model Training

#### Configuration
- **Small Dataset** (< 500 images): Use more epochs, higher augmentation
- **Medium Dataset** (500-2000 images): Standard configuration
- **Large Dataset** (> 2000 images): Fewer epochs, lower learning rate

#### Training Command
```python
# Basic training
python scripts/train.py --config data/dataset.yaml --model_size s --epochs 200

# Advanced training with monitoring
python scripts/train.py --config data/dataset.yaml --model_size m --epochs 200 --optimize_hyperparams
```

#### Training Parameters for Pose Estimation
- **Pose Loss Weight**: 12.0 (high importance for keypoint accuracy)
- **Keypoint Objectness**: 1.0 (balance detection and pose)
- **Learning Rate**: 0.001 (stable for pose learning)
- **Augmentation**: Moderate to preserve pose relationships

### 4. Model Evaluation

#### Validation Metrics
- **mAP@50 (Pose)**: Primary metric for pose detection accuracy
- **mAP@50:95 (Pose)**: Comprehensive pose evaluation
- **Keypoint Accuracy**: Per-keypoint detection rates
- **Throwing Score**: Custom metric for throwing pose detection

#### Evaluation Command
```python
python scripts/evaluate.py --model models/best_model.pt --config data/dataset.yaml --comprehensive
```

### 5. Inference and Deployment

#### Video Processing
```python
# Process new videos
yolo pose predict model=models/best_model.pt source=path/to/video.mp4
```

#### Real-time Detection
```python
# Webcam detection
yolo pose predict model=models/best_model.pt source=0
```

## Expected Results and Benchmarks

### Performance Targets
- **Good Performance**: mAP@50 > 0.7
- **Excellent Performance**: mAP@50 > 0.85
- **Production Ready**: mAP@50 > 0.8 + consistent throwing detection

### Throwing-Specific Keypoints Performance
- **Shoulders**: Should achieve >90% detection rate
- **Elbows**: Target >85% detection rate  
- **Wrists**: Target >80% detection rate (most challenging)
- **Hips**: Target >90% detection rate

### Training Time Estimates
- **Small Dataset**: 2-4 hours (GPU) / 8-16 hours (CPU)
- **Medium Dataset**: 4-8 hours (GPU) / 16-32 hours (CPU)
- **Large Dataset**: 8-16 hours (GPU) / 32+ hours (CPU)

## Troubleshooting Common Issues

### Poor Performance (mAP < 0.5)
- **Data Quality**: Check annotation accuracy
- **Dataset Size**: Add more diverse training data
- **Model Size**: Try larger model (YOLOv11m or YOLOv11l)
- **Training Duration**: Increase epochs

### Keypoint Detection Issues
- **Specific Keypoints Missing**: Focus annotation quality on those points
- **Occlusion Handling**: Add more occluded examples with proper visibility marking
- **Motion Blur**: Include motion-blurred frames in training

### Training Instability
- **Loss Spikes**: Reduce learning rate
- **GPU Memory**: Reduce batch size or image size
- **Overfitting**: Increase data augmentation or add regularization

### Inference Speed Issues
- **Model Size**: Use smaller model (YOLOv11n or YOLOv11s)
- **Image Resolution**: Reduce input size (512 instead of 640)
- **Hardware**: Optimize for target deployment hardware

## Advanced Techniques

### Data Augmentation for Pose
- **Moderate Augmentation**: Preserve anatomical relationships
- **Spatial Augmentations**: Careful with rotations (affects pose interpretation)
- **Color Augmentations**: Safe for all pose applications

### Transfer Learning
- **Pre-trained Models**: Start with COCO-trained pose models
- **Fine-tuning**: Adjust learning rates for different model parts
- **Progressive Training**: Start with detection, then add pose

### Multi-stage Training
1. **Stage 1**: Train on general human poses
2. **Stage 2**: Fine-tune on throwing-specific poses
3. **Stage 3**: Optimize for deployment constraints

## Best Practices

### Annotation Guidelines
- **Consistency**: Same annotator for similar actions
- **Quality over Quantity**: Better to have fewer high-quality annotations
- **Edge Cases**: Include difficult poses and lighting conditions
- **Validation**: Cross-check annotations between team members

### Model Selection
- **Development**: Start with YOLOv11s for fast iteration
- **Production**: YOLOv11m for best balance
- **Resource-Constrained**: YOLOv11n for mobile/embedded
- **High-Accuracy**: YOLOv11l or YOLOv11x for maximum performance

### Deployment Considerations
- **Model Optimization**: Use TensorRT, ONNX, or CoreML
- **Inference Pipeline**: Optimize preprocessing and postprocessing
- **Error Handling**: Robust handling of edge cases
- **Monitoring**: Track model performance in production
