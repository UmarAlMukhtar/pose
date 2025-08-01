# YOLO11 Pose Estimation for Waste Throwing Detection

A comprehensive project for training a YOLO11 pose estimation model to detect people throwing waste from video data. This repository provides everything you need from data preparation to model deployment.

## 🎯 Project Overview

This project uses YOLO11's advanced pose estimation capabilities to:
- **Detect human poses** in videos with 17 keypoints
- **Identify throwing motions** using pose analysis algorithms
- **Train custom models** specifically for waste throwing detection
- **Provide real-time inference** for deployment scenarios

## 📁 Project Structure

```
pose/
├── data/
│   ├── videos/          # 📹 Original training videos
│   ├── frames/          # 🖼️ Extracted frames from videos
│   ├── annotations/     # 📝 YOLO format pose annotations
│   ├── train/          # 🎯 Training dataset (images + labels)
│   ├── val/            # ✅ Validation dataset
│   └── test/           # 🧪 Test dataset
├── models/             # 🤖 Trained model files
├── scripts/            # ⚙️ Python scripts for all operations
│   ├── extract_frames.py    # Extract frames from videos
│   ├── prepare_dataset.py   # Organize and split dataset
│   ├── train.py            # Main training script
│   ├── evaluate.py         # Model evaluation
│   └── inference.py        # Real-time inference
├── notebooks/          # 📓 Jupyter notebooks
│   └── yolo11_pose_training_complete.ipynb
├── configs/            # ⚙️ Configuration files
├── results/            # 📊 Training results and logs
├── requirements.txt    # 📦 Dependencies
├── TRAINING_GUIDE.md   # 📖 Detailed training guide
└── README.md          # 📄 This file
```

## 🚀 Quick Start

### 1. Installation
```bash
# Clone or download this project
cd pose

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from ultralytics import YOLO; print('✅ YOLO11 ready!')"
```

### 2. Prepare Your Data
```bash
# 1. Place your videos in data/videos/
# Supported formats: .mp4, .avi, .mov, .mkv, .wmv, .flv

# 2. Extract frames (2 FPS recommended)
python scripts/extract_frames.py --video_dir data/videos --output_dir data/frames --fps_target 2

# 3. Annotate frames using CVAT or similar tool
# Export annotations in YOLO pose format to data/annotations/

# 4. Prepare dataset splits
python scripts/prepare_dataset.py --frames_dir data/frames --annotations_dir data/annotations
```

### 3. Train the Model
```bash
# Basic training (recommended for beginners)
python scripts/train.py --config data/dataset.yaml --model_size s --epochs 200

# Advanced training with hyperparameter optimization
python scripts/train.py --config data/dataset.yaml --model_size m --epochs 200 --optimize_hyperparams
```

### 4. Evaluate and Use
```bash
# Evaluate model performance
python scripts/evaluate.py --model models/best_model.pt --config data/dataset.yaml --comprehensive

# Test on new videos
python scripts/inference.py --model models/best_model.pt --source path/to/test_video.mp4

# Real-time webcam detection
python scripts/inference.py --model models/best_model.pt --source webcam
```

## 📓 Interactive Notebook Tutorial

For a complete interactive tutorial, open the Jupyter notebook:
```bash
jupyter notebook notebooks/yolo11_pose_training_complete.ipynb
```

This notebook provides:
- ✅ Step-by-step guidance with explanations
- 📊 Data visualization and analysis
- 🔧 Interactive configuration
- 📈 Training monitoring and evaluation
- 🎯 Complete workflow from start to finish

## 📋 Detailed Step-by-Step Process

### Step 1: Environment Setup
- **Python 3.8+** with pip
- **GPU recommended** (CUDA-capable)
- **8GB+ RAM** for training
- **50GB+ storage** for datasets and models

### Step 2: Video Data Collection
- 📹 **Minimum 5-10 minutes** of diverse footage
- 🎥 **Good quality videos** (720p minimum)
- 🌟 **Various scenarios**: Different people, angles, lighting
- 🎯 **Focus on throwing motions**: Clear waste throwing actions

### Step 3: Frame Extraction
```python
# Extract frames at optimal intervals
FrameExtractor(strategy='uniform', fps_target=2)
# Results: ~600-1200 frames per 10-minute video
```

### Step 4: Annotation Process
- **Tool**: CVAT (Computer Vision Annotation Tool) - *recommended*
- **Format**: YOLO pose format (17 keypoints)
- **Target**: 500-1000+ annotated frames minimum
- **Quality**: Focus on accurate shoulder, elbow, wrist positions

**Key Keypoints for Throwing Detection:**
- 🎯 **Shoulders** (5, 6): Body orientation
- 💪 **Elbows** (7, 8): Arm bending during throw
- ✋ **Wrists** (9, 10): Hand position and motion direction
- 🦵 **Hips** (11, 12): Lower body stability

### Step 5: Dataset Preparation
```python
# Automatic train/val/test split (70/20/10)
prepare_yolo_dataset(train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
```

### Step 6: Model Selection
| Model Size | Speed | Accuracy | GPU Memory | Use Case |
|------------|-------|----------|------------|----------|
| **YOLO11n** | ⚡⚡⚡ | ⭐⭐ | 2GB | Mobile/Edge |
| **YOLO11s** | ⚡⚡ | ⭐⭐⭐ | 4GB | Development |
| **YOLO11m** | ⚡ | ⭐⭐⭐⭐ | 6GB | **Recommended** |
| **YOLO11l** | 🐌 | ⭐⭐⭐⭐⭐ | 8GB | High Accuracy |
| **YOLO11x** | 🐌🐌 | ⭐⭐⭐⭐⭐ | 12GB | Maximum Quality |

### Step 7: Training Configuration
```python
# Optimized for pose estimation
training_config = {
    'epochs': 200,
    'pose': 12.0,      # High pose loss weight
    'kobj': 1.0,       # Keypoint objectness
    'lr0': 0.001,      # Learning rate
    'mosaic': 1.0,     # Data augmentation
    'mixup': 0.1,      # Advanced augmentation
}
```

### Step 8: Training Execution
- ⏱️ **Expected time**: 2-8 hours (varies by dataset size and GPU)
- 📊 **Monitoring**: Real-time metrics via Weights & Biases (optional)
- 💾 **Checkpoints**: Automatic saving every 10 epochs
- 🎯 **Target metrics**: mAP@50 > 0.7 for good performance

### Step 9: Model Evaluation
```python
# Comprehensive evaluation metrics
- mAP@50 (Pose): Primary pose detection metric
- mAP@50:95 (Pose): Comprehensive evaluation  
- Keypoint Detection Rates: Per-keypoint analysis
- Throwing Score: Custom throwing motion metric
```

### Step 10: Inference and Deployment
```python
# Simple inference command
yolo pose predict model=best_model.pt source=video.mp4

# Advanced inference with custom analysis
python scripts/inference.py --model best_model.pt --source video.mp4
```

## 🎯 Performance Expectations

### **Good Performance Targets**
- ✅ **mAP@50 > 0.7**: Suitable for development and testing
- ✅ **Keypoint Detection > 80%**: Reliable pose estimation
- ✅ **Throwing Detection**: Consistent identification of throwing motions

### **Excellent Performance Targets**
- 🌟 **mAP@50 > 0.85**: Production-ready quality
- 🌟 **Keypoint Detection > 90%**: High-precision pose estimation
- 🌟 **Real-time Processing**: 30+ FPS on target hardware

### **Training Time Estimates**
| Dataset Size | GPU Training | CPU Training |
|--------------|--------------|--------------|
| Small (< 500 images) | 2-4 hours | 8-16 hours |
| Medium (500-2000) | 4-8 hours | 16-32 hours |
| Large (> 2000) | 8-16 hours | 32+ hours |

## 🛠️ Troubleshooting

### Common Issues and Solutions

**❌ Poor Performance (mAP < 0.5)**
- 🔍 Check annotation quality and consistency
- 📈 Add more diverse training data
- 🤖 Try larger model (YOLOv11m → YOLOv11l)
- ⏱️ Increase training epochs

**❌ GPU Memory Errors**
- 📉 Reduce batch size: `--batch 8`
- 🖼️ Reduce image size: `--imgsz 512`
- 🤖 Use smaller model: YOLOv11s or YOLOv11n

**❌ Specific Keypoints Missing**
- 📝 Focus annotation quality on those keypoints
- 🎯 Add more examples with those keypoints visible
- ⚖️ Adjust pose loss weight: `--pose 15.0`

**❌ Training Instability**
- 📉 Reduce learning rate: `--lr0 0.0005`
- 🎲 Increase data augmentation
- ⏱️ Add warmup epochs: `--warmup_epochs 5`

## 📚 Additional Resources

### **Documentation and Tutorials**
- 🔗 [YOLO11 Official Docs](https://docs.ultralytics.com/)
- 🔗 [CVAT Annotation Tool](https://cvat.org/)
- 🔗 [Pose Estimation Guide](https://blog.roboflow.com/pose-estimation/)

### **Example Datasets**
- 🔗 [COCO Pose Dataset](http://cocodataset.org/#keypoints-2020)
- 🔗 [Human Pose Datasets](https://paperswithcode.com/datasets?task=pose-estimation)

### **Community and Support**
- 🔗 [Ultralytics Community](https://community.ultralytics.com/)
- 🔗 [GitHub Issues](https://github.com/ultralytics/ultralytics/issues)

## 📄 License

This project is provided for educational and research purposes. Please check individual component licenses:
- YOLO11: AGPL-3.0 License
- OpenCV: Apache 2.0 License
- Other dependencies: See requirements.txt

## 🤝 Contributing

We welcome contributions! Please feel free to:
- 🐛 Report bugs and issues
- 💡 Suggest improvements
- 📖 Improve documentation
- 🔧 Submit code enhancements

## 📞 Support

For questions and support:
1. 📖 Check this README and TRAINING_GUIDE.md
2. 🔍 Review the Jupyter notebook tutorial
3. 💬 Ask questions in GitHub Issues
4. 📧 Contact the maintainers

---

**🎉 Ready to detect waste throwing with AI? Start with the Jupyter notebook for an interactive experience!**
