# Git Setup Instructions for YOLO11 Pose Estimation Project

## Quick Git Setup

### 1. Initialize Git Repository
```bash
# Navigate to project directory
cd pose

# Initialize git repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: YOLO11 pose estimation project setup"
```

### 2. Connect to Remote Repository (GitHub/GitLab)
```bash
# Add remote origin (replace with your repository URL)
git remote add origin https://github.com/yourusername/pose-estimation.git

# Push to remote
git branch -M main
git push -u origin main
```

### 3. Project Structure in Git
The repository includes:
- ✅ **Source code**: All Python scripts and notebooks
- ✅ **Documentation**: README, guides, and instructions  
- ✅ **Configuration**: YAML configs and requirements
- ✅ **Directory structure**: Empty folders with .gitkeep files
- ❌ **Large files**: Videos, models, and training data (excluded by .gitignore)

### 4. Working with the Repository

#### Clone and Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/pose-estimation.git
cd pose-estimation

# Install dependencies
pip install -r requirements.txt

# Verify setup
python -c "from ultralytics import YOLO; print('✅ Setup complete!')"
```

#### Add Your Training Data
```bash
# Add your videos
cp your_videos/* data/videos/

# Extract frames
python scripts/extract_frames.py

# Add annotations (after manual annotation)
cp your_annotations/* data/annotations/

# The .gitignore will prevent large files from being committed
```

#### Development Workflow
```bash
# Make changes to code
# Stage specific files
git add scripts/train.py

# Commit changes
git commit -m "Improve training configuration"

# Push changes
git push
```

### 5. File Size Management

#### What's Included in Git:
- 📝 Source code (Python scripts)
- 📓 Jupyter notebooks
- 📖 Documentation (README, guides)
- ⚙️ Configuration files
- 📁 Directory structure

#### What's Excluded (via .gitignore):
- 🎬 Video files (data/videos/*.mp4, etc.)
- 🖼️ Extracted frames (data/frames/**/*.jpg)
- 🤖 Trained models (models/*.pt)
- 📊 Training results (results/*/weights/*.pt)
- 📦 Large datasets

### 6. Sharing Your Project

#### For Code Sharing:
```bash
# Others can clone and run:
git clone your-repo-url
cd pose-estimation
pip install -r requirements.txt
# Then add their own videos and train
```

#### For Model Sharing (separate from Git):
- Upload trained models to Google Drive, Dropbox, or model hosting
- Include download instructions in README
- Use Git LFS for models if repository supports it

### 7. Git LFS (Large File Storage) - Optional

If you want to track large files:
```bash
# Install Git LFS
git lfs install

# Track large file types
git lfs track "*.pt"
git lfs track "*.mp4"
git lfs track "data/videos/*"

# Add .gitattributes
git add .gitattributes

# Now large files will be tracked with LFS
git add models/best_model.pt
git commit -m "Add trained model with LFS"
```

### 8. Collaboration Workflow

#### For Team Development:
```bash
# Create feature branch
git checkout -b feature/improve-detection

# Make changes and commit
git add .
git commit -m "Add new throwing detection algorithm"

# Push feature branch
git push origin feature/improve-detection

# Create pull request on GitHub/GitLab
# After review, merge to main
```

#### For Dataset Sharing:
- Use external storage for large datasets
- Share download links in project documentation
- Keep annotation files in Git (they're usually small)

### 9. Useful Git Commands for This Project

```bash
# Check repository status
git status

# View commit history
git log --oneline

# See what files are ignored
git ls-files --others --ignored --exclude-standard

# Check repository size
git count-objects -vH

# Clean up untracked files (be careful!)
git clean -n  # Preview what would be removed
git clean -f  # Actually remove untracked files
```

### 10. Troubleshooting

#### Large File Errors:
```bash
# If you accidentally added large files:
git rm --cached data/videos/*.mp4
git commit -m "Remove large video files from tracking"
```

#### Reset to Clean State:
```bash
# Remove all untracked files and directories
git clean -fd

# Reset to last commit
git reset --hard HEAD
```

### 11. Example Repository Structure

```
your-pose-estimation-repo/
├── .gitignore              # ✅ In Git
├── README.md               # ✅ In Git  
├── requirements.txt        # ✅ In Git
├── scripts/                # ✅ In Git
├── notebooks/              # ✅ In Git
├── configs/                # ✅ In Git
├── data/
│   ├── videos/.gitkeep     # ✅ In Git (structure only)
│   ├── videos/*.mp4        # ❌ Excluded by .gitignore
│   ├── annotations/*.txt   # ✅ In Git (small files)
│   └── train/.gitkeep      # ✅ In Git (structure only)
├── models/.gitkeep         # ✅ In Git (structure only)
├── models/*.pt             # ❌ Excluded by .gitignore
└── results/.gitkeep        # ✅ In Git (structure only)
```

This setup allows for easy collaboration on code while keeping repository size manageable!
