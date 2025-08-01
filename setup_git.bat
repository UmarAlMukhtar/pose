@echo off
REM Git Setup Script for YOLO11 Pose Estimation Project
REM This script helps initialize the Git repository

echo.
echo ============================================
echo  YOLO11 Pose Estimation - Git Setup
echo ============================================
echo.

REM Check if Git is installed
git --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Git is not installed or not in PATH
    echo Please install Git from: https://git-scm.com/
    pause
    exit /b 1
)

echo ✅ Git is installed
echo.

REM Check if already a Git repository
if exist .git (
    echo ⚠️  This directory is already a Git repository
    echo Current status:
    git status --short
    echo.
    goto :menu
)

REM Initialize Git repository
echo 🚀 Initializing Git repository...
git init
if errorlevel 1 (
    echo ❌ Failed to initialize Git repository
    pause
    exit /b 1
)

echo ✅ Git repository initialized
echo.

REM Add all files
echo 📁 Adding all files...
git add .
if errorlevel 1 (
    echo ❌ Failed to add files
    pause
    exit /b 1
)

echo ✅ Files added to staging area
echo.

REM Show what will be committed
echo 📋 Files to be committed:
git status --short
echo.

REM Create initial commit
echo 💾 Creating initial commit...
git commit -m "Initial commit: YOLO11 pose estimation project setup"
if errorlevel 1 (
    echo ❌ Failed to create initial commit
    pause
    exit /b 1
)

echo ✅ Initial commit created
echo.

:menu
echo 🔗 Next steps:
echo.
echo 1. Connect to remote repository (GitHub/GitLab)
echo 2. View repository status
echo 3. Exit
echo.
set /p choice="Enter your choice (1-3): "

if "%choice%"=="1" goto :remote
if "%choice%"=="2" goto :status
if "%choice%"=="3" goto :end
goto :menu

:remote
echo.
echo 🌐 To connect to a remote repository:
echo.
echo 1. Create a new repository on GitHub/GitLab
echo 2. Copy the repository URL
echo 3. Run these commands:
echo.
echo    git remote add origin YOUR_REPOSITORY_URL
echo    git branch -M main
echo    git push -u origin main
echo.
echo Example:
echo    git remote add origin https://github.com/yourusername/pose-estimation.git
echo    git branch -M main
echo    git push -u origin main
echo.
set /p repo_url="Enter your repository URL (or press Enter to skip): "

if "%repo_url%"=="" (
    echo Skipping remote setup
    goto :menu
)

echo Adding remote origin...
git remote add origin %repo_url%
if errorlevel 1 (
    echo ❌ Failed to add remote origin
    goto :menu
)

echo Setting main branch...
git branch -M main
if errorlevel 1 (
    echo ❌ Failed to set main branch
    goto :menu
)

echo 🚀 Pushing to remote repository...
git push -u origin main
if errorlevel 1 (
    echo ❌ Failed to push to remote repository
    echo This might be due to authentication or network issues
    echo Please check your credentials and try again manually
) else (
    echo ✅ Successfully pushed to remote repository!
)

goto :menu

:status
echo.
echo 📊 Repository Status:
echo.
git status
echo.
echo 📝 Recent commits:
git log --oneline -5
echo.
goto :menu

:end
echo.
echo 🎉 Git setup complete!
echo.
echo 📚 For more information, see:
echo    - GIT_SETUP.md for detailed instructions
echo    - README.md for project overview
echo.
echo 🚀 You can now start working with your YOLO11 pose estimation project!
echo.
pause
