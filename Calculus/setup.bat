@echo off
echo 🦷 Setting up Calculus Detection AI Integration...
echo =================================================

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed. Please install Python 3.8+ first.
    pause
    exit /b 1
)

echo ✅ Python found
python --version

REM Check if pip is installed
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ pip is not installed. Please install pip first.
    pause
    exit /b 1
)

echo ✅ pip found
pip --version

REM Install Python dependencies
echo 📦 Installing Python dependencies...
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo ❌ Failed to install Python dependencies.
    pause
    exit /b 1
)

echo ✅ Python dependencies installed successfully!

REM Check if Node.js is installed
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Node.js is not installed. Please install Node.js first.
    pause
    exit /b 1
)

echo ✅ Node.js found
node --version

REM Install Node.js dependencies
echo 📦 Installing Node.js dependencies...
npm install

if %errorlevel% neq 0 (
    echo ❌ Failed to install Node.js dependencies.
    pause
    exit /b 1
)

echo ✅ Node.js dependencies installed successfully!

REM Check if model files exist
echo 🔍 Checking for model files...

if not exist "..\segmentyolo.pt" (
    echo ⚠️  Warning: segmentyolo.pt not found in parent directory
    echo    Please ensure the YOLO model file is available
)

if not exist "..\best_model.pth" (
    echo ⚠️  Warning: best_model.pth not found in parent directory
    echo    Please ensure the U-Net model file is available
)

if not exist "..\default.yaml" (
    echo ⚠️  Warning: default.yaml not found in parent directory
    echo    Please ensure the configuration file is available
)

echo.
echo 🎉 Setup complete!
echo.
echo To start the server:
echo   npm start
echo.
echo To start in development mode:
echo   npm run dev
echo.
echo The AI detection will be available at: http://localhost:3000/detect
pause
