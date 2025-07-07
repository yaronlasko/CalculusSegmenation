# 🦷 Dental Calculus Detection Setup Guide

This guide will help you set up and run the Dental Calculus Detection project on a new machine.

## 📋 Prerequisites

Before starting, ensure you have the following installed:

### Required Software
1. **Python 3.8+** - [Download from python.org](https://www.python.org/downloads/)
2. **Node.js 16+** - [Download from nodejs.org](https://nodejs.org/)
3. **Git** - [Download from git-scm.com](https://git-scm.com/)

### Verify Installation
Open a terminal/command prompt and run:
```bash
python --version    # Should show Python 3.8+
node --version      # Should show Node.js 16+
npm --version       # Should show npm 6+
git --version       # Should show git version
```

---

## 🚀 Setup Instructions

### Step 1: Clone the Repository
```bash
git clone [your-repository-url]
cd CalculusSegmenation
```

### Step 2: Download Model Files
The following large model files are not included in git and must be downloaded separately:

1. **YOLO Model** (`segmentyolo.pt`) - Place in project root
2. **U-Net Model** (`best_model.pth`) - Place in project root
   - Download from: https://drive.google.com/file/d/1H9YKaYMzEsLSEydyk2XK3GSnE6B1rWUJ/view?usp=sharing

Your directory structure should look like:
```
CalculusSegmenation/
├── segmentyolo.pt         # ← Download this
├── best_model.pth         # ← Download this
├── default.yaml
├── Calculus/
│   ├── package.json
│   ├── server.js
│   └── ...
└── ...
```

### Step 3: Run Setup Script

**For Windows:**
```bash
cd Calculus
.\setup.bat
```

**For Linux/Mac:**
```bash
cd Calculus
chmod +x setup.sh
./setup.sh
```

The setup script will:
- ✅ Check Python and Node.js installation
- 📦 Install Python dependencies (PyTorch, OpenCV, etc.)
- 📦 Install Node.js dependencies (Express, etc.)
- 🔍 Verify model files exist

### Step 4: Start the Server

**Option A: Using start script (Windows)**
```bash
.\start-server.bat
```

**Option B: Using npm**
```bash
npm start
```

**Option C: Development mode (with auto-restart)**
```bash
npm run dev
```

---

## 🌐 Accessing the Application

Once the server starts successfully, you should see:
```
Server running on port 3000
Visit http://localhost:3000 to view the application
```

Open your browser and go to:
- **Main App**: http://localhost:3000
- **AI Detection**: http://localhost:3000/detect
- **Test Interface**: http://localhost:3000/test
- **Annotation Tool**: http://localhost:3000/annotate

---

## ❗ Troubleshooting

### Common Issues and Solutions

#### 1. "Python not found" Error
**Solution**: 
- Install Python 3.8+ from python.org
- Make sure to check "Add Python to PATH" during installation
- Restart your terminal

#### 2. "Node.js not found" Error
**Solution**:
- Install Node.js 16+ from nodejs.org
- Restart your terminal
- Try `node --version` to verify

#### 3. Python Dependencies Installation Fails
**Solution**:
```bash
# Try upgrading pip first
python -m pip install --upgrade pip

# Install dependencies with verbose output
pip install -r requirements.txt -v

# For PyTorch issues, install separately:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### 4. Node.js Dependencies Installation Fails
**Solution**:
```bash
# Clear npm cache
npm cache clean --force

# Delete node_modules and reinstall
rm -rf node_modules package-lock.json  # Linux/Mac
# or
rmdir /s node_modules && del package-lock.json  # Windows

npm install
```

#### 5. Model Files Missing
**Error**: "Warning: segmentyolo.pt not found" or "best_model.pth not found"

**Solution**:
- Download the model files from the provided links
- Place them in the project root directory (not in the Calculus folder)
- Verify the file names match exactly

#### 6. Port 3000 Already in Use
**Error**: "EADDRINUSE: address already in use :::3000"

**Solution**:
```bash
# Find and kill the process using port 3000
# Windows:
netstat -ano | findstr 3000
taskkill /PID [PID_NUMBER] /F

# Linux/Mac:
lsof -ti:3000 | xargs kill -9

# Or change the port in .env file:
echo "PORT=3001" > .env
```

#### 7. AI Model Not Working
**Symptoms**: Detection returns errors or no results

**Solution**:
1. Verify both model files are in the correct location
2. Check the Python environment has all dependencies
3. Test the AI model directly:
   ```bash
   cd Calculus
   python ai_model.py "../path/to/test/image.jpg"
   ```

---

## 🔧 Development Setup

### For Development Work

1. **Install additional dev dependencies**:
   ```bash
   npm install --save-dev nodemon
   ```

2. **Run in development mode**:
   ```bash
   npm run dev
   ```
   This will auto-restart the server when you make changes.

3. **Environment Variables**:
   Create a `.env` file in the Calculus directory:
   ```bash
   PORT=3000
   ADMIN_USERNAME=admin
   ADMIN_PASSWORD=admin123
   SESSION_SECRET=your-secret-key
   MAX_USER_ID=10
   ```

---

## 📁 Project Structure

```
CalculusSegmenation/
├── segmentyolo.pt              # YOLO model weights
├── best_model.pth              # U-Net model weights  
├── default.yaml                # Configuration file
├── main.py                     # Main Python pipeline
├── segment_teeth.py            # YOLO tooth detection
├── unet_overlay.py             # U-Net calculus detection
├── Calculus/                   # Web application
│   ├── server.js               # Express.js server
│   ├── ai_model.py             # AI integration script
│   ├── package.json            # Node.js dependencies
│   ├── requirements.txt        # Python dependencies
│   ├── public/                 # Static assets
│   ├── views/                  # HTML templates
│   └── uploads/                # User uploaded images
└── dataset/                    # Training data (optional)
```

---

## 🎯 Next Steps

After successful setup:

1. **Test the AI Detection**: 
   - Go to http://localhost:3000/detect
   - Upload a dental image
   - Verify results display correctly

2. **Test the Annotation Tool**:
   - Go to http://localhost:3000/test
   - Select an image to annotate
   - Test the drawing and zoom functionality

3. **Check Admin Panel**:
   - Go to http://localhost:3000/view
   - Login with admin credentials
   - Review submitted annotations

---

## 📞 Support

If you encounter issues not covered in this guide:

1. Check the server logs for error messages
2. Ensure all prerequisites are correctly installed
3. Verify model files are in the correct location
4. Try the troubleshooting steps above

For additional help, refer to the project documentation or contact the development team.
