const express = require('express');
const multer = require('multer');
const cors = require('cors');
const helmet = require('helmet');
const morgan = require('morgan');
const path = require('path');
const fs = require('fs');
const session = require('express-session');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;

// Express session for admin authentication
app.use(session({
    secret: process.env.SESSION_SECRET || 'calculus-detection-secret',
    resave: false,
    saveUninitialized: false,
    cookie: { secure: false } // Set to true in production with HTTPS
}));

// Admin credentials (in production, use environment variables or database)
const ADMIN_USERNAME = process.env.ADMIN_USERNAME || 'admin';
const ADMIN_PASSWORD = process.env.ADMIN_PASSWORD || 'admin123';

// Configuration constants
const MAX_USER_ID = parseInt(process.env.MAX_USER_ID) || 10;

// Middleware
app.use(helmet({
    contentSecurityPolicy: {
        directives: {
            defaultSrc: ["'self'"],
            styleSrc: ["'self'", "'unsafe-inline'"],
            scriptSrc: ["'self'", "'unsafe-inline'"],
            imgSrc: ["'self'", "data:", "blob:"],
        },
    },
}));
app.use(cors());
app.use(morgan('combined'));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Serve static files
app.use('/static', express.static(path.join(__dirname, 'public')));
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

// Create necessary directories
const dirs = ['uploads', 'uploads/images', 'uploads/test-images', 'uploads/annotate-images', 'uploads/annotations', 'public/css', 'public/js'];
dirs.forEach(dir => {
    if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
    }
});

// Configure multer for file uploads
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, 'uploads/images/');
    },
    filename: (req, file, cb) => {
        cb(null, Date.now() + '-' + file.originalname);
    }
});

const upload = multer({ 
    storage: storage,
    limits: { fileSize: 10 * 1024 * 1024 }, // 10MB limit
    fileFilter: (req, file, cb) => {
        if (file.mimetype.startsWith('image/')) {
            cb(null, true);
        } else {
            cb(new Error('Only image files are allowed!'), false);
        }
    }
});

// Sample images data (in production, this would be in a database)
let testImages = [];
let annotateImages = [];
let annotations = {};

// Initialize with actual images from uploads directory
function initializeImages() {
    // Load test images
    const testImagesDir = path.join(__dirname, 'uploads', 'test-images');
    if (fs.existsSync(testImagesDir)) {
        const testImageFiles = fs.readdirSync(testImagesDir)
            .filter(file => /\.(jpg|jpeg|png|gif)$/i.test(file))
            .sort();
        
        testImageFiles.forEach((file, index) => {
            testImages.push({
                id: `test-${index + 1}`,
                name: file,
                path: `/uploads/test-images/${file}`,
                annotated: false,
                completedBy: []
            });
        });
    }
    
    // Load annotate images
    const annotateImagesDir = path.join(__dirname, 'uploads', 'annotate-images');
    if (fs.existsSync(annotateImagesDir)) {
        const annotateImageFiles = fs.readdirSync(annotateImagesDir)
            .filter(file => /\.(jpg|jpeg|png|gif)$/i.test(file))
            .sort();
        
        annotateImageFiles.forEach((file, index) => {
            annotateImages.push({
                id: `annotate-${index + 1}`,
                name: file,
                path: `/uploads/annotate-images/${file}`,
                annotated: false,
                completedBy: []
            });
        });
    }
    
    console.log(`Loaded ${testImages.length} test images and ${annotateImages.length} annotate images`);
}

initializeImages();

// Routes

// Main domain - Landing page
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'views', 'index.html'));
});

// Detect subdomain - AI model placeholder
app.get('/detect', (req, res) => {
    res.sendFile(path.join(__dirname, 'views', 'detect.html'));
});

app.post('/detect', upload.single('image'), (req, res) => {
    if (!req.file) {
        return res.status(400).json({ error: 'No image uploaded' });
    }
    
    // Placeholder for AI model processing
    res.json({
        message: 'AI model processing placeholder',
        filename: req.file.filename,
        result: 'This is where the AI detection results would appear'
    });
});

// Test subdomain - User selection and annotation
app.get('/test', (req, res) => {
    res.sendFile(path.join(__dirname, 'views', 'test.html'));
});

app.get('/test/user/:userId', (req, res) => {
    const userId = req.params.userId;
    
    // Validate user ID
    if (!userId || isNaN(userId) || userId < 1 || userId > MAX_USER_ID) {
        return res.redirect('/test?error=invalid_user_id');
    }
    
    res.sendFile(path.join(__dirname, 'views', 'test-annotation.html'));
});

app.get('/api/config', (req, res) => {
    res.json({
        maxUserId: MAX_USER_ID
    });
});

app.get('/api/test/images/:userId', (req, res) => {
    const userId = req.params.userId;
    
    // Validate user ID
    if (!userId || isNaN(userId) || userId < 1 || userId > MAX_USER_ID) {
        return res.status(400).json({ error: `User ID must be between 1 and ${MAX_USER_ID}` });
    }
    
    const userImages = testImages.filter(img => !img.completedBy.includes(userId));
    res.json(userImages.slice(0, 20));
});

// Annotate subdomain - Direct annotation
app.get('/annotate', (req, res) => {
    res.sendFile(path.join(__dirname, 'views', 'annotate.html'));
});

app.get('/api/annotate/images', (req, res) => {
    const unannotatedImages = annotateImages.filter(img => !img.annotated);
    res.json(unannotatedImages);
});

// API endpoints for annotations
app.post('/api/annotations/:imageId', (req, res) => {
    const { imageId } = req.params;
    const { maskData, userId } = req.body;
    
    if (!maskData) {
        return res.status(400).json({ error: 'Mask data is required' });
    }
    
    try {
        // Save annotation
        const annotationPath = path.join(__dirname, 'uploads', 'annotations', `${imageId}-${Date.now()}.png`);
        
        // Convert base64 to image file
        const base64Data = maskData.replace(/^data:image\/png;base64,/, '');
        fs.writeFileSync(annotationPath, base64Data, 'base64');
        
        // Update image data
        let imageIndex = -1;
        let imageArray = null;
        
        if (imageId.startsWith('test-')) {
            imageIndex = testImages.findIndex(img => img.id === imageId);
            imageArray = testImages;
        } else if (imageId.startsWith('annotate-')) {
            imageIndex = annotateImages.findIndex(img => img.id === imageId);
            imageArray = annotateImages;
        }
        
        if (imageIndex !== -1 && imageArray) {
            if (userId) {
                if (!imageArray[imageIndex].completedBy.includes(userId)) {
                    imageArray[imageIndex].completedBy.push(userId);
                }
            } else {
                imageArray[imageIndex].annotated = true;
            }
        }
        
        // Function to get original image name
        function getOriginalImageName(imageId) {
            if (imageId.startsWith('test-')) {
                const img = testImages.find(img => img.id === imageId);
                return img ? img.name : '';
            } else if (imageId.startsWith('annotate-')) {
                const img = annotateImages.find(img => img.id === imageId);
                return img ? img.name : '';
            }
            return '';
        }
        
        // Store annotation metadata
        const annotationData = {
            imageId: imageId,
            userId: userId || 'anonymous',
            timestamp: new Date().toISOString(),
            filename: path.basename(annotationPath),
            source: imageId.startsWith('test-') ? 'test' : 'annotate',
            originalImage: getOriginalImageName(imageId)
        };
        
        // Save annotation metadata
        const metadataPath = path.join(__dirname, 'uploads', 'annotations', `${imageId}-${Date.now()}.json`);
        fs.writeFileSync(metadataPath, JSON.stringify(annotationData, null, 2));
        
        res.json({ 
            message: 'Annotation saved successfully',
            annotationPath: annotationPath,
            metadata: annotationData
        });
    } catch (error) {
        console.error('Error saving annotation:', error);
        res.status(500).json({ error: 'Failed to save annotation: ' + error.message });
    }
});

app.get('/api/annotations/:imageId', (req, res) => {
    const { imageId } = req.params;
    const annotationDir = path.join(__dirname, 'uploads', 'annotations');
    
    if (!fs.existsSync(annotationDir)) {
        return res.json([]);
    }
    
    const files = fs.readdirSync(annotationDir)
        .filter(file => file.startsWith(`${imageId}-`))
        .map(file => ({
            filename: file,
            path: `/uploads/annotations/${file}`
        }));
    
    res.json(files);
});

// Admin routes
app.post('/admin/login', (req, res) => {
    const { username, password } = req.body;
    
    if (username === ADMIN_USERNAME && password === ADMIN_PASSWORD) {
        req.session.isAdmin = true;
        res.json({ success: true });
    } else {
        res.json({ success: false, error: 'Invalid credentials' });
    }
});

// Admin middleware
function requireAdmin(req, res, next) {
    if (req.session.isAdmin) {
        next();
    } else {
        res.redirect('/');
    }
}

app.get('/view', requireAdmin, (req, res) => {
    res.sendFile(path.join(__dirname, 'views', 'admin.html'));
});

app.get('/api/admin/annotations', requireAdmin, (req, res) => {
    const annotationDir = path.join(__dirname, 'uploads', 'annotations');
    
    if (!fs.existsSync(annotationDir)) {
        return res.json({ test: [], annotate: [] });
    }
    
    const files = fs.readdirSync(annotationDir);
    const annotations = { test: [], annotate: [] };
    
    files.forEach(file => {
        if (file.endsWith('.json')) {
            try {
                const filePath = path.join(annotationDir, file);
                const data = JSON.parse(fs.readFileSync(filePath, 'utf8'));
                
                if (data.source === 'test') {
                    annotations.test.push(data);
                } else if (data.source === 'annotate') {
                    annotations.annotate.push(data);
                }
            } catch (error) {
                console.error('Error reading annotation file:', file, error);
            }
        }
    });
    
    // Sort by timestamp (newest first)
    annotations.test.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
    annotations.annotate.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
    
    res.json(annotations);
});

// Admin endpoint to get all test images
app.get('/api/admin/test-images', requireAdmin, (req, res) => {
    res.json(testImages);
});

// Error handling middleware
app.use((error, req, res, next) => {
    if (error instanceof multer.MulterError) {
        if (error.code === 'LIMIT_FILE_SIZE') {
            return res.status(400).json({ error: 'File too large' });
        }
    }
    
    console.error(error);
    res.status(500).json({ error: 'Something went wrong!' });
});

// 404 handler
app.use((req, res) => {
    res.status(404).sendFile(path.join(__dirname, 'views', '404.html'));
});

// Start server
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
    console.log(`Visit http://localhost:${PORT} to view the application`);
});

module.exports = app;
