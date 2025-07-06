// Annotation functionality shared between test and direct annotation
class AnnotationTool {
    constructor(imageCanvas, annotationCanvas) {
        this.imageCanvas = imageCanvas;
        this.annotationCanvas = annotationCanvas;
        this.imageCtx = imageCanvas.getContext('2d');
        this.annotationCtx = annotationCanvas.getContext('2d');
        
        this.isDrawing = false;
        this.brushSize = 10;
        this.currentImage = null;
        this.brushIndicator = null;
        this.zoomLevel = 1;
        this.canvasWrapper = null;
        
        this.setupEventListeners();
        this.createBrushIndicator();
        this.setupZoomControls();
    }
    
    setupEventListeners() {
        // Mouse events
        this.annotationCanvas.addEventListener('mousedown', this.startDrawing.bind(this));
        this.annotationCanvas.addEventListener('mousemove', this.draw.bind(this));
        this.annotationCanvas.addEventListener('mouseup', this.stopDrawing.bind(this));
        this.annotationCanvas.addEventListener('mouseout', this.stopDrawing.bind(this));
        
        // Touch events for mobile
        this.annotationCanvas.addEventListener('touchstart', this.handleTouch.bind(this));
        this.annotationCanvas.addEventListener('touchmove', this.handleTouch.bind(this));
        this.annotationCanvas.addEventListener('touchend', this.stopDrawing.bind(this));
        
        // Brush cursor
        this.annotationCanvas.addEventListener('mousemove', this.updateBrushCursor.bind(this));
        this.annotationCanvas.addEventListener('mouseenter', this.showBrushCursor.bind(this));
        this.annotationCanvas.addEventListener('mouseleave', this.hideBrushCursor.bind(this));
    }
    
    createBrushIndicator() {
        this.brushIndicator = document.createElement('div');
        this.brushIndicator.className = 'brush-indicator';
        this.brushIndicator.style.display = 'none';
        document.body.appendChild(this.brushIndicator);
    }
    
    updateBrushCursor(e) {
        if (this.brushIndicator) {
            const rect = this.annotationCanvas.getBoundingClientRect();
            const x = e.clientX;
            const y = e.clientY;
            
            const scaledSize = this.brushSize * 2 * this.zoomLevel;
            this.brushIndicator.style.left = x + 'px';
            this.brushIndicator.style.top = y + 'px';
            this.brushIndicator.style.width = scaledSize + 'px';
            this.brushIndicator.style.height = scaledSize + 'px';
            this.brushIndicator.style.display = 'block';
        }
    }
    
    showBrushCursor() {
        if (this.brushIndicator) {
            this.brushIndicator.style.display = 'block';
        }
        this.annotationCanvas.classList.add('brush-cursor');
    }
    
    hideBrushCursor() {
        if (this.brushIndicator) {
            this.brushIndicator.style.display = 'none';
        }
        this.annotationCanvas.classList.remove('brush-cursor');
    }
    
    handleTouch(e) {
        e.preventDefault();
        const touch = e.touches[0];
        const mouseEvent = new MouseEvent(e.type === 'touchstart' ? 'mousedown' : 'mousemove', {
            clientX: touch.clientX,
            clientY: touch.clientY
        });
        this.annotationCanvas.dispatchEvent(mouseEvent);
    }
    
    startDrawing(e) {
        this.isDrawing = true;
        this.lastX = undefined;
        this.lastY = undefined;
        this.draw(e);
    }
    
    draw(e) {
        if (!this.isDrawing) return;
        
        const rect = this.annotationCanvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        // Scale coordinates to match canvas resolution and account for zoom
        const scaleX = this.annotationCanvas.width / rect.width;
        const scaleY = this.annotationCanvas.height / rect.height;
        const scaledX = (x * scaleX) / this.zoomLevel;
        const scaledY = (y * scaleY) / this.zoomLevel;
        
        this.annotationCtx.globalCompositeOperation = 'source-over';
        // Save annotation with full alpha (1.0) but display with transparency
        this.annotationCtx.fillStyle = 'rgba(255, 0, 0, 1.0)';
        this.annotationCtx.beginPath();
        this.annotationCtx.arc(scaledX, scaledY, this.brushSize, 0, 2 * Math.PI);
        this.annotationCtx.fill();
        
        // Store previous position for smooth drawing
        if (this.lastX !== undefined && this.lastY !== undefined) {
            this.annotationCtx.beginPath();
            this.annotationCtx.moveTo(this.lastX, this.lastY);
            this.annotationCtx.lineTo(scaledX, scaledY);
            this.annotationCtx.lineWidth = this.brushSize * 2;
            this.annotationCtx.lineCap = 'round';
            this.annotationCtx.strokeStyle = 'rgba(255, 0, 0, 1.0)';
            this.annotationCtx.stroke();
        }
        
        this.lastX = scaledX;
        this.lastY = scaledY;
    }
    
    stopDrawing() {
        this.isDrawing = false;
        this.lastX = undefined;
        this.lastY = undefined;
    }
    
    setBrushSize(size) {
        this.brushSize = size;
        this.updateBrushIndicatorSize();
    }
    
    clearCanvas() {
        this.annotationCtx.clearRect(0, 0, this.annotationCanvas.width, this.annotationCanvas.height);
    }
    
    loadImage(imageSrc) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => {
                this.currentImage = img;
                
                // Calculate display dimensions (max 800px width, maintain aspect ratio)
                const maxWidth = 800;
                const maxHeight = 600;
                let displayWidth = img.width;
                let displayHeight = img.height;
                
                if (displayWidth > maxWidth) {
                    displayHeight = (displayHeight * maxWidth) / displayWidth;
                    displayWidth = maxWidth;
                }
                
                if (displayHeight > maxHeight) {
                    displayWidth = (displayWidth * maxHeight) / displayHeight;
                    displayHeight = maxHeight;
                }
                
                // Set canvas dimensions
                this.imageCanvas.width = displayWidth;
                this.imageCanvas.height = displayHeight;
                this.annotationCanvas.width = displayWidth;
                this.annotationCanvas.height = displayHeight;
                
                // Set canvas style dimensions
                this.imageCanvas.style.width = displayWidth + 'px';
                this.imageCanvas.style.height = displayHeight + 'px';
                this.annotationCanvas.style.width = displayWidth + 'px';
                this.annotationCanvas.style.height = displayHeight + 'px';
                
                // Draw image on image canvas
                this.imageCtx.drawImage(img, 0, 0, displayWidth, displayHeight);
                
                // Clear annotation canvas
                this.clearCanvas();
                
                resolve();
            };
            img.onerror = reject;
            img.src = imageSrc;
        });
    }
    
    getAnnotationData() {
        return this.annotationCanvas.toDataURL('image/png');
    }
    
    destroy() {
        if (this.brushIndicator) {
            document.body.removeChild(this.brushIndicator);
        }
    }
    
    setupZoomControls() {
        // Find zoom controls
        const zoomSlider = document.getElementById('zoomSlider');
        const zoomValue = document.getElementById('zoomValue');
        const zoomReset = document.getElementById('zoomReset');
        
        if (zoomSlider && zoomValue && zoomReset) {
            // Zoom slider
            zoomSlider.addEventListener('input', (e) => {
                const zoom = parseFloat(e.target.value) / 100;
                this.setZoom(zoom);
                zoomValue.textContent = Math.round(zoom * 100) + '%';
            });
            
            // Zoom reset
            zoomReset.addEventListener('click', () => {
                this.setZoom(1);
                zoomSlider.value = 100;
                zoomValue.textContent = '100%';
            });
            
            // Mouse wheel zoom
            this.annotationCanvas.addEventListener('wheel', (e) => {
                if (e.ctrlKey) {
                    e.preventDefault();
                    const delta = e.deltaY > 0 ? -0.1 : 0.1;
                    const newZoom = Math.max(0.5, Math.min(3, this.zoomLevel + delta));
                    this.setZoom(newZoom);
                    zoomSlider.value = newZoom * 100;
                    zoomValue.textContent = Math.round(newZoom * 100) + '%';
                }
            });
        }
    }
    
    setZoom(zoom) {
        this.zoomLevel = zoom;
        
        // Find canvas wrapper - try multiple approaches
        let canvasWrapper = this.annotationCanvas.parentElement;
        if (!canvasWrapper || !canvasWrapper.classList.contains('canvas-wrapper')) {
            canvasWrapper = document.getElementById('canvasWrapper');
        }
        if (!canvasWrapper) {
            canvasWrapper = document.querySelector('.canvas-wrapper');
        }
        
        if (canvasWrapper) {
            canvasWrapper.style.transform = `scale(${zoom})`;
            canvasWrapper.style.transformOrigin = 'top left';
        } else {
            console.warn('Canvas wrapper not found for zoom');
        }
        
        // Update brush indicator size
        this.updateBrushIndicatorSize();
    }
    
    updateBrushIndicatorSize() {
        if (this.brushIndicator) {
            const scaledSize = this.brushSize * 2 * this.zoomLevel;
            this.brushIndicator.style.width = scaledSize + 'px';
            this.brushIndicator.style.height = scaledSize + 'px';
        }
    }
}

// Modal functionality
class AnnotationModal {
    constructor() {
        this.modal = document.getElementById('annotationModal');
        this.closeBtn = document.getElementById('closeModal');
        this.brushSizeSlider = document.getElementById('brushSize');
        this.brushSizeValue = document.getElementById('brushSizeValue');
        this.clearBtn = document.getElementById('clearCanvas');
        this.saveBtn = document.getElementById('saveAnnotation');
        
        this.imageCanvas = document.getElementById('imageCanvas');
        this.annotationCanvas = document.getElementById('annotationCanvas');
        
        this.annotationTool = null;
        this.currentImageData = null;
        this.onSave = null;
        
        this.setupEventListeners();
    }
    
    setupEventListeners() {
        this.closeBtn.addEventListener('click', () => this.close());
        this.modal.addEventListener('click', (e) => {
            if (e.target === this.modal) this.close();
        });
        
        this.brushSizeSlider.addEventListener('input', (e) => {
            const size = parseInt(e.target.value);
            this.brushSizeValue.textContent = size;
            if (this.annotationTool) {
                this.annotationTool.setBrushSize(size);
            }
        });
        
        this.clearBtn.addEventListener('click', () => {
            if (this.annotationTool) {
                this.annotationTool.clearCanvas();
            }
        });
        
        this.saveBtn.addEventListener('click', () => this.saveAnnotation());
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (this.modal.style.display === 'block') {
                if (e.key === 'Escape') this.close();
                if (e.key === 'Enter' && e.ctrlKey) this.saveAnnotation();
            }
        });
    }
    
    open(imageData, onSave) {
        this.currentImageData = imageData;
        this.onSave = onSave;
        
        this.modal.style.display = 'block';
        
        // Initialize annotation tool
        this.annotationTool = new AnnotationTool(this.imageCanvas, this.annotationCanvas);
        this.annotationTool.setBrushSize(parseInt(this.brushSizeSlider.value));
        
        // Load image
        this.annotationTool.loadImage(imageData.path)
            .catch(error => {
                console.error('Error loading image:', error);
                alert('Error loading image');
                this.close();
            });
    }
    
    close() {
        this.modal.style.display = 'none';
        if (this.annotationTool) {
            this.annotationTool.destroy();
            this.annotationTool = null;
        }
        this.currentImageData = null;
        this.onSave = null;
    }
    
    saveAnnotation() {
        if (!this.annotationTool || !this.currentImageData) {
            alert('No annotation to save');
            return;
        }
        
        const maskData = this.annotationTool.getAnnotationData();
        
        if (this.onSave) {
            this.onSave(this.currentImageData, maskData);
        }
        
        this.close();
    }
}
