// Admin page functionality
document.addEventListener('DOMContentLoaded', function() {
    const refreshBtn = document.getElementById('refreshBtn');
    const testAnnotationsDiv = document.getElementById('testAnnotations');
    const annotateAnnotationsDiv = document.getElementById('annotateAnnotations');
    const testCountSpan = document.getElementById('testCount');
    const annotateCountSpan = document.getElementById('annotateCount');
    const totalCountSpan = document.getElementById('totalCount');
    
    const imageModal = document.getElementById('imageModal');
    const closeModal = document.getElementById('closeImageModal');
    const modalImage = document.getElementById('modalImage');
    const modalImageId = document.getElementById('modalImageId');
    const modalUser = document.getElementById('modalUser');
    const modalSource = document.getElementById('modalSource');
    const modalTimestamp = document.getElementById('modalTimestamp');
    
    // Load annotations on page load
    loadAnnotations();
    
    // Refresh button
    refreshBtn.addEventListener('click', loadAnnotations);
    
    // Modal close handlers
    closeModal.addEventListener('click', () => {
        imageModal.style.display = 'none';
    });
    
    window.addEventListener('click', (e) => {
        if (e.target === imageModal) {
            imageModal.style.display = 'none';
        }
    });
    
    function loadAnnotations() {
        fetch('/api/admin/annotations')
            .then(response => response.json())
            .then(data => {
                displayAnnotations(data);
            })
            .catch(error => {
                console.error('Error loading annotations:', error);
            });
    }
    
    function displayAnnotations(data) {
        const testAnnotations = data.test || [];
        const annotateAnnotations = data.annotate || [];
        
        // Update counts
        testCountSpan.textContent = testAnnotations.length;
        annotateCountSpan.textContent = annotateAnnotations.length;
        totalCountSpan.textContent = testAnnotations.length + annotateAnnotations.length;
        
        // Display test annotations
        if (testAnnotations.length === 0) {
            testAnnotationsDiv.innerHTML = '<div class="empty-state">No test annotations found</div>';
        } else {
            testAnnotationsDiv.innerHTML = '';
            testAnnotations.forEach(annotation => {
                const annotationElement = createAnnotationElement(annotation);
                testAnnotationsDiv.appendChild(annotationElement);
            });
        }
        
        // Display annotate annotations
        if (annotateAnnotations.length === 0) {
            annotateAnnotationsDiv.innerHTML = '<div class="empty-state">No annotate annotations found</div>';
        } else {
            annotateAnnotationsDiv.innerHTML = '';
            annotateAnnotations.forEach(annotation => {
                const annotationElement = createAnnotationElement(annotation);
                annotateAnnotationsDiv.appendChild(annotationElement);
            });
        }
    }
    
    function createAnnotationElement(annotation) {
        const div = document.createElement('div');
        div.className = 'annotation-item';
        
        const timestamp = new Date(annotation.timestamp).toLocaleString();
        
        div.innerHTML = `
            <img src="/uploads/annotations/${annotation.filename}" alt="Annotation" class="annotation-preview">
            <div class="annotation-info">
                <h4>${annotation.imageId}</h4>
                <div class="annotation-meta">Created: ${timestamp}</div>
                <div class="annotation-meta">File: ${annotation.filename}</div>
                <span class="user-badge">User: ${annotation.userId}</span>
                <span class="source-badge ${annotation.source}">${annotation.source}</span>
            </div>
        `;
        
        div.addEventListener('click', () => {
            showAnnotationModal(annotation);
        });
        
        return div;
    }
    
    function showAnnotationModal(annotation) {
        modalImageId.textContent = annotation.imageId;
        modalUser.textContent = annotation.userId;
        modalSource.textContent = annotation.source;
        modalTimestamp.textContent = new Date(annotation.timestamp).toLocaleString();
        
        // Create a canvas to overlay the annotation on the original image
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        
        // First load the original image
        const originalImg = new Image();
        originalImg.crossOrigin = 'anonymous';
        originalImg.onload = function() {
            // Set canvas dimensions
            canvas.width = originalImg.width;
            canvas.height = originalImg.height;
            
            // Draw original image
            ctx.drawImage(originalImg, 0, 0);
            
            // Load and draw annotation with transparency
            const annotationImg = new Image();
            annotationImg.crossOrigin = 'anonymous';
            annotationImg.onload = function() {
                // Set global alpha for transparency
                ctx.globalAlpha = 0.6;
                ctx.drawImage(annotationImg, 0, 0, canvas.width, canvas.height);
                ctx.globalAlpha = 1.0; // Reset alpha
                
                // Set the canvas result as the modal image
                modalImage.src = canvas.toDataURL();
            };
            annotationImg.onerror = function() {
                console.error('Error loading annotation image');
                modalImage.src = originalImg.src; // Fallback to original image
            };
            annotationImg.src = `/uploads/annotations/${annotation.filename}`;
        };
        
        originalImg.onerror = function() {
            console.error('Error loading original image:', originalImagePath);
            console.log('Annotation data:', annotation);
            // Fallback: just show the annotation
            modalImage.src = `/uploads/annotations/${annotation.filename}`;
        };
        
        // Get original image path based on annotation source
        // Handle cases where originalImage field might be missing
        let originalImagePath = '';
        if (annotation.originalImage) {
            // New annotations have originalImage field
            if (annotation.source === 'test') {
                originalImagePath = `/uploads/test-images/${annotation.originalImage}`;
            } else {
                originalImagePath = `/uploads/annotate-images/${annotation.originalImage}`;
            }
        } else {
            // Fallback for old annotations without originalImage field
            // Try to guess the original image name from the imageId
            if (annotation.source === 'test') {
                // For test images, we'll try a few common patterns
                // Since we don't know the exact filename, we'll use a placeholder
                // and let the error handler deal with it
                originalImagePath = `/uploads/test-images/placeholder.jpg`;
            } else {
                // For annotate images, similar approach
                originalImagePath = `/uploads/annotate-images/placeholder.jpg`;
            }
        }
        
        originalImg.src = originalImagePath;
        
        imageModal.style.display = 'block';
    }
    
    // Admin zoom functionality
    let adminZoomLevel = 1;
    const adminImageWrapper = document.getElementById('adminImageWrapper');
    const adminZoomSlider = document.getElementById('adminZoomSlider');
    const adminZoomValue = document.getElementById('adminZoomValue');
    const adminZoomReset = document.getElementById('adminZoomReset');
    
    function setupAdminZoom() {
        if (adminZoomSlider && adminZoomValue && adminZoomReset && adminImageWrapper) {
            // Zoom slider
            adminZoomSlider.addEventListener('input', (e) => {
                setAdminZoom(parseInt(e.target.value));
            });
            
            // Zoom reset
            adminZoomReset.addEventListener('click', () => {
                setAdminZoom(100);
                adminZoomSlider.value = 100;
            });
            
            // Mouse wheel zoom
            adminImageWrapper.addEventListener('wheel', (e) => {
                e.preventDefault();
                const delta = e.deltaY > 0 ? -10 : 10;
                const newZoom = Math.max(50, Math.min(300, adminZoomLevel * 100 + delta));
                setAdminZoom(newZoom);
                adminZoomSlider.value = newZoom;
            });
        }
    }
    
    function setAdminZoom(zoomPercent) {
        adminZoomLevel = zoomPercent / 100;
        if (adminImageWrapper) {
            adminImageWrapper.style.transform = `scale(${adminZoomLevel})`;
        }
        
        if (adminZoomValue) {
            adminZoomValue.textContent = `${zoomPercent}%`;
        }
    }
    
    // Setup admin zoom when modal opens
    setupAdminZoom();
});
