// Detect page functionality
document.addEventListener('DOMContentLoaded', function() {
    const detectForm = document.getElementById('detectForm');
    const uploadArea = document.getElementById('uploadArea');
    const imageInput = document.getElementById('imageInput');
    const resultsSection = document.getElementById('results');
    const resultContent = document.getElementById('resultContent');
    
    // Handle file drag and drop
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadArea.style.backgroundColor = '#f8f9ff';
        uploadArea.style.borderColor = '#667eea';
    });
    
    uploadArea.addEventListener('dragleave', function(e) {
        e.preventDefault();
        uploadArea.style.backgroundColor = '';
        uploadArea.style.borderColor = '#ddd';
    });
    
    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadArea.style.backgroundColor = '';
        uploadArea.style.borderColor = '#ddd';
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            imageInput.files = files;
            updateUploadArea(files[0]);
        }
    });
    
    // Handle file input change
    imageInput.addEventListener('change', function(e) {
        if (e.target.files.length > 0) {
            updateUploadArea(e.target.files[0]);
        }
    });
    
    // Update upload area with selected file
    function updateUploadArea(file) {
        uploadArea.innerHTML = `
            <p>Selected: ${file.name}</p>
            <p>Size: ${(file.size / 1024 / 1024).toFixed(2)} MB</p>
        `;
    }
    
    // Handle form submission
    detectForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const formData = new FormData();
        formData.append('image', imageInput.files[0]);
        
        // Show loading state
        resultContent.innerHTML = '<div class="loading">Processing image...</div>';
        resultsSection.style.display = 'block';
        
        fetch('/detect', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }
            
            resultContent.innerHTML = `
                <div class="result-item">
                    <h4>Detection Results</h4>
                    <p><strong>Filename:</strong> ${data.filename}</p>
                    <p><strong>Result:</strong> ${data.result}</p>
                    <div class="placeholder-notice">
                        <p><em>Note: This is a placeholder. The actual AI model will be integrated here.</em></p>
                    </div>
                </div>
            `;
        })
        .catch(error => {
            resultContent.innerHTML = `
                <div class="error">
                    <h4>Error</h4>
                    <p>${error.message}</p>
                </div>
            `;
        });
    });
});
