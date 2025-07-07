// Test page functionality
document.addEventListener('DOMContentLoaded', function() {
    const userForm = document.getElementById('userForm');
    const userIdInput = document.getElementById('userId');
    let maxUserId = 10; // Default value
    
    // Load configuration
    fetch('/api/config')
        .then(response => response.json())
        .then(config => {
            maxUserId = config.maxUserId;
            userIdInput.setAttribute('max', maxUserId);
            const helpText = document.querySelector('.input-group small');
            if (helpText) {
                helpText.textContent = `Please select a user ID between 1 and ${maxUserId}`;
            }
        })
        .catch(error => {
            console.error('Error loading config:', error);
        });
    
    // Handle form submission
    userForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const userId = parseInt(userIdInput.value);
        
        if (!userId || isNaN(userId) || userId < 1 || userId > maxUserId) {
            alert(`Please enter a valid user ID between 1 and ${maxUserId}`);
            return;
        }
        
        // Redirect to user-specific test page
        window.location.href = `/test/user/${userId}`;
    });
    
    // Auto-focus on user ID input
    userIdInput.focus();
});
