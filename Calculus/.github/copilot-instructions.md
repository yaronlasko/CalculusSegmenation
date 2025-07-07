# Copilot Instructions for Calculus Detection Domain

<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

This is a Node.js/Express web server project for the calculusdetection.com domain with three main subdomains:

## Project Structure
- `/detect` - AI model endpoint (placeholder for now)
- `/test` - User-based image annotation system with paintbrush tool
- `/annotate` - Direct image annotation without user selection

## Key Features
- Image upload and storage
- Canvas-based paintbrush annotation tool
- Mask saving and overlay functionality
- User selection system for test subdomain
- Completion tracking for annotated images

## Technical Stack
- Backend: Node.js with Express
- Frontend: HTML5, CSS3, JavaScript (Canvas API)
- File handling: Multer for uploads
- Storage: Local file system (images and annotations)

## Development Guidelines
- Follow RESTful API conventions
- Use proper error handling and validation
- Implement responsive design for mobile compatibility
- Ensure proper file upload security measures
- Use semantic HTML and accessible UI components
