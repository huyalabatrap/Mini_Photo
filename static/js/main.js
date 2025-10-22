/**
 * Segmentation Studio - Main JavaScript
 * ES6 IIFE Module for image preview and form handling
 */

(function() {
    'use strict';

    // DOM elements cache
    const elements = {
        uploadForm: null,
        imageInput: null,
        imagePreview: null,
        previewImg: null,
        fileInputText: null,
        submitButton: null,
        buttonText: null,
        buttonLoading: null
    };

    /**
     * Initialize DOM element references
     */
    function initElements() {
        elements.uploadForm = document.getElementById('uploadForm');
        elements.imageInput = document.getElementById('imageInput');
        elements.imagePreview = document.getElementById('imagePreview');
        elements.previewImg = document.getElementById('previewImg');
        elements.fileInputText = document.querySelector('.file-input-text');
        elements.submitButton = document.querySelector('.btn-primary');
        elements.buttonText = document.querySelector('.btn-text');
        elements.buttonLoading = document.querySelector('.btn-loading');

        // Validate required elements
        if (!elements.uploadForm || !elements.imageInput) {
            console.warn('Required form elements not found');
            return false;
        }

        return true;
    }

    /**
     * Handle file input change event
     * @param {Event} event - Change event from file input
     */
    function handleFileSelect(event) {
        const file = event.target.files[0];
        
        if (!file) {
            hideImagePreview();
            return;
        }

        // Validate file type
        if (!isValidImageFile(file)) {
            showError('Định dạng file không được hỗ trợ. Vui lòng chọn file ảnh hợp lệ.');
            hideImagePreview();
            return;
        }

        // Validate file size (10MB limit)
        const maxSize = 10 * 1024 * 1024; // 10MB in bytes
        if (file.size > maxSize) {
            showError('File quá lớn. Kích thước tối đa là 10MB.');
            hideImagePreview();
            return;
        }

        // Update file input display
        updateFileInputDisplay(file.name);

        // Show image preview
        showImagePreview(file);
    }

    /**
     * Validate if file is a supported image format
     * @param {File} file - File object to validate
     * @returns {boolean} - True if valid image file
     */
    function isValidImageFile(file) {
        const allowedTypes = [
            'image/png',
            'image/jpeg',
            'image/jpg',
            'image/bmp',
            'image/tiff',
            'image/tif'
        ];

        return allowedTypes.includes(file.type.toLowerCase());
    }

    /**
     * Update file input display text
     * @param {string} filename - Name of selected file
     */
    function updateFileInputDisplay(filename) {
        if (elements.fileInputText) {
            // Truncate long filenames
            const displayName = filename.length > 30 
                ? filename.substring(0, 27) + '...' 
                : filename;
            
            elements.fileInputText.textContent = displayName;
        }
    }

    /**
     * Show image preview using FileReader
     * @param {File} file - Image file to preview
     */
    function showImagePreview(file) {
        if (!elements.imagePreview || !elements.previewImg) {
            return;
        }

        const reader = new FileReader();

        reader.onload = function(e) {
            elements.previewImg.src = e.target.result;
            elements.previewImg.alt = `Preview: ${file.name}`;
            elements.imagePreview.classList.remove('hidden');
            
            // Add fade-in animation
            requestAnimationFrame(() => {
                elements.imagePreview.style.opacity = '1';
            });
        };

        reader.onerror = function() {
            showError('Không thể đọc file ảnh. Vui lòng thử lại.');
            hideImagePreview();
        };

        // Read the image file as data URL
        reader.readAsDataURL(file);
    }

    /**
     * Hide image preview
     */
    function hideImagePreview() {
        if (elements.imagePreview) {
            elements.imagePreview.style.opacity = '0';
            setTimeout(() => {
                elements.imagePreview.classList.add('hidden');
            }, 200);
        }

        if (elements.fileInputText) {
            elements.fileInputText.textContent = 'Chọn file ảnh...';
        }
    }

    /**
     * Handle form submission
     * @param {Event} event - Form submit event
     */
    function handleFormSubmit(event) {
        // Don't prevent default - let server handle redirect
        
        // Validate form before submission
        if (!validateForm()) {
            event.preventDefault();
            return false;
        }

        // Update button state to loading
        setButtonLoading(true);

        // Add a timeout fallback in case server doesn't respond
        setTimeout(() => {
            setButtonLoading(false);
        }, 30000); // 30 seconds timeout
    }

    /**
     * Validate form before submission
     * @returns {boolean} - True if form is valid
     */
    function validateForm() {
        // Check if image is selected
        if (!elements.imageInput || !elements.imageInput.files.length) {
            showError('Vui lòng chọn ảnh để xử lý.');
            return false;
        }

        // Check if method is selected
        const methodSelect = document.getElementById('methodSelect');
        if (!methodSelect || !methodSelect.value) {
            showError('Vui lòng chọn phương pháp phân đoạn.');
            return false;
        }

        return true;
    }

    /**
     * Set button loading state
     * @param {boolean} isLoading - Whether button should show loading state
     */
    function setButtonLoading(isLoading) {
        if (!elements.submitButton) return;

        if (isLoading) {
            elements.submitButton.disabled = true;
            elements.submitButton.classList.add('loading');
            
            if (elements.buttonText) {
                elements.buttonText.textContent = 'Đang xử lý...';
            }
            
            if (elements.buttonLoading) {
                elements.buttonLoading.classList.remove('hidden');
            }
        } else {
            elements.submitButton.disabled = false;
            elements.submitButton.classList.remove('loading');
            
            if (elements.buttonText) {
                elements.buttonText.textContent = 'Phân đoạn';
            }
            
            if (elements.buttonLoading) {
                elements.buttonLoading.classList.add('hidden');
            }
        }
    }

    /**
     * Show error message to user
     * @param {string} message - Error message to display
     */
    function showError(message) {
        // Create a temporary flash message
        const flashContainer = document.querySelector('.flash-container');
        if (!flashContainer) return;

        const flashMessage = document.createElement('div');
        flashMessage.className = 'flash-message flash-error';
        flashMessage.innerHTML = `
            <div class="flash-content">
                <span class="flash-icon">❌</span>
                <span class="flash-text">${message}</span>
                <button class="flash-close" onclick="this.parentElement.parentElement.remove()">
                    <span>×</span>
                </button>
            </div>
        `;

        flashContainer.appendChild(flashMessage);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (flashMessage.parentNode) {
                flashMessage.remove();
            }
        }, 5000);
    }

    /**
     * Handle advanced parameters visibility
     */
    function handleAdvancedParams() {
        const methodSelect = document.getElementById('methodSelect');
        const advancedParams = document.getElementById('advancedParams');
        
        if (!methodSelect || !advancedParams) return;

        methodSelect.addEventListener('change', function() {
            const method = this.value;
            const kmeansInput = document.getElementById('kmeansK');
            const cannyLowInput = document.getElementById('cannyLow');
            const cannyHighInput = document.getElementById('cannyHigh');

            // Reset all inputs
            [kmeansInput, cannyLowInput, cannyHighInput].forEach(input => {
                if (input) {
                    input.disabled = true;
                    input.style.opacity = '0.5';
                }
            });

            // Enable relevant inputs based on method
            if (method === 'kmeans' && kmeansInput) {
                kmeansInput.disabled = false;
                kmeansInput.style.opacity = '1';
            }

            if (method === 'canny' && cannyLowInput && cannyHighInput) {
                cannyLowInput.disabled = false;
                cannyHighInput.disabled = false;
                cannyLowInput.style.opacity = '1';
                cannyHighInput.style.opacity = '1';
            }
        });

        // Trigger initial state
        methodSelect.dispatchEvent(new Event('change'));
    }

    /**
     * Initialize event listeners
     */
    function initEventListeners() {
        // File input change event
        if (elements.imageInput) {
            elements.imageInput.addEventListener('change', handleFileSelect);
        }

        // File input display click event
        const fileInputDisplay = document.querySelector('.file-input-display');
        if (fileInputDisplay && elements.imageInput) {
            fileInputDisplay.addEventListener('click', function() {
                elements.imageInput.click();
            });
        }

        // Form submit event
        if (elements.uploadForm) {
            elements.uploadForm.addEventListener('submit', handleFormSubmit);
        }

        // Handle page visibility changes (to reset button state if user navigates back)
        document.addEventListener('visibilitychange', function() {
            if (!document.hidden) {
                setButtonLoading(false);
            }
        });

        // Handle advanced parameters
        handleAdvancedParams();
    }

    /**
     * Initialize the application
     */
    function init() {
        // Wait for DOM to be fully loaded
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', init);
            return;
        }

        // Initialize elements and event listeners
        if (initElements()) {
            initEventListeners();
            console.log('Segmentation Studio JS initialized successfully');
        }
    }

    // Global function for removing preview (called from HTML)
    window.removePreview = function() {
        if (elements.imageInput) {
            elements.imageInput.value = '';
        }
        hideImagePreview();
    };

    // Start initialization
    init();

})();
