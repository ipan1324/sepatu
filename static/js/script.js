document.addEventListener('DOMContentLoaded', function() {
    
    // File Upload Preview & Drag-and-Drop Logic
    const fileInput = document.getElementById('fileInput');
    const uploadArea = document.getElementById('uploadArea');
    const uploadPlaceholder = document.getElementById('uploadPlaceholder');
    const imagePreviewContainer = document.getElementById('imagePreviewContainer');
    const imagePreview = document.getElementById('imagePreview');
    const fileName = document.getElementById('fileName');
    
    // Form Submit Logic
    const uploadForm = document.getElementById('uploadForm');
    const submitBtn = document.getElementById('submitBtn');
    const loadingState = document.getElementById('loadingState');

    if (fileInput && uploadArea) {
        // Highlight drop area when item is dragged over it
        ['dragenter', 'dragover'].forEach(eventName => {
            uploadArea.addEventListener(eventName, preventDefaults, false);
            uploadArea.addEventListener(eventName, highlight, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, preventDefaults, false);
            uploadArea.addEventListener(eventName, unhighlight, false);
        });

        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }

        function highlight(e) {
            uploadArea.classList.add('dragover');
        }

        function unhighlight(e) {
            uploadArea.classList.remove('dragover');
        }

        // Handle dropped files
        uploadArea.addEventListener('drop', function(e) {
            let dt = e.dataTransfer;
            let files = dt.files;
            
            if (files.length > 0) {
                fileInput.files = files; // Set the input files
                handleFiles(files[0]);
            }
        });

        // Handle file selection via click
        fileInput.addEventListener('change', function() {
            if (this.files.length > 0) {
                handleFiles(this.files[0]);
            }
        });

        function handleFiles(file) {
            // Validate file type
            if (!file.type.match('image.*')) {
                alert("Harap unggah file gambar yang valid (JPG, PNG).");
                fileInput.value = "";
                return;
            }

            // Show preview
            const reader = new FileReader();
            reader.onload = function(e) {
                imagePreview.src = e.target.result;
                fileName.textContent = file.name;
                
                uploadPlaceholder.classList.add('d-none');
                imagePreviewContainer.classList.remove('d-none');
            }
            reader.readAsDataURL(file);
        }
    }

    // Handle Form Submission Loading State
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            if (fileInput.files.length > 0) {
                // Hide form area elements
                uploadArea.classList.add('d-none');
                submitBtn.classList.add('d-none');
                
                // Show loading state
                loadingState.classList.remove('d-none');
            }
        });
    }

});
