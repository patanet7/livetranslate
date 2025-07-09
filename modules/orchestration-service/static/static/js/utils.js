/**
 * Utility functions
 * This module provides common utility functions used across the application
 */

// Utility functions
window.LiveTranslateUtils = {
    // Format timestamp for display
    formatTimestamp: function(timestamp) {
        if (!timestamp) return new Date().toLocaleTimeString();
        return new Date(timestamp).toLocaleTimeString();
    },
    
    // Debounce function for search and input handlers
    debounce: function(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },
    
    // Copy text to clipboard
    copyToClipboard: async function(text) {
        if (navigator.clipboard) {
            return await navigator.clipboard.writeText(text);
        } else {
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = text;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
        }
    },
    
    // Validate audio file
    isValidAudioFile: function(file) {
        const validTypes = ['audio/wav', 'audio/mp3', 'audio/m4a', 'audio/webm', 'audio/ogg'];
        return validTypes.includes(file.type);
    },
    
    // Format file size
    formatFileSize: function(bytes) {
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        if (bytes === 0) return '0 Bytes';
        const i = Math.floor(Math.log(bytes) / Math.log(1024));
        return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
    }
};

console.log('Utils module loaded');