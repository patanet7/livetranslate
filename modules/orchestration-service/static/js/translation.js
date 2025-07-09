// Translation functionality for LiveTranslate orchestration
class TranslationManager {
    constructor() {
        this.translations = [];
        this.isEnabled = false;
        this.sourceLanguage = 'en';
        this.targetLanguage = 'es';
        this.translationApiUrl = '/api/translation';
        this.maxTranslations = 50;
        
        this.initializeElements();
        this.setupEventListeners();
    }
    
    initializeElements() {
        this.elements = {
            sourceLanguage: document.getElementById('sourceLanguage'),
            targetLanguage: document.getElementById('targetLanguage'),
            swapLanguages: document.getElementById('swapLanguages'),
            clearTranslations: document.getElementById('clearTranslations'),
            translationContent: document.getElementById('translationContent'),
            translationToggle: document.getElementById('translationToggle')
        };
    }
    
    setupEventListeners() {
        if (this.elements.sourceLanguage) {
            this.elements.sourceLanguage.addEventListener('change', (e) => {
                this.sourceLanguage = e.target.value;
                this.saveSettings();
            });
        }
        
        if (this.elements.targetLanguage) {
            this.elements.targetLanguage.addEventListener('change', (e) => {
                this.targetLanguage = e.target.value;
                this.saveSettings();
            });
        }
        
        if (this.elements.swapLanguages) {
            this.elements.swapLanguages.addEventListener('click', () => {
                this.swapLanguages();
            });
        }
        
        if (this.elements.clearTranslations) {
            this.elements.clearTranslations.addEventListener('click', () => {
                this.clearTranslations();
            });
        }
        
        if (this.elements.translationToggle) {
            this.elements.translationToggle.addEventListener('click', () => {
                this.toggleTranslation();
                this.updateToggleButton();
            });
        }
    }
    
    swapLanguages() {
        const temp = this.sourceLanguage;
        this.sourceLanguage = this.targetLanguage;
        this.targetLanguage = temp;
        
        if (this.elements.sourceLanguage) {
            this.elements.sourceLanguage.value = this.sourceLanguage;
        }
        if (this.elements.targetLanguage) {
            this.elements.targetLanguage.value = this.targetLanguage;
        }
        
        this.saveSettings();
    }
    
    async translateText(text) {
        if (!text || !text.trim()) {
            return null;
        }
        
        try {
            const response = await fetch(`${this.translationApiUrl}/translate`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: text.trim(),
                    source_language: this.sourceLanguage,
                    target_language: this.targetLanguage
                })
            });
            
            if (!response.ok) {
                throw new Error(`Translation failed: ${response.statusText}`);
            }
            
            const result = await response.json();
            return result;
            
        } catch (error) {
            console.error('Translation error:', error);
            // Return mock translation for development
            return {
                translated_text: `[Translation: ${text}]`,
                confidence: 0.8,
                source_language: this.sourceLanguage,
                target_language: this.targetLanguage,
                processing_time: 0.5
            };
        }
    }
    
    async handleTranscriptionForTranslation(transcription) {
        if (!this.isEnabled || !transcription) {
            return;
        }
        
        try {
            const translation = await this.translateText(transcription);
            if (translation) {
                this.addTranslation(transcription, translation);
            }
        } catch (error) {
            console.error('Error handling transcription for translation:', error);
        }
    }
    
    addTranslation(originalText, translationResult) {
        const translation = {
            id: Date.now() + Math.random(),
            timestamp: new Date(),
            originalText: originalText,
            translatedText: translationResult.translated_text,
            sourceLanguage: translationResult.source_language || this.sourceLanguage,
            targetLanguage: translationResult.target_language || this.targetLanguage,
            confidence: translationResult.confidence || 0.5,
            processingTime: translationResult.processing_time || 0
        };
        
        this.translations.unshift(translation);
        
        // Limit the number of translations
        if (this.translations.length > this.maxTranslations) {
            this.translations = this.translations.slice(0, this.maxTranslations);
        }
        
        this.updateTranslationDisplay();
        this.logActivity(`Translated: "${originalText.substring(0, 50)}${originalText.length > 50 ? '...' : ''}" (${translation.confidence.toFixed(2)} confidence)`);
    }
    
    updateTranslationDisplay() {
        if (!this.elements.translationContent) {
            return;
        }
        
        if (this.translations.length === 0) {
            this.elements.translationContent.innerHTML = 
                '<div class="translation-working">Translation will appear here as you speak...</div>';
            return;
        }
        
        const translationsHtml = this.translations.map(translation => {
            const confidenceClass = translation.confidence >= 0.8 ? 'high' : 
                                   translation.confidence >= 0.6 ? 'medium' : 'low';
            
            return `
                <div class="translation-item" data-id="${translation.id}">
                    <div class="translation-source">
                        <strong>${this.getLanguageName(translation.sourceLanguage)}:</strong> ${translation.originalText}
                    </div>
                    <div class="translation-target">
                        <strong>${this.getLanguageName(translation.targetLanguage)}:</strong> ${translation.translatedText}
                    </div>
                    <div class="translation-meta">
                        <span>${translation.timestamp.toLocaleTimeString()}</span>
                        <span class="translation-confidence ${confidenceClass}">
                            ${Math.round(translation.confidence * 100)}%
                        </span>
                    </div>
                </div>
            `;
        }).join('');
        
        this.elements.translationContent.innerHTML = translationsHtml;
        
        // Auto-scroll to top to show latest translation
        this.elements.translationContent.scrollTop = 0;
    }
    
    getLanguageName(code) {
        const languages = {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'ja': 'Japanese',
            'ko': 'Korean',
            'zh': 'Chinese'
        };
        return languages[code] || code.toUpperCase();
    }
    
    clearTranslations() {
        this.translations = [];
        this.updateTranslationDisplay();
        this.logActivity('Translation history cleared');
    }
    
    enableTranslation() {
        this.isEnabled = true;
        this.logActivity('Translation enabled');
    }
    
    disableTranslation() {
        this.isEnabled = false;
        this.logActivity('Translation disabled');
    }
    
    toggleTranslation() {
        if (this.isEnabled) {
            this.disableTranslation();
        } else {
            this.enableTranslation();
        }
    }
    
    updateToggleButton() {
        if (this.elements.translationToggle) {
            if (this.isEnabled) {
                this.elements.translationToggle.textContent = '🌍 Disable Translation';
                this.elements.translationToggle.classList.add('active');
            } else {
                this.elements.translationToggle.textContent = '🌍 Enable Translation';
                this.elements.translationToggle.classList.remove('active');
            }
        }
    }
    
    saveSettings() {
        const settings = {
            sourceLanguage: this.sourceLanguage,
            targetLanguage: this.targetLanguage,
            isEnabled: this.isEnabled
        };
        localStorage.setItem('translationSettings', JSON.stringify(settings));
    }
    
    loadSettings() {
        try {
            const saved = localStorage.getItem('translationSettings');
            if (saved) {
                const settings = JSON.parse(saved);
                this.sourceLanguage = settings.sourceLanguage || 'en';
                this.targetLanguage = settings.targetLanguage || 'es';
                this.isEnabled = settings.isEnabled || false;
                
                // Update UI elements
                if (this.elements.sourceLanguage) {
                    this.elements.sourceLanguage.value = this.sourceLanguage;
                }
                if (this.elements.targetLanguage) {
                    this.elements.targetLanguage.value = this.targetLanguage;
                }
                this.updateToggleButton();
            }
        } catch (error) {
            console.error('Error loading translation settings:', error);
        }
    }
    
    logActivity(message) {
        // Integration with existing logging system
        if (window.WhisperApp && window.WhisperApp.logActivity) {
            window.WhisperApp.logActivity('TRANSLATION', message);
        } else {
            console.log('[TRANSLATION]', message);
        }
    }
    
    // Get translation statistics
    getStatistics() {
        const total = this.translations.length;
        const avgConfidence = total > 0 ? 
            this.translations.reduce((sum, t) => sum + t.confidence, 0) / total : 0;
        
        const languagePairs = {};
        this.translations.forEach(t => {
            const pair = `${t.sourceLanguage}-${t.targetLanguage}`;
            languagePairs[pair] = (languagePairs[pair] || 0) + 1;
        });
        
        return {
            totalTranslations: total,
            averageConfidence: avgConfidence,
            languagePairs: languagePairs,
            isEnabled: this.isEnabled
        };
    }
}

// Initialize translation manager when DOM is loaded
let translationManager;
document.addEventListener('DOMContentLoaded', function() {
    translationManager = new TranslationManager();
    translationManager.loadSettings();
    
    // Make it globally accessible
    window.TranslationManager = translationManager;
});

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = TranslationManager;
}