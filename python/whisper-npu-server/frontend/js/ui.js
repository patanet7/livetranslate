// UI module for interface management
const UI = {
    updateServerStatus(status, data) {
        state.serverStatus = status;
        
        if (status === 'healthy' && data) {
            if (elements.statusDot) elements.statusDot.className = 'status-dot connected';
            if (elements.statusText) elements.statusText.textContent = 'Connected';
            if (elements.deviceType) elements.deviceType.textContent = data.device || 'Unknown';
            if (elements.modelCount) elements.modelCount.textContent = data.models_available || 0;
        } else {
            if (elements.statusDot) elements.statusDot.className = 'status-dot disconnected';
            if (elements.statusText) elements.statusText.textContent = 'Disconnected';
        }
    },

    updateModelsDisplay(data) {
        if (!elements.modelsContent) return;
        
        elements.modelsContent.innerHTML = '';
        
        if (data.note) {
            const noteDiv = document.createElement('div');
            noteDiv.className = 'log-entry WARNING';
            noteDiv.textContent = data.note;
            elements.modelsContent.appendChild(noteDiv);
        }
        
        if (state.availableModels.length === 0) {
            elements.modelsContent.innerHTML += '<div class="log-entry WARNING">No models available</div>';
            if (elements.selectedModel) {
                elements.selectedModel.innerHTML = '<option value="">No models available</option>';
            }
            return;
        }
        
        // Update model selector
        if (elements.selectedModel) {
            elements.selectedModel.innerHTML = '<option value="">Select a model</option>';
        }
        
        // Auto-select default model
        const defaultModel = 'whisper-base'; // Server's default model
        let modelToSelect = null;
        
        state.availableModels.forEach(model => {
            const modelDiv = document.createElement('div');
            modelDiv.className = 'model-item';
            modelDiv.innerHTML = `
                <span>${model}</span>
                <button class="button small" onclick="selectModel('${model}')">Select</button>
            `;
            elements.modelsContent.appendChild(modelDiv);
            
            // Add to selector
            if (elements.selectedModel) {
                const option = document.createElement('option');
                option.value = model;
                option.textContent = model;
                elements.selectedModel.appendChild(option);
            }
            
            // Remember default model or first available
            if (model === defaultModel) {
                modelToSelect = model;
            } else if (!modelToSelect) {
                modelToSelect = model; // Fallback to first model
            }
        });
        
        // Auto-select the chosen model
        if (modelToSelect && elements.selectedModel) {
            elements.selectedModel.value = modelToSelect;
            state.selectedModel = modelToSelect;
            this.updateModelSelection(modelToSelect);
            addLog('SUCCESS', `Auto-selected model: ${modelToSelect}`);
        }
    },

    updateModelSelection(modelName) {
        // Update visual selection in model list
        const modelItems = document.querySelectorAll('.model-item');
        modelItems.forEach(item => {
            item.classList.remove('selected');
            if (item.querySelector('span').textContent === modelName) {
                item.classList.add('selected');
            }
        });
        
        // Update test transcribe button state when model changes
        if (typeof updateAudioTestUI === 'function') {
            updateAudioTestUI();
        }
    },

    addTranscription(text, isFinal) {
        if (!elements.transcriptionContent) return;
        
        const transcriptDiv = document.createElement('div');
        transcriptDiv.className = isFinal ? 'transcript-final' : 'transcript-working';
        
        const timestamp = new Date().toLocaleTimeString();
        transcriptDiv.innerHTML = `<strong>[${timestamp}]</strong> ${text}`;
        
        elements.transcriptionContent.appendChild(transcriptDiv);
        elements.transcriptionContent.scrollTop = elements.transcriptionContent.scrollHeight;
        
        // Store transcript for export
        state.allTranscripts.push({
            timestamp: new Date().toISOString(),
            time: timestamp,
            text: text,
            isFinal: isFinal
        });
        
        // Limit number of transcriptions
        while (elements.transcriptionContent.children.length > config.maxTranscripts) {
            elements.transcriptionContent.removeChild(elements.transcriptionContent.firstChild);
        }
    },

    clearTranscripts() {
        if (elements.transcriptionContent) {
            elements.transcriptionContent.innerHTML = '<div class="transcript-working">Ready to transcribe... Select a model and start recording.</div>';
        }
        state.allTranscripts = [];
        addLog('INFO', 'Transcripts cleared');
    },

    exportTranscripts() {
        if (state.allTranscripts.length === 0) {
            addLog('WARNING', 'No transcripts to export');
            return;
        }
        
        const exportData = {
            exportDate: new Date().toISOString(),
            device: state.serverStatus === 'healthy' ? (elements.deviceType?.textContent || 'Unknown') : 'Unknown',
            totalTranscripts: state.allTranscripts.length,
            transcripts: state.allTranscripts
        };
        
        const jsonString = JSON.stringify(exportData, null, 2);
        const blob = new Blob([jsonString], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `whisper-transcripts-${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        addLog('SUCCESS', `Exported ${state.allTranscripts.length} transcripts`);
    },

    downloadLastAudio() {
        if (!state.lastAudioBlob) {
            addLog('WARNING', 'No audio recording to download');
            return;
        }
        
        const url = URL.createObjectURL(state.lastAudioBlob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `whisper-audio-${new Date().toISOString().split('T')[0]}-${Date.now()}.webm`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        addLog('SUCCESS', 'Audio recording downloaded');
    }
};

// File upload transcription
async function transcribeUploadedFile() {
    const fileInput = elements.audioFileInput;
    const selectedModel = elements.selectedModel?.value;
    
    if (!fileInput?.files || fileInput.files.length === 0) {
        addLog('WARNING', 'Please select an audio file first');
        return;
    }
    
    if (!selectedModel) {
        addLog('WARNING', 'Please select a model first');
        return;
    }
    
    const file = fileInput.files[0];
    
    try {
        await API.transcribeFile(file, selectedModel);
    } catch (error) {
        // Error already logged in API module
    }
}

// Global functions for backwards compatibility
window.updateServerStatus = UI.updateServerStatus.bind(UI);
window.updateModelsDisplay = UI.updateModelsDisplay.bind(UI);
window.updateModelSelection = UI.updateModelSelection.bind(UI);
window.addTranscription = UI.addTranscription.bind(UI);
window.clearTranscripts = UI.clearTranscripts.bind(UI);
window.exportTranscripts = UI.exportTranscripts.bind(UI);
window.downloadLastAudio = UI.downloadLastAudio.bind(UI);
window.transcribeUploadedFile = transcribeUploadedFile; 