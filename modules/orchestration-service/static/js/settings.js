// Settings Page JavaScript - Based on working settings-simple.html functionality

// Configuration
const serverUrl = 'http://localhost:5000';
let currentSettings = {};

// Logging function
function log(message, type = 'info') {
    const timestamp = new Date().toLocaleTimeString();
    const logMessage = `[${timestamp}] ${message}`;
            console.log(logMessage);
    
    // Also add to debug panel if it exists
    const debugLog = document.getElementById('debugLog');
    if (debugLog) {
        debugLog.textContent += logMessage + '\n';
        debugLog.scrollTop = debugLog.scrollHeight;
    }
}

// Status message display
function showStatus(message, type = 'info', duration = 5000) {
    log(`Status: ${message}`);
    
    // Create status message element
    const statusDiv = document.createElement('div');
    statusDiv.className = `status-message ${type}`;
    statusDiv.textContent = message;
    statusDiv.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 12px 20px;
        border-radius: 6px;
        color: white;
        font-weight: 500;
        z-index: 10000;
        max-width: 400px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        animation: slideIn 0.3s ease-out;
    `;
    
    // Set background color based on type
    switch(type) {
        case 'success': statusDiv.style.backgroundColor = '#10b981'; break;
        case 'error': statusDiv.style.backgroundColor = '#ef4444'; break;
        case 'warning': statusDiv.style.backgroundColor = '#f59e0b'; break;
        default: statusDiv.style.backgroundColor = '#3b82f6'; break;
    }
    
    document.body.appendChild(statusDiv);
    
    // Auto-remove after duration
    setTimeout(() => {
        if (statusDiv.parentNode) {
            statusDiv.style.animation = 'slideOut 0.3s ease-in';
            setTimeout(() => {
                if (statusDiv.parentNode) {
                    statusDiv.parentNode.removeChild(statusDiv);
                }
            }, 300);
        }
    }, duration);
}

// Add CSS for animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
`;
document.head.appendChild(style);

// API call function with detailed logging
async function apiCall(endpoint, options = {}) {
    try {
        log(`API call: ${endpoint}`);
        const response = await fetch(`${serverUrl}${endpoint}`, {
            headers: { 'Content-Type': 'application/json' },
            ...options
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        log(`API success: ${endpoint}`);
        return data;
    } catch (error) {
        log(`API error: ${endpoint} - ${error.message}`, 'error');
        throw error;
    }
}

// Server status check
async function checkStatus() {
    try {
        const health = await apiCall('/health');
        updateServerStatusDisplay('online', health);
        showStatus('Server status checked successfully', 'success', 2000);
        } catch (error) {
        updateServerStatusDisplay('offline', null);
        showStatus(`Failed to check server status: ${error.message}`, 'error');
        }
    }

// Update server status display
function updateServerStatusDisplay(status, data) {
        const statusElement = document.getElementById('serverStatus');
        const deviceElement = document.getElementById('deviceType');
        const modelCountElement = document.getElementById('modelCount');
        const currentModelElement = document.getElementById('currentModel');
        const serverUptimeElement = document.getElementById('serverUptime');
        const lastInferenceElement = document.getElementById('lastInference');
        const modelsLoadedElement = document.getElementById('modelsLoaded');
        
        if (statusElement) {
        if (status === 'online') {
            statusElement.textContent = '✅ ONLINE';
            statusElement.className = 'status-badge success';
        } else {
            statusElement.textContent = '❌ OFFLINE';
            statusElement.className = 'status-badge error';
        }
        }
        
        if (data && !data.error) {
            if (deviceElement) deviceElement.textContent = data.device || 'Unknown';
            if (modelCountElement) modelCountElement.textContent = data.models_available || '0';
            if (currentModelElement) currentModelElement.textContent = data.current_model || 'Unknown';
            if (modelsLoadedElement) modelsLoadedElement.textContent = data.models_loaded || '0';
            
            // Format uptime
            if (data.server_uptime && serverUptimeElement) {
                const hours = Math.floor(data.server_uptime / 3600);
                const minutes = Math.floor((data.server_uptime % 3600) / 60);
                const seconds = Math.floor(data.server_uptime % 60);
                serverUptimeElement.textContent = `${hours}h ${minutes}m ${seconds}s`;
            }
            
            // Format last inference time
            if (data.last_inference_ago !== undefined && data.last_inference_ago >= 0 && lastInferenceElement) {
                if (data.last_inference_ago < 60) {
                    lastInferenceElement.textContent = `${Math.floor(data.last_inference_ago)}s ago`;
                } else if (data.last_inference_ago < 3600) {
                    lastInferenceElement.textContent = `${Math.floor(data.last_inference_ago / 60)}m ago`;
                } else {
                    lastInferenceElement.textContent = `${Math.floor(data.last_inference_ago / 3600)}h ago`;
                }
            } else if (lastInferenceElement) {
                lastInferenceElement.textContent = 'Never';
            }
        } else {
            // Set fallback values for error states
        if (deviceElement) deviceElement.textContent = 'Unknown';
        if (modelCountElement) modelCountElement.textContent = '0';
            if (currentModelElement) currentModelElement.textContent = 'Unknown';
            if (serverUptimeElement) serverUptimeElement.textContent = 'Unknown';
            if (lastInferenceElement) lastInferenceElement.textContent = 'Unknown';
            if (modelsLoadedElement) modelsLoadedElement.textContent = '0';
    }
}

// CORS test functionality
async function testCors() {
    try {
        log('Testing CORS functionality...');
        const corsStatusElement = document.getElementById('apiConnectionStatus');
        if (corsStatusElement) {
            corsStatusElement.textContent = '🔄 Testing CORS...';
        }
        
        // Test the CORS endpoint
        const response = await fetch(`${serverUrl}/cors-test`, {
            method: 'GET',
            headers: { 'Content-Type': 'application/json' }
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        // Check CORS headers
        const corsHeaders = {
            'Access-Control-Allow-Origin': response.headers.get('Access-Control-Allow-Origin'),
            'Access-Control-Allow-Methods': response.headers.get('Access-Control-Allow-Methods'),
            'Access-Control-Allow-Headers': response.headers.get('Access-Control-Allow-Headers')
        };
        
        log(`CORS Test Response: ${JSON.stringify(data)}`);
        log(`CORS Headers: ${JSON.stringify(corsHeaders)}`);
        
        if (corsHeaders['Access-Control-Allow-Origin']) {
            if (corsStatusElement) {
                corsStatusElement.textContent = '✅ CORS Working';
            }
            showStatus('CORS test successful - headers present', 'success');
        } else {
            if (corsStatusElement) {
                corsStatusElement.textContent = '❌ CORS Failed';
            }
            showStatus('CORS test failed - headers missing', 'error');
        }
        
    } catch (error) {
        const corsStatusElement = document.getElementById('apiConnectionStatus');
        if (corsStatusElement) {
            corsStatusElement.textContent = '❌ CORS Error';
        }
        log(`CORS test failed: ${error.message}`, 'error');
        showStatus(`CORS test failed: ${error.message}`, 'error');
    }
}

// Load available models
async function loadModels() {
    try {
        const corsStatusElement = document.getElementById('apiConnectionStatus');
        if (corsStatusElement) {
            corsStatusElement.textContent = 'Loading models...';
        }
        
        const models = await apiCall('/models');
        const modelSelect = document.getElementById('defaultModel');
        const currentValue = modelSelect ? modelSelect.value : '';
        
        if (modelSelect && models.models && models.models.length > 0) {
            // Clear and repopulate
        modelSelect.innerHTML = '';
            models.models.forEach(model => {
            const option = document.createElement('option');
            option.value = model;
            option.textContent = model;
            modelSelect.appendChild(option);
        });
        
            // Restore selection if possible
            if (models.models.includes(currentValue)) {
            modelSelect.value = currentValue;
            }
            
            if (corsStatusElement) {
                corsStatusElement.textContent = `✅ Connected (${models.models.length} models)`;
            }
            showStatus(`Loaded ${models.models.length} models`, 'success', 2000);
        } else {
            if (modelSelect) {
                modelSelect.innerHTML = '<option value="">No models found</option>';
            }
            if (corsStatusElement) {
                corsStatusElement.textContent = '❌ No models found';
            }
            showStatus('No models found', 'error');
        }
        } catch (error) {
        const corsStatusElement = document.getElementById('apiConnectionStatus');
        if (corsStatusElement) {
            corsStatusElement.textContent = `❌ Connection failed: ${error.message}`;
        }
        showStatus(`Failed to load models: ${error.message}`, 'error');
    }
}

// Load current configuration
async function loadCurrentConfig() {
    try {
        const config = await apiCall('/settings');
        currentSettings = config;
        
        // Display in JSON format
        const configElement = document.getElementById('currentConfig');
        if (configElement) {
            configElement.textContent = JSON.stringify(config, null, 2);
        }
        
        // Populate form fields
        populateFormFields(config);
        showStatus('Configuration loaded successfully', 'success', 2000);
        
    } catch (error) {
        showStatus(`Failed to load configuration: ${error.message}`, 'error');
    }
}

// Populate form fields with config data
function populateFormFields(config) {
    // Model settings
    setFieldValue('defaultModel', config.default_model);
    setFieldValue('devicePreference', config.device_preference);
    setFieldValue('minInferenceInterval', config.min_inference_interval);
    
    // Audio settings
    setFieldValue('bufferDuration', config.buffer_duration);
    setFieldValue('inferenceInterval', config.inference_interval);
    setFieldValue('sampleRate', config.sample_rate);
    
    // VAD settings
    setFieldValue('vadEnabled', config.vad_enabled, 'checkbox');
    setFieldValue('vadAggressiveness', config.vad_aggressiveness);
    
    // Diarization settings
    setFieldValue('diarizationEnabled', config.diarization_enabled, 'checkbox');
    setFieldValue('nSpeakers', config.n_speakers);
    setFieldValue('embeddingMethod', config.embedding_method);
    setFieldValue('clusteringMethod', config.clustering_method);
    setFieldValue('overlapDuration', config.overlap_duration);
    setFieldValue('speechEnhancementEnabled', config.speech_enhancement_enabled, 'checkbox');
    
    // Advanced settings
    setFieldValue('maxQueueSize', config.max_queue_size);
    setFieldValue('maxTranscriptionHistory', config.max_transcription_history);
    setFieldValue('logLevel', config.log_level);
    setFieldValue('enableFileLogging', config.enable_file_logging, 'checkbox');
    setFieldValue('openvinoDevice', config.openvino_device);
    setFieldValue('openvinoLogLevel', config.openvino_log_level);
}

// Set form field value
function setFieldValue(id, value, type = 'input') {
    const element = document.getElementById(id);
    if (element && value !== undefined && value !== null) {
        if (type === 'checkbox') {
            element.checked = Boolean(value);
        } else {
            element.value = value;
        }
    }
}

// Gather all settings from form
function gatherAllSettings() {
            return {
                // Model settings
        default_model: document.getElementById('defaultModel')?.value,
        device_preference: document.getElementById('devicePreference')?.value,
        min_inference_interval: parseInt(document.getElementById('minInferenceInterval')?.value) || 200,
                
                // Audio settings
        buffer_duration: parseFloat(document.getElementById('bufferDuration')?.value) || 6.0,
        inference_interval: parseFloat(document.getElementById('inferenceInterval')?.value) || 3.0,
        sample_rate: parseInt(document.getElementById('sampleRate')?.value) || 16000,
                
                // VAD settings
        vad_enabled: document.getElementById('vadEnabled')?.checked || false,
        vad_aggressiveness: parseInt(document.getElementById('vadAggressiveness')?.value) || 2,
                
                // Diarization settings
        diarization_enabled: document.getElementById('diarizationEnabled')?.checked || false,
        n_speakers: document.getElementById('nSpeakers')?.value ? parseInt(document.getElementById('nSpeakers').value) : null,
        embedding_method: document.getElementById('embeddingMethod')?.value || 'resemblyzer',
        clustering_method: document.getElementById('clusteringMethod')?.value || 'hdbscan',
        overlap_duration: parseFloat(document.getElementById('overlapDuration')?.value) || 2.0,
        speech_enhancement_enabled: document.getElementById('speechEnhancementEnabled')?.checked || true,
                
                // Advanced settings
        max_queue_size: parseInt(document.getElementById('maxQueueSize')?.value) || 10,
        max_transcription_history: parseInt(document.getElementById('maxTranscriptionHistory')?.value) || 200,
        log_level: document.getElementById('logLevel')?.value || 'INFO',
        enable_file_logging: document.getElementById('enableFileLogging')?.checked || true,
        openvino_device: document.getElementById('openvinoDevice')?.value || '',
        openvino_log_level: parseInt(document.getElementById('openvinoLogLevel')?.value) || 1
    };
}

// Save all settings
async function saveAllSettings() {
    try {
        const settings = gatherAllSettings();
        await apiCall('/settings', {
            method: 'POST',
            body: JSON.stringify(settings)
        });
        
        currentSettings = settings;
        const configElement = document.getElementById('currentConfig');
        if (configElement) {
            configElement.textContent = JSON.stringify(settings, null, 2);
        }
        
        showStatus('All settings saved successfully', 'success');
        
        // Refresh data after save
        setTimeout(() => {
            checkStatus();
            loadCurrentConfig();
        }, 1000);
        
    } catch (error) {
        showStatus(`Failed to save settings: ${error.message}`, 'error');
    }
}

// Update individual setting
async function updateSetting(key, value) {
    try {
        await apiCall(`/settings/${key}`, {
            method: 'POST',
            body: JSON.stringify({ value })
        });
        showStatus(`${key} updated successfully`, 'success', 2000);
        
        // For model changes, wait longer for loading
        if (key === 'default_model') {
            showStatus('Loading new model... This may take up to 60 seconds', 'info', 3000);
            
            // Wait for model to load, then refresh
            setTimeout(async () => {
                await checkStatus();
                await loadCurrentConfig();
                showStatus(`Model ${value} loaded successfully`, 'success');
            }, 5000);
        }
        
    } catch (error) {
        showStatus(`Failed to update ${key}: ${error.message}`, 'error');
    }
}

// Restart server with improved waiting
async function restartServer() {
        if (!confirm('Are you sure you want to restart the server? This will interrupt any ongoing transcriptions.')) {
            return;
        }
        
    try {
        showStatus('Initiating server restart...', 'info');
        const serverStatusElement = document.getElementById('serverStatus');
        if (serverStatusElement) {
            serverStatusElement.textContent = '🔄 Restarting...';
            serverStatusElement.className = 'status-badge warning';
        }
        
        await apiCall('/restart', { method: 'POST' });
        showStatus('Server restart initiated - waiting for server to come back online...', 'info');
        
        // Wait for restart with longer timeouts
        setTimeout(async () => {
            showStatus('Checking if server is back online...', 'info');
            
            // Try to check status multiple times with longer delays
            for (let attempt = 1; attempt <= 15; attempt++) {
                try {
                    await new Promise(resolve => setTimeout(resolve, 4000)); // Wait 4 seconds between attempts
                    const health = await apiCall('/health');
                    
                    if (health.status === 'healthy') {
                        updateServerStatusDisplay('online', health);
            
            // Update last restart time
            const lastRestartElement = document.getElementById('lastRestart');
            if (lastRestartElement) {
                lastRestartElement.textContent = new Date().toLocaleString();
            }
            
                        showStatus('Server restart completed successfully!', 'success');
                        
                        // Refresh all data after restart
                        setTimeout(async () => {
                            await loadModels();
                            await loadCurrentConfig();
                            await testCors();
                        }, 2000);
                        
                        break;
                    }
                } catch (error) {
                    if (attempt === 15) {
                        if (serverStatusElement) {
                            serverStatusElement.textContent = '❌ Restart failed';
                            serverStatusElement.className = 'status-badge error';
                        }
                        showStatus('Server restart may have failed - please check manually', 'error');
                    } else {
                        showStatus(`Restart attempt ${attempt}/15 - server not ready yet...`, 'info', 2000);
                    }
                }
            }
        }, 5000); // Initial wait before first check
        
        } catch (error) {
        const serverStatusElement = document.getElementById('serverStatus');
        if (serverStatusElement) {
            serverStatusElement.textContent = '❌ Restart failed';
            serverStatusElement.className = 'status-badge error';
        }
        showStatus(`Failed to restart server: ${error.message}`, 'error');
    }
}

// Stop server
async function stopServer() {
        if (!confirm('Are you sure you want to stop the server?')) {
            return;
        }
        
        try {
        await apiCall('/shutdown', { method: 'POST' });
        showStatus('Server shutdown initiated', 'warning');
        } catch (error) {
        showStatus(`Failed to stop server: ${error.message}`, 'error');
    }
}

// Export configuration
function exportConfig() {
    try {
        const config = gatherAllSettings();
            const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            
            const a = document.createElement('a');
            a.href = url;
            a.download = `whisper-npu-config-${new Date().toISOString().split('T')[0]}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            
            URL.revokeObjectURL(url);
        showStatus('Configuration exported', 'success');
        } catch (error) {
        showStatus(`Failed to export configuration: ${error.message}`, 'error');
    }
}

// Add debug panel
function addDebugPanel() {
    const debugPanel = document.createElement('div');
    debugPanel.innerHTML = `
        <div style="position: fixed; bottom: 10px; right: 10px; width: 400px; max-height: 300px; background: #f8f9fa; border: 1px solid #ddd; border-radius: 5px; padding: 10px; font-family: monospace; font-size: 12px; z-index: 9999; display: none;" id="debugPanel">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <strong>Debug Log</strong>
                <button onclick="document.getElementById('debugPanel').style.display='none'" style="margin-left: auto; padding: 2px 6px;">×</button>
            </div>
            <div id="debugLog" style="max-height: 250px; overflow-y: auto; background: white; padding: 5px; border: 1px solid #ccc; white-space: pre-wrap; font-size: 11px;"></div>
            <button onclick="document.getElementById('debugLog').textContent=''" style="margin-top: 5px; padding: 2px 6px;">Clear</button>
        </div>
    `;
    document.body.appendChild(debugPanel);
    
    // Add toggle button
    const toggleButton = document.createElement('button');
    toggleButton.textContent = '🐛 Debug';
    toggleButton.style.cssText = 'position: fixed; bottom: 10px; right: 10px; z-index: 10000; padding: 5px 10px; background: #007bff; color: white; border: none; border-radius: 3px; cursor: pointer;';
    toggleButton.onclick = () => {
        const panel = document.getElementById('debugPanel');
        panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
    };
    document.body.appendChild(toggleButton);
}

// Event binding
function bindEvents() {
    // Server control buttons
    const restartBtn = document.getElementById('restartServer');
    if (restartBtn) restartBtn.onclick = restartServer;
    
    const stopBtn = document.getElementById('stopServer');
    if (stopBtn) stopBtn.onclick = stopServer;
    
    const saveAllBtn = document.getElementById('saveAllSettings');
    if (saveAllBtn) saveAllBtn.onclick = saveAllSettings;
    
    const refreshBtn = document.getElementById('refreshModelSettings');
    if (refreshBtn) refreshBtn.onclick = loadModels;
    
    const exportBtn = document.getElementById('exportConfig');
    if (exportBtn) exportBtn.onclick = exportConfig;
    
    const corsBtn = document.getElementById('testCors');
    if (corsBtn) corsBtn.onclick = testCors;
    
    // Auto-save on change for key settings
    const autoSaveSettings = [
        { id: 'defaultModel', key: 'default_model' },
        { id: 'devicePreference', key: 'device_preference' },
        { id: 'bufferDuration', key: 'buffer_duration', type: 'number' },
        { id: 'inferenceInterval', key: 'inference_interval', type: 'number' },
        { id: 'vadEnabled', key: 'vad_enabled', type: 'checkbox' },
        { id: 'vadAggressiveness', key: 'vad_aggressiveness', type: 'number' },
        { id: 'logLevel', key: 'log_level' }
    ];
    
    autoSaveSettings.forEach(setting => {
        const element = document.getElementById(setting.id);
        if (element) {
            element.onchange = () => {
                let value = element.value;
                if (setting.type === 'checkbox') {
                    value = element.checked;
                } else if (setting.type === 'number') {
                    value = parseFloat(element.value);
                }
                updateSetting(setting.key, value);
            };
        }
    });
    }

    // Initialize when DOM is ready
function init() {
    log('Settings page initializing...');
    
    addDebugPanel();
    bindEvents();
    
    // Load initial data
    checkStatus();
    loadModels();
    loadCurrentConfig();
    
    // Auto-refresh server status every 10 seconds
    setInterval(checkStatus, 10000);
    
    log('Settings page initialized successfully');
}

// Initialize on DOM ready
    if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
    } else {
    init();
}

// Export functions for global access
window.checkStatus = checkStatus;
window.testCors = testCors;
window.loadModels = loadModels;
window.saveAllSettings = saveAllSettings;
window.restartServer = restartServer;
window.stopServer = stopServer;
window.exportConfig = exportConfig; 