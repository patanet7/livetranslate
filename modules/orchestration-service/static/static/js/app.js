/**
 * LiveTranslate Frontend Application
 * Main JavaScript application handling UI interactions, WebSocket connections, and service coordination
 */

class LiveTranslateApp {
    constructor() {
        this.socket = null;
        this.config = {};
        this.state = {
            currentTab: 'transcription',
            isRecording: false,
            isStreaming: false,
            translationEnabled: false,
            audioDevice: null,
            selectedModel: null,
            sessionId: null,
            services: {
                whisper: { status: 'unknown', data: null },
                speaker: { status: 'unknown', data: null },
                translation: { status: 'unknown', data: null },
                gateway: { status: 'unknown', data: null }
            }
        };
        
        this.init();
    }
    
    async init() {
        console.log('Initializing LiveTranslate Frontend...');
        
        try {
            // Initialize UI components
            this.initializeUI();
            
            // Load configuration
            await this.loadConfig();
            
            // Initialize WebSocket connection
            this.initializeWebSocket();
            
            // Load initial data
            await this.loadInitialData();
            
            console.log('LiveTranslate Frontend initialized successfully');
            this.showNotification('Frontend initialized successfully', 'success');
            
        } catch (error) {
            console.error('Failed to initialize frontend:', error);
            this.showNotification('Failed to initialize frontend', 'error');
        }
    }
    
    initializeUI() {
        // Tab navigation
        this.setupTabNavigation();
        
        // Audio device selection
        this.setupAudioDevices();
        
        // Service status updates
        this.setupServiceStatus();
        
        // Theme management
        this.setupThemeManager();
        
        // Event listeners
        this.setupEventListeners();
        
        console.log('UI components initialized');
    }
    
    setupTabNavigation() {
        const navItems = document.querySelectorAll('.nav-item[data-tab]');
        const tabContents = document.querySelectorAll('.tab-content');
        
        navItems.forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                const tabName = item.dataset.tab;
                this.switchTab(tabName);
            });
        });
        
        // Set initial active tab
        this.switchTab(this.state.currentTab);
    }
    
    switchTab(tabName) {
        // Remove active class from all nav items and tab contents
        document.querySelectorAll('.nav-item').forEach(item => {
            item.classList.remove('active');
        });
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        
        // Add active class to selected tab
        const navItem = document.querySelector(`[data-tab="${tabName}"]`);
        const tabContent = document.getElementById(`${tabName}-tab`);
        
        if (navItem && tabContent) {
            navItem.classList.add('active');
            tabContent.classList.add('active');
            this.state.currentTab = tabName;
            
            // Trigger tab-specific initialization
            this.onTabSwitch(tabName);
        }
    }
    
    onTabSwitch(tabName) {
        switch (tabName) {
            case 'transcription':
                this.initializeTranscriptionTab();
                break;
            case 'translation':
                this.initializeTranslationTab();
                break;
            case 'speaker':
                this.initializeSpeakerTab();
                break;
            case 'testing':
                this.initializeTestingTab();
                break;
            case 'settings':
                this.initializeSettingsTab();
                break;
        }
    }
    
    setupAudioDevices() {
        if (navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) {
            navigator.mediaDevices.enumerateDevices()
                .then(devices => {
                    const audioInputs = devices.filter(device => device.kind === 'audioinput');
                    this.populateAudioDeviceSelect(audioInputs);
                })
                .catch(error => {
                    console.error('Error enumerating audio devices:', error);
                    this.addLog('ERROR', 'Failed to enumerate audio devices');
                });
        }
    }
    
    populateAudioDeviceSelect(devices) {
        const deviceSelect = document.getElementById('audioDevice');
        if (!deviceSelect) return;
        
        // Clear existing options except the first one
        deviceSelect.innerHTML = '<option value="">Select device...</option>';
        
        devices.forEach(device => {
            const option = document.createElement('option');
            option.value = device.deviceId;
            option.textContent = device.label || `Audio Device ${devices.indexOf(device) + 1}`;
            deviceSelect.appendChild(option);
        });
        
        this.addLog('INFO', `Found ${devices.length} audio input devices`);
    }
    
    setupServiceStatus() {
        // Update status indicators
        this.updateStatusIndicator('connecting');
        
        // Check service health periodically
        setInterval(() => {
            this.checkAllServicesHealth();
        }, 30000); // Every 30 seconds
    }
    
    setupThemeManager() {
        const themeSelect = document.getElementById('themeSelect');
        if (themeSelect) {
            // Load saved theme
            const savedTheme = localStorage.getItem('livetranslate-theme') || 'light';
            themeSelect.value = savedTheme;
            this.applyTheme(savedTheme);
            
            // Handle theme changes
            themeSelect.addEventListener('change', (e) => {
                const theme = e.target.value;
                this.applyTheme(theme);
                localStorage.setItem('livetranslate-theme', theme);
            });
        }
    }
    
    applyTheme(theme) {
        if (theme === 'auto') {
            const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
            theme = prefersDark ? 'dark' : 'light';
        }
        
        document.documentElement.setAttribute('data-theme', theme);
    }
    
    setupEventListeners() {
        // Audio controls
        document.getElementById('recordButton')?.addEventListener('click', () => {
            this.toggleRecording();
        });
        
        document.getElementById('streamButton')?.addEventListener('click', () => {
            this.toggleStreaming();
        });
        
        document.getElementById('uploadButton')?.addEventListener('click', () => {
            document.getElementById('audioFileInput')?.click();
        });
        
        document.getElementById('audioFileInput')?.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.transcribeFile(e.target.files[0]);
            }
        });
        
        // Clear buttons
        document.getElementById('clearTranscripts')?.addEventListener('click', () => {
            this.clearTranscripts();
        });
        
        document.getElementById('clearLogs')?.addEventListener('click', () => {
            this.clearLogs();
        });
        
        // Export buttons
        document.getElementById('exportTranscripts')?.addEventListener('click', () => {
            this.exportTranscripts();
        });
        
        // Model selection
        document.getElementById('selectedModel')?.addEventListener('change', (e) => {
            this.state.selectedModel = e.target.value;
            this.addLog('INFO', `Model selected: ${e.target.value}`);
        });
        
        // Audio device selection
        document.getElementById('audioDevice')?.addEventListener('change', (e) => {
            this.state.audioDevice = e.target.value;
            this.addLog('INFO', `Audio device selected: ${e.target.selectedOptions[0]?.text || 'Default'}`);
        });
        
        // Settings save configuration
        document.getElementById('saveServiceConfig')?.addEventListener('click', () => {
            this.saveServiceConfiguration();
        });
        
        // Test all services button
        document.getElementById('testAllServices')?.addEventListener('click', () => {
            this.testAllServices();
        });
        
        // Copy and export buttons for translation
        document.getElementById('copyTranslation')?.addEventListener('click', () => {
            this.copyTranslationResults();
        });
        
        document.getElementById('exportTranslation')?.addEventListener('click', () => {
            this.exportTranslationResults();
        });
    }
    
    async loadConfig() {
        try {
            const response = await fetch('/api/config');
            this.config = await response.json();
            console.log('Configuration loaded:', this.config);
        } catch (error) {
            console.error('Failed to load configuration:', error);
            this.config = this.getDefaultConfig();
        }
    }
    
    getDefaultConfig() {
        return {
            services: {
                whisper: { url: 'http://localhost:5001', features: [] },
                speaker: { url: 'http://localhost:5002', features: [] },
                translation: { url: 'http://localhost:5003', features: [] },
                gateway: { url: 'http://localhost:5000', features: [] }
            },
            ui: {
                theme: 'light',
                language: 'en',
                audio_visualization: true,
                speaker_colors: ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            },
            features: {
                real_time_transcription: true,
                speaker_diarization: true,
                translation: true,
                audio_testing: true
            }
        };
    }
    
    initializeWebSocket() {
        try {
            // Connect to frontend service
            this.socket = io();
            
            // Connect to whisper service for real-time transcription
            this.whisperSocket = io('http://localhost:5001');
            
            this.socket.on('connect', () => {
                console.log('WebSocket connected');
                this.state.sessionId = this.socket.id;
                this.updateStatusIndicator('connected');
                this.addLog('SUCCESS', 'Connected to frontend service');
            });
            
            this.socket.on('disconnect', () => {
                console.log('WebSocket disconnected');
                this.updateStatusIndicator('disconnected');
                this.addLog('WARNING', 'Disconnected from frontend service');
            });
            
            this.socket.on('connected', (data) => {
                this.addLog('INFO', `Session established: ${data.session_id}`);
            });
            
            // Subscribe to service updates
            this.socket.on('transcription_update', (data) => {
                this.handleTranscriptionUpdate(data);
            });
            
            this.socket.on('speaker_update', (data) => {
                this.handleSpeakerUpdate(data);
            });
            
            this.socket.on('translation_update', (data) => {
                this.handleTranslationUpdate(data);
            });
            
            // Whisper service WebSocket handlers
            this.whisperSocket.on('connect', () => {
                console.log('Connected to Whisper service');
                this.addLog('SUCCESS', 'Connected to Whisper service');
            });
            
            this.whisperSocket.on('disconnect', () => {
                console.log('Disconnected from Whisper service');
                this.addLog('WARNING', 'Disconnected from Whisper service');
            });
            
            this.whisperSocket.on('transcription_result', (data) => {
                this.handleTranscriptionResult(data);
            });
            
            this.whisperSocket.on('error', (error) => {
                console.error('Whisper service error:', error);
                this.addLog('ERROR', `Whisper service error: ${error.message || error}`);
                this.showNotification('Whisper service error', 'error');
            });
            
        } catch (error) {
            console.error('Failed to initialize WebSocket:', error);
            this.updateStatusIndicator('error');
        }
    }
    
    async loadInitialData() {
        // Check service health
        await this.checkAllServicesHealth();
        
        // Load available models
        await this.loadModels();
        
        // Initialize audio visualization
        this.initializeAudioVisualization();
    }
    
    async checkAllServicesHealth() {
        try {
            const response = await fetch('/api/services/health');
            const healthData = await response.json();
            
            Object.keys(healthData.services).forEach(serviceName => {
                const serviceHealth = healthData.services[serviceName];
                this.state.services[serviceName] = serviceHealth;
                this.updateServiceStatus(serviceName, serviceHealth);
            });
            
            // Update overall status
            const allHealthy = Object.values(healthData.services).every(s => s.status === 'healthy');
            this.updateStatusIndicator(allHealthy ? 'connected' : 'warning');
            
        } catch (error) {
            console.error('Failed to check service health:', error);
            this.updateStatusIndicator('error');
        }
    }
    
    async loadModels() {
        try {
            const response = await fetch('/api/whisper/models');
            const models = await response.json();
            
            if (models && models.models) {
                this.populateModelSelect(models.models);
                this.addLog('INFO', `Loaded ${models.models.length} models`);
                
                // Update model count display
                const modelCountElement = document.getElementById('modelCount');
                if (modelCountElement) {
                    modelCountElement.textContent = `${models.models.length} models`;
                }
            }
        } catch (error) {
            console.error('Failed to load models:', error);
            this.addLog('ERROR', 'Failed to load models');
        }
    }
    
    populateModelSelect(models) {
        const modelSelect = document.getElementById('selectedModel');
        if (!modelSelect) return;
        
        modelSelect.innerHTML = '<option value="">Select model...</option>';
        
        models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.name || model;
            option.textContent = model.display_name || model.name || model;
            modelSelect.appendChild(option);
        });
    }
    
    updateStatusIndicator(status) {
        const statusDot = document.getElementById('statusDot');
        const statusText = document.getElementById('statusText');
        
        if (statusDot) {
            statusDot.className = `status-dot ${status}`;
        }
        
        if (statusText) {
            const statusMessages = {
                connected: 'Connected',
                connecting: 'Connecting...',
                disconnected: 'Disconnected',
                warning: 'Partial Connection',
                error: 'Connection Error'
            };
            statusText.textContent = statusMessages[status] || 'Unknown';
        }
    }
    
    updateServiceStatus(serviceName, healthData) {
        const statusElement = document.getElementById(`${serviceName}Status`) || 
                             document.getElementById(`${serviceName}ServiceStatus`);
        
        if (statusElement) {
            const icon = healthData.status === 'healthy' ? 'fa-check-circle' : 'fa-times-circle';
            const color = healthData.status === 'healthy' ? 'var(--success-color)' : 'var(--error-color)';
            
            statusElement.innerHTML = `
                <i class="fas ${icon}" style="color: ${color}"></i>
                ${healthData.status}
            `;
        }
    }
    
    // Tab-specific initialization methods
    initializeTranscriptionTab() {
        // Tab already handles most initialization
        this.addLog('INFO', 'Transcription tab activated');
    }
    
    initializeTranslationTab() {
        this.addLog('INFO', 'Translation tab activated');
        this.setupTranslationControls();
        this.loadSupportedLanguages();
    }
    
    initializeSpeakerTab() {
        this.addLog('INFO', 'Speaker diarization tab activated');
        // Initialize speaker settings if needed
    }
    
    initializeTestingTab() {
        this.addLog('INFO', 'Testing tab activated');
        this.setupTestingControls();
    }
    
    initializeSettingsTab() {
        this.addLog('INFO', 'Settings tab activated');
        this.loadSettingsValues();
    }
    
    setupTestingControls() {
        // Service test buttons
        document.querySelectorAll('.test-service-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const serviceName = btn.dataset.service;
                this.testService(serviceName);
            });
        });
        
        // Audio test buttons
        document.getElementById('testAudioButton')?.addEventListener('click', () => {
            this.testAudio();
        });
        
        document.getElementById('testAllServices')?.addEventListener('click', () => {
            this.testAllServices();
        });
    }

    setupTranslationControls() {
        // Translation enable/disable button
        const enableTranslationBtn = document.getElementById('enableTranslation');
        if (enableTranslationBtn) {
            enableTranslationBtn.addEventListener('click', () => {
                this.toggleTranslation();
            });
        }

        // File translation button
        const translateFileBtn = document.getElementById('translateFile');
        if (translateFileBtn) {
            translateFileBtn.addEventListener('click', () => {
                this.openFileTranslation();
            });
        }

        // Add text area for manual translation testing
        this.createTranslationTestArea();

        // Language change handlers for bidirectional setup
        document.getElementById('languageA')?.addEventListener('change', () => {
            this.onLanguageSettingsChange();
        });
        
        document.getElementById('languageB')?.addEventListener('change', () => {
            this.onLanguageSettingsChange();
        });

        document.getElementById('translationMode')?.addEventListener('change', () => {
            this.onTranslationModeChange();
        });
    }
    
    loadSettingsValues() {
        // Load current service URLs into settings
        Object.keys(this.config.services).forEach(serviceName => {
            const urlInput = document.getElementById(`${serviceName}Url`);
            if (urlInput) {
                urlInput.value = this.config.services[serviceName].url;
            }
        });
        
        // Load other settings
        const autoSave = document.getElementById('autoSave');
        if (autoSave) {
            autoSave.checked = localStorage.getItem('livetranslate-autosave') === 'true';
        }
        
        const showTimestamps = document.getElementById('showTimestamps');
        if (showTimestamps) {
            showTimestamps.checked = localStorage.getItem('livetranslate-timestamps') === 'true';
        }
    }
    
    // Audio and recording methods
    async toggleRecording() {
        if (this.state.isRecording) {
            this.stopRecording();
        } else {
            await this.startRecording();
        }
    }
    
    async startRecording() {
        try {
            this.showLoading('Starting recording...');
            
            const constraints = {
                audio: {
                    deviceId: this.state.audioDevice || undefined,
                    sampleRate: parseInt(document.getElementById('sampleRate')?.value || '16000'),
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true
                }
            };
            
            this.audioStream = await navigator.mediaDevices.getUserMedia(constraints);
            
            // Setup audio recording for streaming
            this.audioRecorder = new MediaRecorder(this.audioStream, {
                mimeType: 'audio/webm;codecs=opus'
            });
            
            this.audioRecorder.ondataavailable = (event) => {
                if (event.data && event.data.size > 0) {
                    this.processAudioChunk(event.data);
                }
            };
            
            this.audioRecorder.onerror = (error) => {
                console.error('MediaRecorder error:', error);
                this.addLog('ERROR', `Recording error: ${error.message}`);
            };
            
            // Setup real-time audio level monitoring
            this.setupRealTimeAudioMonitoring();
            
            // Start recording with small chunks for real-time processing
            this.audioRecorder.start(1000); // 1 second chunks
            
            this.state.isRecording = true;
            this.updateRecordingUI();
            this.addLog('SUCCESS', 'Recording started');
            
            this.hideLoading();
            
        } catch (error) {
            console.error('Failed to start recording:', error);
            this.addLog('ERROR', `Recording failed: ${error.message}`);
            this.hideLoading();
            this.showNotification('Failed to start recording', 'error');
        }
    }
    
    stopRecording() {
        this.state.isRecording = false;
        
        // Stop audio level monitoring
        this.stopAudioLevelMonitoring();
        
        // Stop media recorder
        if (this.audioRecorder && this.audioRecorder.state !== 'inactive') {
            this.audioRecorder.stop();
        }
        
        // Stop audio stream
        if (this.audioStream) {
            this.audioStream.getTracks().forEach(track => track.stop());
            this.audioStream = null;
        }
        
        this.updateRecordingUI();
        this.addLog('INFO', 'Recording stopped');
    }
    
    toggleStreaming() {
        if (this.state.isStreaming) {
            this.stopStreaming();
        } else {
            this.startStreaming();
        }
    }
    
    startStreaming() {
        this.state.isStreaming = true;
        this.updateStreamingUI();
        this.addLog('SUCCESS', 'Streaming started');
    }
    
    stopStreaming() {
        this.state.isStreaming = false;
        this.updateStreamingUI();
        this.addLog('INFO', 'Streaming stopped');
    }
    
    updateRecordingUI() {
        const recordButton = document.getElementById('recordButton');
        if (recordButton) {
            if (this.state.isRecording) {
                recordButton.innerHTML = '<i class="fas fa-stop"></i> Stop Recording';
                recordButton.classList.add('btn-danger');
                recordButton.classList.remove('btn-primary');
            } else {
                recordButton.innerHTML = '<i class="fas fa-microphone"></i> Start Recording';
                recordButton.classList.add('btn-primary');
                recordButton.classList.remove('btn-danger');
            }
        }
    }
    
    updateStreamingUI() {
        const streamButton = document.getElementById('streamButton');
        if (streamButton) {
            if (this.state.isStreaming) {
                streamButton.innerHTML = '<i class="fas fa-stop"></i> Stop Stream';
                streamButton.classList.add('btn-danger');
                streamButton.classList.remove('btn-secondary');
            } else {
                streamButton.innerHTML = '<i class="fas fa-stream"></i> Stream';
                streamButton.classList.add('btn-secondary');
                streamButton.classList.remove('btn-danger');
            }
        }
    }
    
    // File transcription
    async transcribeFile(file) {
        try {
            this.showLoading('Transcribing file...');
            
            const formData = new FormData();
            formData.append('audio', file);
            formData.append('model', this.state.selectedModel || 'whisper-base');
            
            const response = await fetch('/api/whisper/transcribe', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (result.text) {
                this.addTranscription(result.text, 'File Upload');
                this.addLog('SUCCESS', 'File transcribed successfully');
            }
            
            this.hideLoading();
            
        } catch (error) {
            console.error('File transcription failed:', error);
            this.addLog('ERROR', `File transcription failed: ${error.message}`);
            this.hideLoading();
            this.showNotification('File transcription failed', 'error');
        }
    }
    
    // UI utility methods
    addTranscription(text, speaker = 'Unknown', timestamp = null) {
        const transcriptionContent = document.getElementById('transcriptionContent');
        if (!transcriptionContent) return;
        
        // Remove placeholder if it exists
        const placeholder = transcriptionContent.querySelector('.placeholder');
        if (placeholder) {
            placeholder.remove();
        }
        
        const transcriptionItem = document.createElement('div');
        transcriptionItem.className = 'transcription-item';
        
        const timestampStr = timestamp || new Date().toLocaleTimeString();
        
        transcriptionItem.innerHTML = `
            <div class="transcription-timestamp">${timestampStr}</div>
            <div class="transcription-speaker">${speaker}</div>
            <div class="transcription-text">${text}</div>
        `;
        
        transcriptionContent.appendChild(transcriptionItem);
        transcriptionContent.scrollTop = transcriptionContent.scrollHeight;
    }
    
    clearTranscripts() {
        const transcriptionContent = document.getElementById('transcriptionContent');
        if (transcriptionContent) {
            transcriptionContent.innerHTML = `
                <div class="placeholder">
                    <i class="fas fa-microphone-slash"></i>
                    <p>Select a model and start recording to see transcriptions here</p>
                </div>
            `;
        }
        this.addLog('INFO', 'Transcripts cleared');
    }
    
    addLog(level, message) {
        const logsContent = document.getElementById('logsContent');
        if (!logsContent) return;
        
        const logEntry = document.createElement('div');
        logEntry.className = `log-entry ${level}`;
        
        const timestamp = new Date().toLocaleTimeString();
        logEntry.textContent = `[${timestamp}] ${level}: ${message}`;
        
        logsContent.appendChild(logEntry);
        logsContent.scrollTop = logsContent.scrollHeight;
        
        // Keep only last 100 log entries
        const logEntries = logsContent.querySelectorAll('.log-entry');
        if (logEntries.length > 100) {
            logEntries[0].remove();
        }
    }
    
    clearLogs() {
        const logsContent = document.getElementById('logsContent');
        if (logsContent) {
            logsContent.innerHTML = '';
        }
    }
    
    exportTranscripts() {
        const transcriptionItems = document.querySelectorAll('.transcription-item');
        if (transcriptionItems.length === 0) {
            this.showNotification('No transcripts to export', 'warning');
            return;
        }
        
        let exportText = '';
        transcriptionItems.forEach(item => {
            const timestamp = item.querySelector('.transcription-timestamp')?.textContent || '';
            const speaker = item.querySelector('.transcription-speaker')?.textContent || '';
            const text = item.querySelector('.transcription-text')?.textContent || '';
            
            exportText += `[${timestamp}] ${speaker}: ${text}\n\n`;
        });
        
        this.downloadTextFile(exportText, 'livetranslate-transcripts.txt');
        this.addLog('INFO', 'Transcripts exported');
    }
    
    downloadTextFile(text, filename) {
        const blob = new Blob([text], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        
        URL.revokeObjectURL(url);
    }
    
    showNotification(message, type = 'info') {
        const container = document.getElementById('notificationContainer');
        if (!container) return;
        
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        
        container.appendChild(notification);
        
        // Remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 5000);
    }
    
    showLoading(text = 'Loading...') {
        const overlay = document.getElementById('loadingOverlay');
        const loadingText = document.getElementById('loadingText');
        
        if (overlay) {
            overlay.style.display = 'flex';
        }
        if (loadingText) {
            loadingText.textContent = text;
        }
    }
    
    hideLoading() {
        const overlay = document.getElementById('loadingOverlay');
        if (overlay) {
            overlay.style.display = 'none';
        }
    }
    
    // Service testing methods
    async testService(serviceName) {
        try {
            this.addLog('INFO', `Testing ${serviceName} service...`);
            
            const response = await fetch(`/api/services/${serviceName}/status`);
            const result = await response.json();
            
            this.updateServiceStatus(serviceName, result);
            
            if (result.status === 'healthy') {
                this.addLog('SUCCESS', `${serviceName} service is healthy`);
                this.showNotification(`${serviceName} service is healthy`, 'success');
            } else {
                this.addLog('ERROR', `${serviceName} service failed: ${result.message}`);
                this.showNotification(`${serviceName} service failed`, 'error');
            }
            
        } catch (error) {
            this.addLog('ERROR', `Failed to test ${serviceName}: ${error.message}`);
            this.showNotification(`Failed to test ${serviceName}`, 'error');
        }
    }
    
    async testAllServices() {
        this.addLog('INFO', 'Testing all services...');
        const services = Object.keys(this.config.services);
        
        for (const serviceName of services) {
            await this.testService(serviceName);
        }
        
        this.addLog('INFO', 'Service testing completed');
    }
    
    async testAudio() {
        try {
            this.addLog('INFO', 'Testing audio recording...');
            
            const constraints = {
                audio: {
                    deviceId: this.state.audioDevice || undefined,
                    sampleRate: 16000,
                    channelCount: 1
                }
            };
            
            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            
            // Test recording for 3 seconds
            const mediaRecorder = new MediaRecorder(stream);
            const audioChunks = [];
            
            mediaRecorder.ondataavailable = (event) => {
                audioChunks.push(event.data);
            };
            
            mediaRecorder.onstop = () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                this.addLog('SUCCESS', `Audio test completed: ${audioBlob.size} bytes recorded`);
                
                // Enable play test button
                const playButton = document.getElementById('playTestButton');
                const transcribeButton = document.getElementById('transcribeTestButton');
                if (playButton) {
                    playButton.disabled = false;
                    playButton.onclick = () => this.playTestAudio(audioBlob);
                }
                if (transcribeButton) {
                    transcribeButton.disabled = false;
                    transcribeButton.onclick = () => this.transcribeTestAudio(audioBlob);
                }
                
                stream.getTracks().forEach(track => track.stop());
            };
            
            mediaRecorder.start();
            
            setTimeout(() => {
                mediaRecorder.stop();
            }, 3000);
            
            this.showNotification('Recording 3-second audio test...', 'info');
            
        } catch (error) {
            this.addLog('ERROR', `Audio test failed: ${error.message}`);
            this.showNotification('Audio test failed', 'error');
        }
    }
    
    playTestAudio(audioBlob) {
        const audio = new Audio(URL.createObjectURL(audioBlob));
        audio.play();
        this.addLog('INFO', 'Playing test audio');
    }
    
    async transcribeTestAudio(audioBlob) {
        try {
            const formData = new FormData();
            formData.append('audio', audioBlob, 'test-audio.wav');
            formData.append('model', this.state.selectedModel || 'whisper-base');
            
            const response = await fetch('/api/whisper/transcribe', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (result.text) {
                this.addLog('SUCCESS', `Test transcription: "${result.text}"`);
                this.showNotification('Test transcription completed', 'success');
                
                // Show result in testing tab
                const testOutput = document.getElementById('testOutput');
                if (testOutput) {
                    testOutput.textContent = JSON.stringify(result, null, 2);
                    document.getElementById('testResults').style.display = 'block';
                }
            }
            
        } catch (error) {
            this.addLog('ERROR', `Test transcription failed: ${error.message}`);
            this.showNotification('Test transcription failed', 'error');
        }
    }
    
    // Audio visualization and level monitoring
    setupRealTimeAudioMonitoring() {
        try {
            // Create audio context and analyser
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            this.analyser = this.audioContext.createAnalyser();
            this.microphone = this.audioContext.createMediaStreamSource(this.audioStream);
            
            // Configure analyser
            this.analyser.fftSize = 256;
            this.analyser.smoothingTimeConstant = 0.3;
            this.dataArray = new Uint8Array(this.analyser.frequencyBinCount);
            
            // Connect microphone to analyser
            this.microphone.connect(this.analyser);
            
            // Start monitoring
            this.isMonitoring = true;
            this.monitorAudioLevel();
            
            this.addLog('SUCCESS', 'Audio level monitoring started');
            
        } catch (error) {
            console.error('Failed to setup audio monitoring:', error);
            this.addLog('ERROR', `Audio monitoring failed: ${error.message}`);
        }
    }
    
    monitorAudioLevel() {
        if (!this.isMonitoring || !this.analyser) {
            return;
        }
        
        // Get frequency data
        this.analyser.getByteFrequencyData(this.dataArray);
        
        // Calculate average volume level
        let sum = 0;
        for (let i = 0; i < this.dataArray.length; i++) {
            sum += this.dataArray[i];
        }
        const averageLevel = sum / this.dataArray.length;
        const normalizedLevel = (averageLevel / 255) * 100;
        
        // Update UI elements
        this.updateAudioLevelDisplay(normalizedLevel);
        
        // Continue monitoring
        requestAnimationFrame(() => this.monitorAudioLevel());
    }
    
    updateAudioLevelDisplay(level) {
        const audioLevelBar = document.getElementById('audioLevelBar');
        const audioLevelText = document.getElementById('audioLevelText');
        
        if (audioLevelBar) {
            audioLevelBar.style.setProperty('--level', `${level}%`);
            
            // Add visual feedback based on level
            audioLevelBar.className = 'audio-level-bar';
            if (level > 70) {
                audioLevelBar.classList.add('high-level');
            } else if (level > 30) {
                audioLevelBar.classList.add('medium-level');
            } else {
                audioLevelBar.classList.add('low-level');
            }
        }
        
        if (audioLevelText) {
            audioLevelText.textContent = `${Math.round(level)}%`;
        }
        
        // Store current level for other components
        this.currentAudioLevel = level;
    }
    
    stopAudioLevelMonitoring() {
        this.isMonitoring = false;
        
        // Disconnect audio nodes
        if (this.microphone) {
            this.microphone.disconnect();
            this.microphone = null;
        }
        
        if (this.analyser) {
            this.analyser.disconnect();
            this.analyser = null;
        }
        
        // Close audio context
        if (this.audioContext && this.audioContext.state !== 'closed') {
            this.audioContext.close();
            this.audioContext = null;
        }
        
        // Reset UI
        this.updateAudioLevelDisplay(0);
        
        this.addLog('INFO', 'Audio level monitoring stopped');
    }
    
    // Audio visualization
    initializeAudioVisualization() {
        // Basic audio level monitoring setup
        if (navigator.mediaDevices) {
            this.addLog('INFO', 'Audio visualization ready');
        }
    }
    
    // Audio processing methods
    async processAudioChunk(audioBlob) {
        if (!this.whisperSocket || !this.whisperSocket.connected) {
            console.warn('Whisper service not connected - skipping audio chunk');
            return;
        }
        
        try {
            // Convert blob to ArrayBuffer
            const arrayBuffer = await audioBlob.arrayBuffer();
            
            // Convert to base64 for transmission
            const base64Audio = this.arrayBufferToBase64(arrayBuffer);
            
            // Send to whisper service
            const streamData = {
                audio_data: base64Audio,
                model_name: this.state.selectedModel || 'whisper-base',
                session_id: this.state.sessionId,
                sample_rate: 16000,
                enable_vad: true,
                streaming: true
            };
            
            this.whisperSocket.emit('transcribe_stream', streamData);
            
        } catch (error) {
            console.error('Failed to process audio chunk:', error);
            this.addLog('ERROR', `Audio processing failed: ${error.message}`);
        }
    }
    
    arrayBufferToBase64(buffer) {
        let binary = '';
        const bytes = new Uint8Array(buffer);
        const len = bytes.byteLength;
        for (let i = 0; i < len; i++) {
            binary += String.fromCharCode(bytes[i]);
        }
        return btoa(binary);
    }
    
    handleTranscriptionResult(data) {
        if (data.text && data.text.trim()) {
            this.addTranscription(data.text, data.speaker || 'Unknown', data.timestamp);
            this.addLog('SUCCESS', `Transcription: "${data.text}"`);
            
            // If translation is enabled, translate the text
            if (this.state.translationEnabled) {
                this.translateTranscription(data.text);
            }
        }
    }
    
    async translateTranscription(text) {
        try {
            const languageA = document.getElementById('languageA')?.value || 'zh';
            const languageB = document.getElementById('languageB')?.value || 'en';
            
            const { sourceLanguage, targetLanguage } = await this.detectTranslationDirection(text, languageA, languageB);
            await this.translateText(text, sourceLanguage, targetLanguage);
            
        } catch (error) {
            console.error('Auto-translation failed:', error);
            this.addLog('ERROR', `Auto-translation failed: ${error.message}`);
        }
    }

    // WebSocket event handlers
    handleTranscriptionUpdate(data) {
        if (data.text) {
            this.addTranscription(data.text, data.speaker || 'Unknown', data.timestamp);
        }
    }
    
    handleSpeakerUpdate(data) {
        // Update speaker diarization display
        console.log('Speaker update:', data);
    }
    
    // Translation functionality
    async loadSupportedLanguages() {
        try {
            const response = await fetch('/api/translation/languages');
            if (response.ok) {
                const languages = await response.json();
                this.populateLanguageSelects(languages);
            }
        } catch (error) {
            console.warn('Could not load supported languages:', error);
            // Use default languages if service is not available
        }
    }

    populateLanguageSelects(languages) {
        const languageASelect = document.getElementById('languageA');
        const languageBSelect = document.getElementById('languageB');
        
        if (languages && languages.length > 0) {
            // Store current selections
            const currentA = languageASelect?.value;
            const currentB = languageBSelect?.value;
            
            // Clear existing options
            if (languageASelect) {
                languageASelect.innerHTML = '';
            }
            if (languageBSelect) {
                languageBSelect.innerHTML = '';
            }

            // Add language options
            languages.forEach(lang => {
                if (lang.code !== 'auto') { // Skip auto-detect for bidirectional setup
                    if (languageASelect) {
                        const option = document.createElement('option');
                        option.value = lang.code;
                        option.textContent = lang.name;
                        if (lang.code === currentA) option.selected = true;
                        languageASelect.appendChild(option);
                    }
                    
                    if (languageBSelect) {
                        const option = document.createElement('option');
                        option.value = lang.code;
                        option.textContent = lang.name;
                        if (lang.code === currentB) option.selected = true;
                        languageBSelect.appendChild(option);
                    }
                }
            });
        }
    }

    createTranslationTestArea() {
        const translationInput = document.querySelector('.translation-input .panel-content');
        if (translationInput) {
            // Check if test area already exists
            if (translationInput.querySelector('.translation-test-area')) return;

            const testArea = document.createElement('div');
            testArea.className = 'translation-test-area';
            testArea.innerHTML = `
                <div class="test-input-group">
                    <label>Test Translation:</label>
                    <textarea id="manualTranslationInput" 
                              placeholder="Type or paste text here to test translation..." 
                              rows="3" 
                              style="width: 100%; margin: 8px 0; padding: 8px; border: 1px solid #ddd; border-radius: 4px;"></textarea>
                    <button id="translateTextBtn" class="btn btn-sm btn-primary" style="margin-top: 8px;">
                        <i class="fas fa-language"></i> Translate
                    </button>
                </div>
            `;
            
            translationInput.insertBefore(testArea, translationInput.firstChild);

            // Add event listener for translate button
            document.getElementById('translateTextBtn')?.addEventListener('click', () => {
                this.translateManualText();
            });

            // Add event listener for Enter key in textarea
            document.getElementById('manualTranslationInput')?.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && e.ctrlKey) {
                    this.translateManualText();
                }
            });
        }
    }

    async translateManualText() {
        const inputText = document.getElementById('manualTranslationInput')?.value.trim();
        if (!inputText) {
            this.showNotification('Please enter text to translate', 'warning');
            return;
        }

        const languageA = document.getElementById('languageA')?.value || 'zh';
        const languageB = document.getElementById('languageB')?.value || 'en';

        // Auto-detect source and target for bidirectional translation
        const { sourceLanguage, targetLanguage } = await this.detectTranslationDirection(inputText, languageA, languageB);
        
        await this.translateText(inputText, sourceLanguage, targetLanguage);
    }

    async detectTranslationDirection(text, languageA, languageB) {
        // Simple language detection based on character patterns
        // For a more robust solution, you could call the translation service's language detection API
        
        // Chinese character detection (simplified heuristic)
        const chinesePattern = /[\u4e00-\u9fff]/;
        const hasChineseChars = chinesePattern.test(text);
        
        // Simple detection logic
        if (languageA === 'zh' && languageB === 'en') {
            if (hasChineseChars) {
                return { sourceLanguage: 'zh', targetLanguage: 'en' };
            } else {
                return { sourceLanguage: 'en', targetLanguage: 'zh' };
            }
        }
        
        // For other language pairs, try to detect or use API
        try {
            const response = await fetch('/api/translation/detect', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text })
            });
            
            if (response.ok) {
                const result = await response.json();
                const detectedLang = result.language;
                
                if (detectedLang === languageA) {
                    return { sourceLanguage: languageA, targetLanguage: languageB };
                } else if (detectedLang === languageB) {
                    return { sourceLanguage: languageB, targetLanguage: languageA };
                }
            }
        } catch (error) {
            console.warn('Language detection failed, using fallback:', error);
        }
        
        // Fallback: assume languageA -> languageB
        return { sourceLanguage: languageA, targetLanguage: languageB };
    }

    async translateText(text, sourceLanguage = 'auto', targetLanguage = 'en') {
        try {
            this.showLoading('Translating...');

            const response = await fetch('/api/translation/translate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    text: text,
                    source_language: sourceLanguage,
                    target_language: targetLanguage,
                    session_id: this.state.sessionId || 'frontend-test'
                })
            });

            if (!response.ok) {
                throw new Error(`Translation failed: ${response.status}`);
            }

            const result = await response.json();
            
            this.displayTranslationResult(text, result);
            this.addLog('SUCCESS', `Translation completed (${result.backend_used || 'unknown'} backend)`);
            
            this.hideLoading();

        } catch (error) {
            console.error('Translation failed:', error);
            this.addLog('ERROR', `Translation failed: ${error.message}`);
            this.hideLoading();
            this.showNotification('Translation failed', 'error');
        }
    }

    displayTranslationResult(originalText, result) {
        // Update original text display
        const originalTextEl = document.getElementById('originalText');
        if (originalTextEl) {
            const placeholder = originalTextEl.querySelector('.placeholder');
            if (placeholder) placeholder.remove();
            
            const textElement = document.createElement('div');
            textElement.className = 'text-item';
            textElement.innerHTML = `
                <div class="text-meta">
                    <span class="language">${result.source_language || 'auto'}</span>
                    <span class="timestamp">${new Date().toLocaleTimeString()}</span>
                </div>
                <div class="text-content">${originalText}</div>
            `;
            originalTextEl.appendChild(textElement);
        }

        // Update translated text display
        const translatedTextEl = document.getElementById('translatedText');
        if (translatedTextEl) {
            const placeholder = translatedTextEl.querySelector('.placeholder');
            if (placeholder) placeholder.remove();
            
            const textElement = document.createElement('div');
            textElement.className = 'text-item translation-result';
            textElement.innerHTML = `
                <div class="text-meta">
                    <span class="language">${result.target_language || 'en'}</span>
                    <span class="confidence">Confidence: ${(result.confidence_score * 100).toFixed(0)}%</span>
                    <span class="backend">Backend: ${result.backend_used || 'unknown'}</span>
                    <span class="timestamp">${new Date().toLocaleTimeString()}</span>
                </div>
                <div class="text-content">${result.translated_text}</div>
            `;
            translatedTextEl.appendChild(textElement);
        }

        // Auto-scroll to bottom
        if (translatedTextEl) {
            translatedTextEl.scrollTop = translatedTextEl.scrollHeight;
        }
    }

    toggleTranslation() {
        const enableBtn = document.getElementById('enableTranslation');
        if (!enableBtn) return;

        if (this.state.translationEnabled) {
            this.state.translationEnabled = false;
            enableBtn.innerHTML = '<i class="fas fa-play"></i> Enable Translation';
            enableBtn.classList.remove('btn-danger');
            enableBtn.classList.add('btn-primary');
            this.addLog('INFO', 'Translation disabled');
        } else {
            this.state.translationEnabled = true;
            enableBtn.innerHTML = '<i class="fas fa-stop"></i> Disable Translation';
            enableBtn.classList.remove('btn-primary');
            enableBtn.classList.add('btn-danger');
            this.addLog('SUCCESS', 'Translation enabled');
        }
    }

    openFileTranslation() {
        // Create file input for text files
        const fileInput = document.createElement('input');
        fileInput.type = 'file';
        fileInput.accept = '.txt,.srt,.vtt';
        fileInput.style.display = 'none';
        
        fileInput.onchange = async (e) => {
            const file = e.target.files[0];
            if (file) {
                await this.translateFile(file);
            }
        };
        
        document.body.appendChild(fileInput);
        fileInput.click();
        document.body.removeChild(fileInput);
    }

    async translateFile(file) {
        try {
            this.showLoading('Reading file...');
            
            const text = await this.readFileAsText(file);
            const languageA = document.getElementById('languageA')?.value || 'zh';
            const languageB = document.getElementById('languageB')?.value || 'en';
            
            this.hideLoading();
            
            // Auto-detect direction for file content
            const { sourceLanguage, targetLanguage } = await this.detectTranslationDirection(text, languageA, languageB);
            await this.translateText(text, sourceLanguage, targetLanguage);
            
        } catch (error) {
            console.error('File translation failed:', error);
            this.addLog('ERROR', `File translation failed: ${error.message}`);
            this.hideLoading();
            this.showNotification('File translation failed', 'error');
        }
    }

    readFileAsText(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => resolve(e.target.result);
            reader.onerror = (e) => reject(new Error('Failed to read file'));
            reader.readAsText(file);
        });
    }

    onLanguageSettingsChange() {
        const languageA = document.getElementById('languageA')?.value;
        const languageB = document.getElementById('languageB')?.value;
        
        this.addLog('INFO', `Bidirectional languages: ${languageA} ⟷ ${languageB}`);
        
        // Save settings to localStorage
        if (languageA) localStorage.setItem('livetranslate-language-a', languageA);
        if (languageB) localStorage.setItem('livetranslate-language-b', languageB);
    }

    onTranslationModeChange() {
        const mode = document.getElementById('translationMode')?.value;
        this.addLog('INFO', `Translation mode: ${mode}`);
        
        // Save setting to localStorage
        if (mode) localStorage.setItem('livetranslate-translation-mode', mode);
    }

    handleTranslationUpdate(data) {
        // Update translation display from WebSocket
        console.log('Translation update:', data);
        
        if (data.translated_text) {
            this.displayTranslationResult(data.original_text || '', data);
        }
    }
    
    // Additional helper functions
    async saveServiceConfiguration() {
        try {
            const config = {
                services: {}
            };
            
            // Collect service URLs from inputs
            Object.keys(this.config.services).forEach(serviceName => {
                const urlInput = document.getElementById(`${serviceName}Url`);
                if (urlInput) {
                    config.services[serviceName] = { url: urlInput.value };
                }
            });
            
            const response = await fetch('/api/config', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(config)
            });
            
            if (response.ok) {
                this.showNotification('Configuration saved successfully', 'success');
                this.config = { ...this.config, ...config };
                this.addLog('SUCCESS', 'Service configuration updated');
                
                // Recheck service health with new URLs
                await this.checkAllServicesHealth();
            } else {
                throw new Error(`Failed to save configuration: ${response.status}`);
            }
            
        } catch (error) {
            console.error('Failed to save configuration:', error);
            this.showNotification('Failed to save configuration', 'error');
            this.addLog('ERROR', `Configuration save failed: ${error.message}`);
        }
    }
    
    copyTranslationResults() {
        const translatedTextEl = document.getElementById('translatedText');
        if (!translatedTextEl) return;
        
        // Get all translation results
        const results = translatedTextEl.querySelectorAll('.text-item .text-content');
        if (results.length === 0) {
            this.showNotification('No translations to copy', 'warning');
            return;
        }
        
        const textToCopy = Array.from(results)
            .map(el => el.textContent.trim())
            .join('\n\n');
            
        if (navigator.clipboard) {
            navigator.clipboard.writeText(textToCopy).then(() => {
                this.showNotification('Translations copied to clipboard', 'success');
                this.addLog('INFO', 'Translation results copied to clipboard');
            }).catch(error => {
                console.error('Failed to copy to clipboard:', error);
                this.showNotification('Failed to copy to clipboard', 'error');
            });
        } else {
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = textToCopy;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
            this.showNotification('Translations copied to clipboard', 'success');
        }
    }
    
    exportTranslationResults() {
        const translatedTextEl = document.getElementById('translatedText');
        if (!translatedTextEl) return;
        
        // Get all translation results with metadata
        const results = translatedTextEl.querySelectorAll('.text-item');
        if (results.length === 0) {
            this.showNotification('No translations to export', 'warning');
            return;
        }
        
        let exportText = 'LiveTranslate Translation Export\n';
        exportText += `Generated: ${new Date().toLocaleString()}\n`;
        exportText += '=' .repeat(50) + '\n\n';
        
        results.forEach((result, index) => {
            const meta = result.querySelector('.text-meta');
            const content = result.querySelector('.text-content');
            
            if (meta && content) {
                exportText += `Translation ${index + 1}:\n`;
                exportText += `${meta.textContent.trim()}\n`;
                exportText += `${content.textContent.trim()}\n\n`;
                exportText += '-'.repeat(30) + '\n\n';
            }
        });
        
        this.downloadTextFile(exportText, `translations-${new Date().toISOString().split('T')[0]}.txt`);
        this.addLog('SUCCESS', 'Translation results exported');
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.liveTranslateApp = new LiveTranslateApp();
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = LiveTranslateApp;
} 