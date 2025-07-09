// Audio Test Page JavaScript

// Use existing config if available, otherwise create test config
let testConfig;
if (typeof config !== 'undefined') {
    testConfig = config;
} else {
    testConfig = {
        orchestrationUrl: 'http://localhost:3000',
        whisperApiUrl: 'http://localhost:3000/api/whisper',
        serverUrl: 'http://localhost:3000'
    };
}

// Test state management - separate from main app state
let testState = {
    isRecording: false,
    isPlaying: false,
    recordedBlob: null,
    testStream: null,
    testRecorder: null,
    testAudio: null,
    audioContext: null,
    analyser: null,
    microphone: null,
    animationFrame: null,
    audioProcessor: null
};

// Initialize the test page
function initializeTestPage() {
    addTestLog('INFO', 'Initializing audio test page...');
    
    // Check browser capabilities first
    checkBrowserCapabilities();
    
    // Load audio devices
    loadTestAudioDevices();
    
    // Setup event listeners
    setupTestEventListeners();
    
    // Load available models
    loadServerModels();
    
    // Initialize audio processor
    if (window.AudioProcessingTester) {
        testState.audioProcessor = new AudioProcessingTester();
        testState.audioProcessor.initialize().then(() => {
            addTestLog('SUCCESS', 'Audio processor initialized');
        }).catch(err => {
            addTestLog('ERROR', `Audio processor init failed: ${err.message}`);
        });
    }
    
    addTestLog('SUCCESS', 'Audio test page initialized successfully');
}

function checkBrowserCapabilities() {
    // Check browser capabilities
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        addTestLog('ERROR', 'Browser does not support audio recording');
        return false;
    } else {
        addTestLog('SUCCESS', 'Browser supports audio recording');
    }
    
    if (!window.MediaRecorder) {
        addTestLog('ERROR', 'Browser does not support MediaRecorder');
        return false;
    } else {
        addTestLog('SUCCESS', 'Browser supports MediaRecorder');
        const supported = getSupportedMimeTypes();
        addTestLog('INFO', `Supported formats: ${supported.join(', ')}`);
    }
    
    return true;
}

// Get supported MIME types for MediaRecorder
function getSupportedMimeTypes() {
    const types = [
        'audio/webm;codecs=opus',
        'audio/webm',
        'audio/ogg;codecs=opus',
        'audio/mp4',
        'audio/wav',
        'audio/mpeg'
    ];
    
    const supported = [];
    types.forEach(type => {
        try {
            if (MediaRecorder.isTypeSupported(type)) {
                supported.push(type);
            }
        } catch (e) {
            // Type not supported
        }
    });
    
    return supported;
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function() {
    // Only initialize if we're on the test page
    if (document.getElementById('startTestButton')) {
        console.log('Test page detected, initializing...');
        setTimeout(initializeTestPage, 100); // Small delay to ensure DOM is ready
    }
});

function setupTestEventListeners() {
    const elements = {
        startTestButton: document.getElementById('startTestButton'),
        playTestButton: document.getElementById('playTestButton'),
        analyzeTestButton: document.getElementById('analyzeTestButton'),
        transcribeTestButton: document.getElementById('transcribeTestButton'),
        processAudioButton: document.getElementById('processAudioButton'),
        clearProcessingButton: document.getElementById('clearProcessingButton'),
        checkCompatibilityButton: document.getElementById('checkCompatibilityButton'),
        testConnectionButton: document.getElementById('testConnectionButton'),
        testTranscriptionButton: document.getElementById('testTranscriptionButton'),
        clearLogsButton: document.getElementById('clearLogsButton')
    };
    
    // Check which elements exist
    Object.entries(elements).forEach(([name, element]) => {
        if (element) {
            addTestLog('DEBUG', `Found element: ${name}`);
        } else {
            addTestLog('ERROR', `Missing element: ${name}`);
        }
    });
    
    if (elements.startTestButton) {
        elements.startTestButton.addEventListener('click', toggleTestRecording);
        addTestLog('SUCCESS', 'Start button listener attached');
    }
    if (elements.playTestButton) {
        elements.playTestButton.addEventListener('click', playTestRecording);
        addTestLog('SUCCESS', 'Play button listener attached');
    }
    if (elements.analyzeTestButton) {
        elements.analyzeTestButton.addEventListener('click', analyzeTestAudio);
        addTestLog('SUCCESS', 'Analyze button listener attached');
    }
    if (elements.transcribeTestButton) {
        elements.transcribeTestButton.addEventListener('click', testTranscription);
        addTestLog('SUCCESS', 'Transcribe button listener attached');
    }
    if (elements.processAudioButton) {
        elements.processAudioButton.addEventListener('click', processAudioPipeline);
        addTestLog('SUCCESS', 'Process audio button listener attached');
    }
    if (elements.clearProcessingButton) {
        elements.clearProcessingButton.addEventListener('click', clearProcessingResults);
        addTestLog('SUCCESS', 'Clear processing button listener attached');
    }
    if (elements.checkCompatibilityButton) {
        elements.checkCompatibilityButton.addEventListener('click', checkBrowserCompatibility);
        addTestLog('SUCCESS', 'Compatibility button listener attached');
    }
    if (elements.testConnectionButton) {
        elements.testConnectionButton.addEventListener('click', testServerConnection);
        addTestLog('SUCCESS', 'Connection test button listener attached');
    }
    if (elements.testTranscriptionButton) {
        elements.testTranscriptionButton.addEventListener('click', testServerTranscription);
        addTestLog('SUCCESS', 'Server transcription button listener attached');
    }
    if (elements.clearLogsButton) {
        elements.clearLogsButton.addEventListener('click', clearTestLogs);
        addTestLog('SUCCESS', 'Clear logs button listener attached');
    }
}

async function loadTestAudioDevices() {
    try {
        addTestLog('INFO', 'Loading audio devices...');
        
        // Check if getUserMedia is available
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            addTestLog('ERROR', 'getUserMedia not supported in this browser');
            return;
        }
        
        // Request permission first
        addTestLog('DEBUG', 'Requesting microphone permission...');
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        addTestLog('SUCCESS', 'Microphone permission granted');
        
        // Stop the stream immediately
        stream.getTracks().forEach(track => track.stop());
        
        // Enumerate devices
        addTestLog('DEBUG', 'Enumerating audio devices...');
        const devices = await navigator.mediaDevices.enumerateDevices();
        const audioDevices = devices.filter(device => device.kind === 'audioinput');
        
        addTestLog('DEBUG', `Total devices found: ${devices.length}`);
        addTestLog('DEBUG', `Audio input devices: ${audioDevices.length}`);
        
        const deviceSelect = document.getElementById('testDevice');
        if (!deviceSelect) {
            addTestLog('ERROR', 'Device select element not found');
            return;
        }
        
        deviceSelect.innerHTML = '<option value="">Default Device</option>';
        audioDevices.forEach((device, index) => {
            const option = document.createElement('option');
            option.value = device.deviceId;
            option.textContent = device.label || `Audio Device ${index + 1}`;
            deviceSelect.appendChild(option);
            addTestLog('DEBUG', `Added device: ${option.textContent} (ID: ${device.deviceId.substring(0, 8)}...)`);
        });
        
        addTestLog('SUCCESS', `Found ${audioDevices.length} audio input devices`);
        
    } catch (error) {
        addTestLog('ERROR', `Failed to load audio devices: ${error.message}`);
        if (error.name === 'NotAllowedError') {
            addTestLog('INFO', 'Please allow microphone access to test audio recording');
        }
    }
}

async function toggleTestRecording() {
    if (testState.isRecording) {
        stopTestRecording();
    } else {
        await startTestRecording();
    }
}

async function startTestRecording() {
    try {
        const duration = parseInt(document.getElementById('testDuration').value) || 5;
        const deviceId = document.getElementById('testDevice').value;
        const sampleRate = document.getElementById('testSampleRate').value;
        const format = document.getElementById('testFormat').value;
        
        addTestLog('INFO', `Starting ${duration}s test recording...`);
        
        // Audio constraints
        const constraints = {
            audio: {
                deviceId: deviceId ? { exact: deviceId } : undefined,
                sampleRate: sampleRate ? parseInt(sampleRate) : undefined,
                echoCancellation: false,
                noiseSuppression: false,
                autoGainControl: false
            }
        };
        
        testState.testStream = await navigator.mediaDevices.getUserMedia(constraints);
        
        // Log actual settings
        const audioTrack = testState.testStream.getAudioTracks()[0];
        const settings = audioTrack.getSettings();
        addTestLog('INFO', `Recording settings: ${settings.sampleRate}Hz, ${settings.channelCount} channels`);
        
        // Setup audio monitoring
        setupTestAudioMonitoring();
        
        // Determine MIME type
        let mimeType = 'audio/webm;codecs=opus';
        if (format === 'auto') {
            // Auto-detect best format
            if (MediaRecorder.isTypeSupported('audio/mp4')) {
                mimeType = 'audio/mp4';
            } else if (MediaRecorder.isTypeSupported('audio/ogg;codecs=opus')) {
                mimeType = 'audio/ogg;codecs=opus';
            }
        } else if (format !== 'auto') {
            mimeType = format;
        }
        
        addTestLog('INFO', `Using recording format: ${mimeType}`);
        
        // Create MediaRecorder
        testState.testRecorder = new MediaRecorder(testState.testStream, {
            mimeType: mimeType
        });
        
        const testChunks = [];
        
        testState.testRecorder.ondataavailable = function(event) {
            if (event.data.size > 0) {
                testChunks.push(event.data);
                addTestLog('DEBUG', `Chunk received: ${event.data.size} bytes`);
            }
        };
        
        testState.testRecorder.onstop = function() {
            stopTestAudioMonitoring();
            
            testState.recordedBlob = new Blob(testChunks, { type: mimeType });
            testState.isRecording = false;
            updateTestUI();
            
            addTestLog('SUCCESS', `Recording completed: ${testState.recordedBlob.size} bytes`);
            
            // Enable analysis buttons
            document.getElementById('playTestButton').disabled = false;
            document.getElementById('analyzeTestButton').disabled = false;
            document.getElementById('transcribeTestButton').disabled = false;
            document.getElementById('processAudioButton').disabled = false;
        };
        
        // Start recording
        testState.testRecorder.start();
        testState.isRecording = true;
        updateTestUI();
        
        // Stop after duration
        setTimeout(() => {
            if (testState.testRecorder && testState.isRecording) {
                testState.testRecorder.stop();
                testState.testStream.getTracks().forEach(track => track.stop());
            }
        }, duration * 1000);
        
    } catch (error) {
        addTestLog('ERROR', `Test recording failed: ${error.message}`);
        testState.isRecording = false;
        updateTestUI();
    }
}

function stopTestRecording() {
    if (testState.testRecorder && testState.isRecording) {
        testState.testRecorder.stop();
        if (testState.testStream) {
            testState.testStream.getTracks().forEach(track => track.stop());
        }
        stopTestAudioMonitoring();
    }
}

function setupTestAudioMonitoring() {
    try {
        testState.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        testState.analyser = testState.audioContext.createAnalyser();
        testState.microphone = testState.audioContext.createMediaStreamSource(testState.testStream);
        
        testState.analyser.fftSize = 1024;
        testState.analyser.smoothingTimeConstant = 0.3;
        testState.microphone.connect(testState.analyser);
        
        // Start monitoring
        monitorAudioLevel();
        drawWaveform();
        
    } catch (error) {
        addTestLog('WARNING', `Audio monitoring setup failed: ${error.message}`);
    }
}

function stopTestAudioMonitoring() {
    if (testState.animationFrame) {
        cancelAnimationFrame(testState.animationFrame);
        testState.animationFrame = null;
    }
    
    if (testState.microphone) {
        testState.microphone.disconnect();
        testState.microphone = null;
    }
    
    if (testState.audioContext) {
        testState.audioContext.close();
        testState.audioContext = null;
    }
    
    testState.analyser = null;
    
    // Reset UI
    const levelBar = document.getElementById('testAudioLevelBar');
    const levelText = document.getElementById('testAudioLevel');
    if (levelBar) levelBar.style.width = '0%';
    if (levelText) levelText.textContent = '0%';
}

function monitorAudioLevel() {
    if (!testState.analyser) return;
    
    const bufferLength = testState.analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    testState.analyser.getByteFrequencyData(dataArray);
    
    // Calculate RMS level
    let sum = 0;
    for (let i = 0; i < bufferLength; i++) {
        sum += dataArray[i] * dataArray[i];
    }
    const rms = Math.sqrt(sum / bufferLength);
    const level = Math.min(100, (rms / 128) * 100);
    
    // Update level display
    const levelBar = document.getElementById('testAudioLevelBar');
    const levelText = document.getElementById('testAudioLevel');
    if (levelBar) levelBar.style.width = `${level}%`;
    if (levelText) levelText.textContent = `${level.toFixed(1)}%`;
    
    // Continue monitoring
    if (testState.isRecording) {
        setTimeout(monitorAudioLevel, 50);
    }
}

function drawWaveform() {
    if (!testState.analyser) return;
    
    const canvas = document.getElementById('testWaveform');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const bufferLength = testState.analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    const draw = () => {
        if (!testState.analyser) return;
        
        testState.animationFrame = requestAnimationFrame(draw);
        
        testState.analyser.getByteFrequencyData(dataArray);
        
        // Clear canvas
        ctx.fillStyle = 'rgb(30, 30, 30)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Draw frequency bars
        const barWidth = (canvas.width / bufferLength) * 2.5;
        let barHeight;
        let x = 0;
        
        for (let i = 0; i < bufferLength; i++) {
            barHeight = (dataArray[i] / 255) * canvas.height;
            
            // Color gradient
            const hue = (i / bufferLength) * 360;
            ctx.fillStyle = `hsl(${hue}, 70%, 50%)`;
            
            ctx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);
            x += barWidth + 1;
        }
    };
    
    draw();
}

async function playTestRecording() {
    if (!testState.recordedBlob) {
        addTestLog('WARNING', 'No recording to play');
        return;
    }
    
    try {
        addTestLog('INFO', `Playing recording: ${testState.recordedBlob.size} bytes`);
        
        const audioUrl = URL.createObjectURL(testState.recordedBlob);
        testState.testAudio = new Audio(audioUrl);
        testState.testAudio.volume = 1.0;
        
        testState.testAudio.onloadedmetadata = () => {
            addTestLog('INFO', `Audio duration: ${testState.testAudio.duration.toFixed(2)}s`);
        };
        
        testState.testAudio.onended = () => {
            testState.isPlaying = false;
            updateTestUI();
            URL.revokeObjectURL(audioUrl);
            addTestLog('SUCCESS', 'Playback completed');
        };
        
        testState.testAudio.onerror = (error) => {
            testState.isPlaying = false;
            updateTestUI();
            URL.revokeObjectURL(audioUrl);
            addTestLog('ERROR', `Playback failed: ${error.type}`);
        };
        
        testState.isPlaying = true;
        updateTestUI();
        await testState.testAudio.play();
        
    } catch (error) {
        addTestLog('ERROR', `Playback error: ${error.message}`);
        testState.isPlaying = false;
        updateTestUI();
    }
}

async function analyzeTestRecording() {
    // Alias for analyzeTestAudio for compatibility
    return analyzeTestAudio();
}

async function analyzeTestAudio() {
    if (!testState.recordedBlob) {
        addTestLog('WARNING', 'No recording to analyze');
        return;
    }
    
    try {
        addTestLog('INFO', 'Analyzing audio recording...');
        
        // Basic analysis
        const results = {
            size: testState.recordedBlob.size,
            type: testState.recordedBlob.type,
            duration: testState.testAudio ? testState.testAudio.duration : 'Unknown'
        };
        
        // Display results
        const resultsDiv = document.getElementById('testResults');
        if (resultsDiv) {
            resultsDiv.innerHTML = `
                <h4>Audio Analysis Results</h4>
                <p><strong>File Size:</strong> ${(results.size / 1024).toFixed(1)} KB</p>
                <p><strong>Format:</strong> ${results.type}</p>
                <p><strong>Duration:</strong> ${results.duration}s</p>
                <p><strong>Estimated Bitrate:</strong> ${results.duration !== 'Unknown' ? 
                    ((results.size * 8) / (results.duration * 1000)).toFixed(1) + ' kbps' : 'Unknown'}</p>
            `;
        }
        
        addTestLog('SUCCESS', 'Audio analysis completed');
        
    } catch (error) {
        addTestLog('ERROR', `Analysis failed: ${error.message}`);
    }
}

async function transcribeTestRecording() {
    // Alias for testTranscription for compatibility
    return testTranscription();
}

async function testTranscription() {
    if (!testState.recordedBlob) {
        addTestLog('WARNING', 'No recording to transcribe');
        return;
    }
    
    const modelSelect = document.getElementById('testModel');
    const selectedModel = modelSelect?.value;
    
    if (!selectedModel) {
        addTestLog('WARNING', 'Please select a model first');
        return;
    }
    
    try {
        addTestLog('INFO', `Testing transcription with model: ${selectedModel}`);
        
        const url = `${testConfig.whisperApiUrl}/transcribe/${selectedModel}`;
        const formData = new FormData();
        formData.append('audio', testState.recordedBlob, 'audio.wav');
        
        const response = await fetch(url, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok) {
            addTestLog('SUCCESS', `Transcription result: "${data.text}"`);
            addTestLog('INFO', `Processing time: ${data.processing_time}`);
            addTestLog('INFO', `Audio length: ${data.audio_length}`);
            addTestLog('INFO', `Device: ${data.device}`);
            
            // Update test results
            const resultsDiv = document.getElementById('testResults');
            if (resultsDiv) {
                resultsDiv.innerHTML += `
                    <h4>Transcription Test Results</h4>
                    <p><strong>Transcribed Text:</strong> "${data.text}"</p>
                    <p><strong>Processing Time:</strong> ${data.processing_time}</p>
                    <p><strong>Audio Length:</strong> ${data.audio_length}</p>
                    <p><strong>Device Used:</strong> ${data.device}</p>
                    <p><strong>Language:</strong> ${data.language}</p>
                `;
            }
            
        } else {
            addTestLog('ERROR', `Transcription failed: ${data.error}`);
        }
        
    } catch (error) {
        addTestLog('ERROR', `Transcription test error: ${error.message}`);
    }
}

function checkBrowserCompatibility() {
    addTestLog('INFO', 'Checking browser compatibility...');
    
    const results = {
        mediaRecorderSupport: typeof MediaRecorder !== 'undefined',
        webAudioSupport: typeof AudioContext !== 'undefined' || typeof webkitAudioContext !== 'undefined',
        getUserMediaSupport: !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia),
        formats: []
    };
    
    // Test audio formats
    const testFormats = [
        'audio/mp4',
        'audio/mp4;codecs=mp4a.40.2',
        'audio/ogg;codecs=opus',
        'audio/webm;codecs=opus',
        'audio/webm',
        'audio/wav'
    ];
    
    testFormats.forEach(format => {
        if (typeof MediaRecorder !== 'undefined') {
            const supported = MediaRecorder.isTypeSupported(format);
            results.formats.push({ format, supported });
        }
    });
    
    // Display results
    const resultsDiv = document.getElementById('compatibilityResults');
    if (resultsDiv) {
        let html = `
            <h4>Browser Compatibility Results</h4>
            <p><strong>MediaRecorder API:</strong> ${results.mediaRecorderSupport ? '✅ Supported' : '❌ Not Supported'}</p>
            <p><strong>Web Audio API:</strong> ${results.webAudioSupport ? '✅ Supported' : '❌ Not Supported'}</p>
            <p><strong>getUserMedia API:</strong> ${results.getUserMediaSupport ? '✅ Supported' : '❌ Not Supported'}</p>
            <h5>Audio Format Support:</h5>
            <ul>
        `;
        
        results.formats.forEach(({ format, supported }) => {
            html += `<li>${format}: ${supported ? '✅ Supported' : '❌ Not Supported'}</li>`;
        });
        
        html += '</ul>';
        resultsDiv.innerHTML = html;
    }
    
    addTestLog('SUCCESS', 'Compatibility check completed');
}

async function loadServerModels() {
    try {
        const response = await fetch(`${testConfig.whisperApiUrl}/models`);
        const data = await response.json();
        
        const modelSelect = document.getElementById('testModel');
        const models = data.available_models || data.models || [];
        if (modelSelect && models.length > 0) {
            modelSelect.innerHTML = '<option value="">Select a model</option>';
            models.forEach(model => {
                const option = document.createElement('option');
                option.value = model;
                option.textContent = model;
                modelSelect.appendChild(option);
            });
        }
        
        addTestLog('SUCCESS', `Loaded ${models.length} models: ${models.join(', ')}`);
        
    } catch (error) {
        addTestLog('ERROR', `Failed to load models: ${error.message}`);
    }
}

async function testServerConnection() {
    try {
        addTestLog('INFO', 'Testing server connection...');
        
        const serverUrl = document.getElementById('serverUrl').value || testConfig.serverUrl;
        const response = await fetch(`${serverUrl}/health`);
        const data = await response.json();
        
        if (response.ok) {
            addTestLog('SUCCESS', 'Server connection successful');
            
            const resultsDiv = document.getElementById('serverTestResults');
            if (resultsDiv) {
                resultsDiv.innerHTML = `
                    <h4>Server Connection Results</h4>
                    <p><strong>Status:</strong> ✅ Connected</p>
                    <p><strong>Device:</strong> ${data.device}</p>
                    <p><strong>Models Available:</strong> ${data.models_available}</p>
                    <p><strong>Server URL:</strong> ${serverUrl}</p>
                `;
            }
            
            // Enable transcription test
            document.getElementById('testTranscriptionButton').disabled = false;
            
        } else {
            throw new Error(`Server responded with status ${response.status}`);
        }
        
    } catch (error) {
        addTestLog('ERROR', `Server connection failed: ${error.message}`);
        
        const resultsDiv = document.getElementById('serverTestResults');
        if (resultsDiv) {
            resultsDiv.innerHTML = `
                <h4>Server Connection Results</h4>
                <p><strong>Status:</strong> ❌ Connection Failed</p>
                <p><strong>Error:</strong> ${error.message}</p>
            `;
        }
    }
}

async function testServerTranscription() {
    // This will use the current recorded audio for transcription testing
    await testTranscription();
}

function updateTestUI() {
    const startButton = document.getElementById('startTestButton');
    const playButton = document.getElementById('playTestButton');
    const statusDiv = document.getElementById('testStatus');
    
    if (startButton) {
        if (testState.isRecording) {
            startButton.textContent = '⏹ Stop Recording';
            startButton.classList.add('recording');
        } else {
            startButton.textContent = '🎤 Start Test Recording';
            startButton.classList.remove('recording');
        }
    }
    
    if (playButton) {
        if (testState.isPlaying) {
            playButton.textContent = '⏸ Stop Playback';
        } else {
            playButton.textContent = '▶ Play Recording';
        }
    }
    
    if (statusDiv) {
        if (testState.isRecording) {
            statusDiv.textContent = '● Recording in progress...';
            statusDiv.className = 'test-status recording';
        } else if (testState.isPlaying) {
            statusDiv.textContent = '▶ Playing recording...';
            statusDiv.className = 'test-status playing';
        } else if (testState.recordedBlob) {
            statusDiv.textContent = '✓ Recording ready for testing';
            statusDiv.className = 'test-status ready';
        } else {
            statusDiv.textContent = 'Ready to record';
            statusDiv.className = 'test-status';
        }
    }
}

function addTestLog(level, message) {
    const timestamp = new Date().toLocaleTimeString();
    const logEntry = document.createElement('div');
    logEntry.className = `log-entry ${level}`;
    logEntry.innerHTML = `[${timestamp}] ${message}`;
    
    const logsContent = document.getElementById('testLogsContent');
    if (logsContent) {
        logsContent.appendChild(logEntry);
        logsContent.scrollTop = logsContent.scrollHeight;
        
        // Limit log entries
        while (logsContent.children.length > 50) {
            logsContent.removeChild(logsContent.firstChild);
        }
    }
    
    console.log(`[${level}] ${message}`);
}

function clearTestLogs() {
    const logsContent = document.getElementById('testLogsContent');
    if (logsContent) {
        logsContent.innerHTML = '<div class="log-entry INFO">Logs cleared</div>';
    }
}

// Make sure we have a fallback addLog function if it's not available
if (typeof addLog === 'undefined') {
    window.addLog = function(level, message) {
        console.log(`[${level}] ${message}`);
        // Also try to add to test logs if available
        addTestLog(level, message);
    };
}

// Audio Processing Pipeline Functions
async function processAudioThroughPipeline() {
    // Use enhanced version if available and no issues
    if (window.processAudioPipelineEnhanced && typeof window.processAudioPipelineEnhanced === 'function') {
        try {
            return await window.processAudioPipelineEnhanced();
        } catch (error) {
            addTestLog('WARNING', 'Enhanced pipeline failed, falling back to basic version: ' + error.message);
        }
    }
    
    // Use old method as fallback
    return processAudioPipelineOld();
}

async function processAudioPipeline() {
    // Main entry point - delegate to the appropriate implementation
    return processAudioThroughPipeline();
}

async function processAudioPipelineOld() {
    if (!testState.recordedBlob) {
        addTestLog('WARNING', 'No recording to process');
        return;
    }

    // Initialize audio processor if needed
    if (!testState.audioProcessor) {
        testState.audioProcessor = new AudioProcessingTester();
        await testState.audioProcessor.initialize();
    }

    try {
        const statusDiv = document.getElementById('processingStatus');
        const stagesDiv = document.getElementById('processingStages');
        const analysisDiv = document.getElementById('audioAnalysis');

        statusDiv.textContent = '⏳ Processing audio through pipeline...';
        statusDiv.className = 'test-status processing';
        
        addTestLog('INFO', 'Starting audio pipeline processing...');

        // Process through all stages using the old method if available
        let stages;
        if (testState.audioProcessor.processAudioStages) {
            stages = await testState.audioProcessor.processAudioStages(testState.recordedBlob);
        } else {
            // Manually process stages for compatibility
            const audioBuffer = await testState.audioProcessor.blobToAudioBuffer(testState.recordedBlob);
            stages = {
                original: { blob: testState.recordedBlob, description: 'Original audio' },
                decoded: { audioBuffer: audioBuffer, description: 'Decoded audio' }
            };
        }
        
        statusDiv.textContent = '✅ Audio processing complete';
        statusDiv.className = 'test-status ready';
        
        // Display stages
        stagesDiv.innerHTML = '';
        stagesDiv.style.display = 'block';
        
        Object.entries(stages).forEach(([stageName, stageData]) => {
            const stageCard = createStageCard(stageName, stageData);
            stagesDiv.appendChild(stageCard);
        });
        
        // Show analysis
        analysisDiv.style.display = 'block';
        
        addTestLog('SUCCESS', `Processed ${Object.keys(stages).length} audio stages`);
        
    } catch (error) {
        addTestLog('ERROR', `Pipeline processing failed: ${error.message}`);
        const statusDiv = document.getElementById('processingStatus');
        statusDiv.textContent = '❌ Processing failed';
        statusDiv.className = 'test-status error';
    }
}

function createStageCard(stageName, stageData) {
    const card = document.createElement('div');
    card.className = 'stage-card';
    card.id = `stage-${stageName}`;
    
    // Stage info
    const info = document.createElement('div');
    info.className = 'stage-info';
    
    const title = document.createElement('h4');
    title.textContent = `${getStageIcon(stageName)} ${formatStageName(stageName)}`;
    info.appendChild(title);
    
    const details = document.createElement('div');
    details.className = 'stage-details';
    details.innerHTML = getStageDetails(stageName, stageData);
    info.appendChild(details);
    
    card.appendChild(info);
    
    // Stage controls
    const controls = document.createElement('div');
    controls.className = 'stage-controls';
    
    // Play button
    const playBtn = document.createElement('button');
    playBtn.className = 'button small';
    playBtn.textContent = '▶ Play';
    playBtn.onclick = () => playStageAudio(stageName);
    controls.appendChild(playBtn);
    
    // Analyze button
    const analyzeBtn = document.createElement('button');
    analyzeBtn.className = 'button small';
    analyzeBtn.textContent = '📊 Analyze';
    analyzeBtn.onclick = () => analyzeStageAudio(stageName);
    controls.appendChild(analyzeBtn);
    
    // Export button
    const exportBtn = document.createElement('button');
    exportBtn.className = 'button small';
    exportBtn.textContent = '💾 Export';
    exportBtn.onclick = () => exportStageAudio(stageName);
    controls.appendChild(exportBtn);
    
    // Test with Whisper button
    const whisperBtn = document.createElement('button');
    whisperBtn.className = 'button small';
    whisperBtn.textContent = '🎤 Test';
    whisperBtn.onclick = () => testStageWithWhisper(stageName);
    controls.appendChild(whisperBtn);
    
    card.appendChild(controls);
    
    return card;
}

function getStageIcon(stageName) {
    const icons = {
        'original': '🎵',
        'decoded': '🔍',
        'resampled': '📊',
        'mono': '1️⃣',
        'normalized': '📈',
        'denoised': '🔇',
        'vad_trimmed': '✂️'
    };
    return icons[stageName] || '📁';
}

function formatStageName(stageName) {
    return stageName.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

function getStageDetails(stageName, stageData) {
    let details = [];
    
    if (stageData.description) {
        details.push(stageData.description);
    }
    
    if (stageData.format) {
        details.push(`Format: ${stageData.format}`);
    }
    
    if (stageData.size) {
        details.push(`Size: ${(stageData.size / 1024).toFixed(1)} KB`);
    }
    
    if (stageData.sampleRate) {
        details.push(`Sample Rate: ${stageData.sampleRate} Hz`);
    }
    
    if (stageData.channels !== undefined) {
        details.push(`Channels: ${stageData.channels}`);
    }
    
    if (stageData.duration) {
        details.push(`Duration: ${stageData.duration.toFixed(2)}s`);
    }
    
    return details.join(' • ');
}

async function playStageAudio(stageName) {
    try {
        addTestLog('INFO', `Playing ${stageName} audio...`);
        await testState.audioProcessor.playStage(stageName);
    } catch (error) {
        addTestLog('ERROR', `Failed to play ${stageName}: ${error.message}`);
    }
}

async function analyzeStageAudio(stageName) {
    try {
        const stage = testState.audioProcessor.getStageInfo(stageName);
        if (!stage || !stage.audioBuffer) {
            addTestLog('WARNING', `No audio buffer for stage: ${stageName}`);
            return;
        }
        
        const analysis = testState.audioProcessor.analyzeAudio(stage.audioBuffer);
        
        // Update analysis display
        const analysisGrid = document.getElementById('analysisGrid');
        analysisGrid.innerHTML = '';
        
        Object.entries(analysis).forEach(([key, value]) => {
            const item = document.createElement('div');
            item.className = 'analysis-item';
            item.innerHTML = `
                <label>${formatAnalysisLabel(key)}</label>
                <div class="value">${value}</div>
            `;
            analysisGrid.appendChild(item);
        });
        
        addTestLog('SUCCESS', `Analyzed ${stageName} audio`);
        
    } catch (error) {
        addTestLog('ERROR', `Analysis failed for ${stageName}: ${error.message}`);
    }
}

function formatAnalysisLabel(key) {
    const labels = {
        'rms': 'RMS Level',
        'peak': 'Peak Level',
        'dynamicRange': 'Dynamic Range',
        'clipping': 'Clipping',
        'duration': 'Duration',
        'sampleRate': 'Sample Rate',
        'channels': 'Channels'
    };
    return labels[key] || key;
}

async function exportStageAudio(stageName) {
    try {
        await testState.audioProcessor.exportStage(stageName);
        addTestLog('SUCCESS', `Exported ${stageName} audio`);
    } catch (error) {
        addTestLog('ERROR', `Export failed for ${stageName}: ${error.message}`);
    }
}

async function testStageWithWhisper(stageName) {
    const modelSelect = document.getElementById('testModel');
    const selectedModel = modelSelect?.value;
    
    if (!selectedModel) {
        addTestLog('WARNING', 'Please select a model first');
        return;
    }
    
    try {
        addTestLog('INFO', `Testing ${stageName} with model: ${selectedModel}`);
        
        const result = await testState.audioProcessor.sendToWhisper(stageName, selectedModel);
        
        if (result.text) {
            addTestLog('SUCCESS', `${stageName} transcription: "${result.text}"`);
            addTestLog('INFO', `Processing time: ${result.processing_time}`);
            addTestLog('INFO', `Confidence: ${result.confidence}`);
        } else if (result.error) {
            addTestLog('ERROR', `${stageName} transcription failed: ${result.error}`);
        }
        
    } catch (error) {
        addTestLog('ERROR', `Whisper test failed for ${stageName}: ${error.message}`);
    }
}

function clearProcessingResults() {
    const stagesDiv = document.getElementById('processingStages');
    const analysisDiv = document.getElementById('audioAnalysis');
    const statusDiv = document.getElementById('processingStatus');
    
    stagesDiv.innerHTML = '';
    stagesDiv.style.display = 'none';
    
    analysisDiv.style.display = 'none';
    
    statusDiv.textContent = 'Record audio first, then process through pipeline';
    statusDiv.className = 'test-status';
    
    addTestLog('INFO', 'Cleared processing results');
} 