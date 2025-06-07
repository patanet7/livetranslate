// Audio Test Page JavaScript

// Configuration for test page
const config = config || {
    serverUrl: 'http://localhost:5000'
};

// Test state management - separate from main app state
const testState = {
    isRecording: false,
    isPlaying: false,
    recordedBlob: null,
    testStream: null,
    testRecorder: null,
    testAudio: null,
    audioContext: null,
    analyser: null,
    microphone: null,
    animationFrame: null
};

// Initialize the test page
document.addEventListener('DOMContentLoaded', function() {
    // Only initialize if we're on the test page
    if (document.getElementById('startTestButton')) {
        initializeTestPage();
    }
});

function initializeTestPage() {
    addTestLog('INFO', 'Initializing audio test page...');
    
    // Load audio devices
    loadTestAudioDevices();
    
    // Setup event listeners
    setupTestEventListeners();
    
    // Load available models
    loadServerModels();
    
    addTestLog('SUCCESS', 'Audio test page initialized successfully');
}

function setupTestEventListeners() {
    const startTestButton = document.getElementById('startTestButton');
    const playTestButton = document.getElementById('playTestButton');
    const analyzeTestButton = document.getElementById('analyzeTestButton');
    const transcribeTestButton = document.getElementById('transcribeTestButton');
    const checkCompatibilityButton = document.getElementById('checkCompatibilityButton');
    const testConnectionButton = document.getElementById('testConnectionButton');
    const testTranscriptionButton = document.getElementById('testTranscriptionButton');
    const clearLogsButton = document.getElementById('clearLogsButton');
    
    startTestButton?.addEventListener('click', toggleTestRecording);
    playTestButton?.addEventListener('click', playTestRecording);
    analyzeTestButton?.addEventListener('click', analyzeTestAudio);
    transcribeTestButton?.addEventListener('click', testTranscription);
    checkCompatibilityButton?.addEventListener('click', checkBrowserCompatibility);
    testConnectionButton?.addEventListener('click', testServerConnection);
    testTranscriptionButton?.addEventListener('click', testServerTranscription);
    clearLogsButton?.addEventListener('click', clearTestLogs);
}

async function loadTestAudioDevices() {
    try {
        addTestLog('INFO', 'Loading audio devices...');
        
        // Request permission first
        await navigator.mediaDevices.getUserMedia({ audio: true });
        
        const devices = await navigator.mediaDevices.enumerateDevices();
        const audioDevices = devices.filter(device => device.kind === 'audioinput');
        
        const deviceSelect = document.getElementById('testDevice');
        if (deviceSelect) {
            deviceSelect.innerHTML = '<option value="">Default Device</option>';
            audioDevices.forEach(device => {
                const option = document.createElement('option');
                option.value = device.deviceId;
                option.textContent = device.label || `Audio Device ${device.deviceId.substring(0, 8)}`;
                deviceSelect.appendChild(option);
            });
        }
        
        addTestLog('SUCCESS', `Found ${audioDevices.length} audio input devices`);
        
    } catch (error) {
        addTestLog('ERROR', `Failed to load audio devices: ${error.message}`);
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
        
        const url = `${config.serverUrl}/transcribe/${selectedModel}`;
        const response = await fetch(url, {
            method: 'POST',
            body: testState.recordedBlob
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
        const response = await fetch(`${config.serverUrl}/models`);
        const data = await response.json();
        
        const modelSelect = document.getElementById('testModel');
        if (modelSelect && data.models) {
            modelSelect.innerHTML = '<option value="">Select a model</option>';
            data.models.forEach(model => {
                const option = document.createElement('option');
                option.value = model;
                option.textContent = model;
                modelSelect.appendChild(option);
            });
        }
        
        addTestLog('SUCCESS', `Loaded ${data.models ? data.models.length : 0} models`);
        
    } catch (error) {
        addTestLog('ERROR', `Failed to load models: ${error.message}`);
    }
}

async function testServerConnection() {
    try {
        addTestLog('INFO', 'Testing server connection...');
        
        const serverUrl = document.getElementById('serverUrl').value || config.serverUrl;
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