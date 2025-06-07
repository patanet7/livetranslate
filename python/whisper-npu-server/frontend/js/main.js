// Main application state and configuration
const config = {
    serverUrl: 'http://localhost:5000',
    chunkDuration: 1000,
    maxTranscripts: 50,
    maxRetries: 3,
    retryDelay: 1000
};

const state = {
    isRecording: false,
    isStreaming: false,
    currentStream: null,
    mediaRecorder: null,
    audioChunks: [],
    selectedModel: '',
    availableModels: [],
    serverStatus: 'unknown',
    allTranscripts: [],
    lastAudioBlob: null,
    chunkCount: 0,
    processingChunk: false,
    
    // Audio visualization
    audioContext: null,
    analyser: null,
    microphone: null,
    audioLevelInterval: null,
    fftAnimationFrame: null,
    
    // Audio testing
    audioTest: {
        isRecording: false,
        isPlaying: false,
        recordedBlob: null,
        testStream: null,
        testRecorder: null,
        testAudio: null
    }
};

// DOM elements cache
const elements = {};

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeElements();
    setupEventListeners();
    initializeApp();
});

function initializeElements() {
    // Cache all DOM elements
    elements.statusDot = document.getElementById('statusDot');
    elements.statusText = document.getElementById('statusText');
    elements.deviceType = document.getElementById('deviceType');
    elements.modelCount = document.getElementById('modelCount');
    elements.transcriptionContent = document.getElementById('transcriptionContent');
    elements.modelsContent = document.getElementById('modelsContent');
    elements.logsContent = document.getElementById('logsContent');
    elements.selectedModel = document.getElementById('selectedModel');
    elements.recordButton = document.getElementById('recordButton');
    elements.streamButton = document.getElementById('streamButton');
    elements.clearTranscripts = document.getElementById('clearTranscripts');
    elements.exportTranscripts = document.getElementById('exportTranscripts');
    elements.downloadAudio = document.getElementById('downloadAudio');
    elements.refreshStatus = document.getElementById('refreshStatus');
    elements.refreshModels = document.getElementById('refreshModels');
    elements.audioDevice = document.getElementById('audioDevice');
    elements.sampleRate = document.getElementById('sampleRate');
    elements.audioFileInput = document.getElementById('audioFileInput');
    elements.transcribeFileButton = document.getElementById('transcribeFileButton');
    elements.audioLevelBar = document.getElementById('audioLevelBar');
    elements.audioLevelText = document.getElementById('audioLevelText');
    elements.selectedAudioDevice = document.getElementById('selectedAudioDevice');
    elements.fftCanvas = document.getElementById('fftCanvas');
    
    // Streaming output (use transcription content for streaming results)
    elements.streamingOutput = document.getElementById('transcriptionContent');
    
    // Audio test elements
    elements.testAudioButton = document.getElementById('testAudioButton');
    elements.playTestButton = document.getElementById('playTestButton');
    elements.testStatus = document.getElementById('testStatus');
    elements.testWaveform = document.getElementById('testWaveform');
}

function setupEventListeners() {
    // Main control buttons
    elements.recordButton?.addEventListener('click', toggleRecording);
    elements.streamButton?.addEventListener('click', toggleStreaming);
    elements.clearTranscripts?.addEventListener('click', clearTranscripts);
    elements.exportTranscripts?.addEventListener('click', exportTranscripts);
    elements.downloadAudio?.addEventListener('click', downloadLastAudio);
    elements.refreshStatus?.addEventListener('click', () => checkServerHealth(false));
    elements.refreshModels?.addEventListener('click', loadModels);
    elements.transcribeFileButton?.addEventListener('click', transcribeUploadedFile);
    
    // Audio test buttons
    elements.testAudioButton?.addEventListener('click', toggleAudioTest);
    elements.playTestButton?.addEventListener('click', playTestAudio);
    
    // Model selector
    elements.selectedModel?.addEventListener('change', function() {
        state.selectedModel = this.value;
        updateModelSelection(this.value);
    });
    
    // Wait a moment for audio.js to load, then setup device handlers
    setTimeout(() => {
        setupAudioDeviceHandlers();
    }, 100);
}

function setupAudioDeviceHandlers() {
    // Audio device and sample rate changes - use AudioVisualization module handlers
    if (elements.audioDevice) {
        addLog('DEBUG', 'Setting up audio device change handler');
        elements.audioDevice.addEventListener('change', function() {
            const selectedDevice = this.value;
            const selectedDeviceName = this.options[this.selectedIndex]?.text || 'Default';
            addLog('INFO', `Audio device dropdown changed to: ${selectedDeviceName} (ID: ${selectedDevice || 'default'})`);
            
            // Check if AudioVisualization is available
            if (typeof window.AudioVisualization !== 'undefined' && window.AudioVisualization.handleAudioDeviceChange) {
                window.AudioVisualization.handleAudioDeviceChange();
            } else if (typeof AudioVisualization !== 'undefined' && AudioVisualization.handleAudioDeviceChange) {
                AudioVisualization.handleAudioDeviceChange();
            } else {
                addLog('ERROR', 'AudioVisualization module not available, trying to reload...');
                // Try to restart visualization manually
                setTimeout(() => {
                    if (typeof window.AudioVisualization !== 'undefined') {
                        window.AudioVisualization.startAudioVisualization();
                    }
                }, 500);
            }
        });
    } else {
        addLog('WARNING', 'Audio device dropdown element not found');
    }
    
    if (elements.sampleRate) {
        addLog('DEBUG', 'Setting up sample rate change handler');
        elements.sampleRate.addEventListener('change', function() {
            const sampleRate = this.value;
            addLog('INFO', `Sample rate dropdown changed to: ${sampleRate}Hz`);
            
            // Check if AudioVisualization is available
            if (typeof window.AudioVisualization !== 'undefined' && window.AudioVisualization.handleSampleRateChange) {
                window.AudioVisualization.handleSampleRateChange();
            } else if (typeof AudioVisualization !== 'undefined' && AudioVisualization.handleSampleRateChange) {
                AudioVisualization.handleSampleRateChange();
            } else {
                addLog('ERROR', 'AudioVisualization module not available for sample rate change');
            }
        });
    } else {
        addLog('WARNING', 'Sample rate dropdown element not found');
    }
}

async function initializeApp() {
    addLog('INFO', 'Initializing Whisper NPU Frontend...');
    
    try {
        // Load settings from backend first
        await loadSettings();
        
        // Load available audio devices using AudioModule
        if (typeof AudioModule !== 'undefined' && AudioModule.loadAudioDevices) {
            await AudioModule.loadAudioDevices();
        } else {
            addLog('WARNING', 'AudioModule not found, audio devices may not be loaded');
        }
        
        // Check server health
        await checkServerHealth(false);
        
        // Load available models
        await loadModels();
        
        // Start audio visualization with the AudioVisualization module
        setTimeout(() => {
            if (typeof AudioVisualization !== 'undefined' && AudioVisualization.startAudioVisualization) {
                addLog('INFO', 'Starting audio visualization...');
                AudioVisualization.startAudioVisualization();
            } else {
                addLog('WARNING', 'AudioVisualization module not found');
            }
        }, 1000);
        
        addLog('SUCCESS', 'Frontend initialized successfully');
        
    } catch (error) {
        addLog('ERROR', `Initialization failed: ${error.message}`);
    }
}

// Settings management
async function loadSettings() {
    try {
        const response = await fetch(`${config.serverUrl}/settings`);
        if (response.ok) {
            const settings = await response.json();
            
            // Apply settings to UI
            if (settings.sample_rate && elements.sampleRate) {
                elements.sampleRate.value = settings.sample_rate;
            }
            
            if (settings.default_model && elements.selectedModel) {
                // Will be set when models are loaded
                state.preferredModel = settings.default_model;
            }
            
            addLog('INFO', `Settings loaded: ${Object.keys(settings).length} preferences`);
            return settings;
        }
    } catch (error) {
        addLog('WARNING', `Failed to load settings: ${error.message}`);
    }
    return {};
}

// Audio Test Functions
async function toggleAudioTest() {
    if (state.audioTest.isRecording) {
        stopAudioTest();
    } else {
        startAudioTest();
    }
}

// Audio Device and Sample Rate Change Handlers - REMOVED (moved to AudioVisualization module)

async function startAudioTest() {
    try {
        addLog('INFO', 'Starting audio test recording (5 seconds)...');
        
        // Use more compatible audio constraints for testing
        // Don't force sample rate for test - let browser use native rate
        const constraints = {
            audio: {
                deviceId: elements.audioDevice?.value ? { exact: elements.audioDevice.value } : undefined,
                // Remove sample rate constraint for test - use browser default
                echoCancellation: false,  // Better for hearing actual input
                noiseSuppression: false,  // Better for hearing actual input
                autoGainControl: false    // Prevent auto-adjustment during test
            }
        };
        
        state.audioTest.testStream = await navigator.mediaDevices.getUserMedia(constraints);
        
        // Log the actual track settings
        const audioTrack = state.audioTest.testStream.getAudioTracks()[0];
        const settings = audioTrack.getSettings();
        addLog('INFO', `Test audio settings: ${settings.sampleRate}Hz, ${settings.channelCount} channels`);
        
        // Create a temporary audio context to monitor volume during test
        const tempAudioContext = new (window.AudioContext || window.webkitAudioContext)();
        const tempAnalyser = tempAudioContext.createAnalyser();
        const tempMicrophone = tempAudioContext.createMediaStreamSource(state.audioTest.testStream);
        tempAnalyser.fftSize = 256;
        tempMicrophone.connect(tempAnalyser);
        
        // Monitor volume during recording
        const volumeMonitor = setInterval(() => {
            const bufferLength = tempAnalyser.frequencyBinCount;
            const dataArray = new Uint8Array(bufferLength);
            tempAnalyser.getByteFrequencyData(dataArray);
            
            let sum = 0;
            for (let i = 0; i < bufferLength; i++) {
                sum += dataArray[i] * dataArray[i];
            }
            const rms = Math.sqrt(sum / bufferLength);
            const level = (rms / 128) * 100;
            
            if (level > 1) {  // Only log if there's some audio
                addLog('INFO', `Test recording level: ${level.toFixed(1)}%`);
            }
        }, 500); // Reduced frequency for 5-second recording
        
        // Use a more compatible MIME type for testing
        let mimeType = 'audio/webm';
        if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) {
            mimeType = 'audio/webm;codecs=opus';
        } else if (MediaRecorder.isTypeSupported('audio/mp4')) {
            mimeType = 'audio/mp4';
        } else if (MediaRecorder.isTypeSupported('audio/ogg')) {
            mimeType = 'audio/ogg';
        }
        
        addLog('INFO', `Using MIME type for test: ${mimeType}`);
        
        // Create media recorder with compatible settings
        state.audioTest.testRecorder = new MediaRecorder(state.audioTest.testStream, {
            mimeType: mimeType
        });
        
        const testChunks = [];
        
        state.audioTest.testRecorder.ondataavailable = function(event) {
            if (event.data.size > 0) {
                testChunks.push(event.data);
                addLog('INFO', `Test chunk received: ${event.data.size} bytes`);
            }
        };
        
        state.audioTest.testRecorder.onstop = function() {
            clearInterval(volumeMonitor);
            tempAudioContext.close();
            
            state.audioTest.recordedBlob = new Blob(testChunks, { type: mimeType });
            state.audioTest.isRecording = false;
            updateAudioTestUI();
            addLog('SUCCESS', `Audio test recording completed: ${state.audioTest.recordedBlob.size} bytes`);
            
            // Enable play button
            if (elements.playTestButton) {
                elements.playTestButton.disabled = false;
            }
        };
        
        // Start recording
        state.audioTest.testRecorder.start();
        state.audioTest.isRecording = true;
        updateAudioTestUI();
        
        // Stop after 5 seconds
        setTimeout(() => {
            if (state.audioTest.testRecorder && state.audioTest.isRecording) {
                state.audioTest.testRecorder.stop();
                state.audioTest.testStream.getTracks().forEach(track => track.stop());
            }
        }, 5000);
        
    } catch (error) {
        addLog('ERROR', `Audio test failed: ${error.message}`);
        state.audioTest.isRecording = false;
        updateAudioTestUI();
    }
}

function stopAudioTest() {
    if (state.audioTest.testRecorder && state.audioTest.isRecording) {
        state.audioTest.testRecorder.stop();
        if (state.audioTest.testStream) {
            state.audioTest.testStream.getTracks().forEach(track => track.stop());
        }
    }
}

async function playTestAudio() {
    if (!state.audioTest.recordedBlob) {
        addLog('WARNING', 'No test audio to play. Record audio first.');
        return;
    }
    
    try {
        addLog('INFO', `Playing test audio: ${state.audioTest.recordedBlob.size} bytes, type: ${state.audioTest.recordedBlob.type}`);
        
        // Create audio URL and play
        const audioUrl = URL.createObjectURL(state.audioTest.recordedBlob);
        state.audioTest.testAudio = new Audio(audioUrl);
        
        // Set volume to 1.0 to ensure we can hear it
        state.audioTest.testAudio.volume = 1.0;
        
        state.audioTest.testAudio.onloadedmetadata = () => {
            addLog('INFO', `Test audio metadata: ${state.audioTest.testAudio.duration.toFixed(2)}s duration`);
        };
        
        state.audioTest.testAudio.onended = () => {
            state.audioTest.isPlaying = false;
            updateAudioTestUI();
            URL.revokeObjectURL(audioUrl);
            addLog('SUCCESS', 'Test audio playback completed');
        };
        
        state.audioTest.testAudio.onerror = (error) => {
            state.audioTest.isPlaying = false;
            updateAudioTestUI();
            URL.revokeObjectURL(audioUrl);
            addLog('ERROR', `Test audio playback failed: ${error.type} - ${error.message || 'Unknown audio error'}`);
        };
        
        state.audioTest.isPlaying = true;
        updateAudioTestUI();
        
        // Add a small delay to ensure UI updates
        await new Promise(resolve => setTimeout(resolve, 100));
        await state.audioTest.testAudio.play();
        
    } catch (error) {
        addLog('ERROR', `Failed to play test audio: ${error.message}`);
        state.audioTest.isPlaying = false;
        updateAudioTestUI();
    }
}

// Add new test transcription function
async function testTranscribeAudio() {
    if (!state.audioTest.recordedBlob) {
        addLog('WARNING', 'No test audio to transcribe. Record audio first.');
        return;
    }
    
    if (!state.selectedModel) {
        addLog('WARNING', 'Please select a model first');
        return;
    }
    
    try {
        addLog('INFO', `🧪 TESTING: Sending test audio to server for transcription...`);
        addLog('INFO', `🧪 Test audio: ${state.audioTest.recordedBlob.size} bytes, type: ${state.audioTest.recordedBlob.type}`);
        
        const url = `${config.serverUrl}/transcribe/${state.selectedModel}`;
        const response = await fetch(url, {
            method: 'POST',
            body: state.audioTest.recordedBlob
        });
        
        const data = await response.json();
        
        if (response.ok) {
            addLog('SUCCESS', `🧪 TEST TRANSCRIPTION RESULT: "${data.text}"`);
            addLog('INFO', `🧪 Processing time: ${data.processing_time}`);
            addLog('INFO', `🧪 Audio length: ${data.audio_length}`);
            addLog('INFO', `🧪 Device: ${data.device}`);
            addLog('INFO', `🧪 Language: ${data.language}`);
            
            if (data.text && data.text.trim()) {
                addTranscription(`[TEST] ${data.text}`, false);
            } else {
                addLog('WARNING', '🧪 Test returned no transcription - check if you actually spoke during recording');
            }
        } else {
            addLog('ERROR', `🧪 Test transcription failed: ${data.error}`);
        }
        
    } catch (error) {
        addLog('ERROR', `🧪 Test transcription error: ${error.message}`);
    }
}

function updateAudioTestUI() {
    const testButton = elements.testAudioButton;
    const playButton = elements.playTestButton;
    const transcribeButton = document.getElementById('testTranscribeButton');
    const status = elements.testStatus;
    
    if (testButton) {
        testButton.className = 'button test-audio';
        if (state.audioTest.isRecording) {
            testButton.textContent = 'Recording...';
            testButton.classList.add('recording');
            testButton.disabled = true;
        } else {
            testButton.textContent = 'Test Audio';
            testButton.disabled = false;
        }
    }
    
    if (playButton) {
        if (state.audioTest.isPlaying) {
            playButton.textContent = 'Playing...';
            playButton.classList.add('playing');
            playButton.disabled = true;
        } else {
            playButton.textContent = 'Play Test';
            playButton.classList.remove('playing');
            playButton.disabled = !state.audioTest.recordedBlob;
        }
    }
    
    if (transcribeButton) {
        transcribeButton.disabled = !state.audioTest.recordedBlob || !state.selectedModel;
        if (!state.selectedModel) {
            transcribeButton.title = 'Select a model first';
        } else if (!state.audioTest.recordedBlob) {
            transcribeButton.title = 'Record test audio first';
        } else {
            transcribeButton.title = 'Send test audio to server for transcription';
        }
    }
    
    if (status) {
        if (state.audioTest.isRecording) {
            status.textContent = '● Recording...';
            status.className = 'test-status recording';
        } else if (state.audioTest.isPlaying) {
            status.textContent = '▶ Playing...';
            status.className = 'test-status playing';
        } else if (state.audioTest.recordedBlob) {
            status.textContent = '✓ Ready to play';
            status.className = 'test-status ready';
        } else {
            status.textContent = 'No recording';
            status.className = 'test-status';
        }
    }
}

// Logging system
function addLog(level, message) {
    const timestamp = new Date().toLocaleTimeString();
    const logEntry = document.createElement('div');
    logEntry.className = `log-entry ${level}`;
    logEntry.innerHTML = `[${timestamp}] ${message}`;
    
    if (elements.logsContent) {
        elements.logsContent.appendChild(logEntry);
        elements.logsContent.scrollTop = elements.logsContent.scrollHeight;
        
        // Limit log entries
        while (elements.logsContent.children.length > 100) {
            elements.logsContent.removeChild(elements.logsContent.firstChild);
        }
    }
    
    // Also log to console for debugging
    console.log(`[${level}] ${message}`);
}

// Export the main functions for global access
window.WhisperApp = {
    selectModel: function(modelName) {
        state.selectedModel = modelName;
        if (elements.selectedModel) {
            elements.selectedModel.value = modelName;
        }
        updateModelSelection(modelName);
        addLog('INFO', `Selected model: ${modelName}`);
    },
    
    clearCache: async function() {
        try {
            const response = await fetch(`${config.serverUrl}/clear-cache`, { method: 'POST' });
            const data = await response.json();
            
            if (response.ok) {
                addLog('SUCCESS', 'NPU cache cleared successfully');
            } else {
                throw new Error(data.error || 'Failed to clear cache');
            }
        } catch (error) {
            addLog('ERROR', `Failed to clear cache: ${error.message}`);
        }
    },
    
    // Debug function to test device selection
    testDeviceSelection: function() {
        const deviceId = elements.audioDevice?.value;
        const selectedDeviceName = elements.audioDevice?.options[elements.audioDevice?.selectedIndex]?.text || 'Default';
        
        addLog('INFO', `🔍 DEVICE TEST: Current selection = "${selectedDeviceName}" (ID: ${deviceId || 'default'})`);
        addLog('INFO', `🔍 DEVICE TEST: Dropdown element exists = ${!!elements.audioDevice}`);
        addLog('INFO', `🔍 DEVICE TEST: window.AudioVisualization exists = ${typeof window.AudioVisualization !== 'undefined'}`);
        addLog('INFO', `🔍 DEVICE TEST: AudioVisualization exists = ${typeof AudioVisualization !== 'undefined'}`);
        addLog('INFO', `🔍 DEVICE TEST: Current visualization stream exists = ${!!state.visualizationStream}`);
        
        if (state.visualizationStream) {
            const tracks = state.visualizationStream.getAudioTracks();
            if (tracks.length > 0) {
                const settings = tracks[0].getSettings();
                addLog('INFO', `🔍 DEVICE TEST: Current stream device = "${settings.deviceId}" (${settings.sampleRate}Hz)`);
            }
        }
        
        // Force restart visualization
        addLog('INFO', '🔍 DEVICE TEST: Force restarting visualization...');
        if (typeof window.AudioVisualization !== 'undefined' && window.AudioVisualization.handleAudioDeviceChange) {
            window.AudioVisualization.handleAudioDeviceChange();
        } else if (typeof AudioVisualization !== 'undefined' && AudioVisualization.handleAudioDeviceChange) {
            AudioVisualization.handleAudioDeviceChange();
        } else {
            addLog('ERROR', '🔍 DEVICE TEST: No AudioVisualization module found!');
        }
    },
    
    // Debug function to test microphone access
    testMicrophone: async function() {
        try {
            const selectedDevice = elements.audioDevice?.value;
            const selectedDeviceName = elements.audioDevice?.options[elements.audioDevice?.selectedIndex]?.text;
            
            addLog('INFO', `Testing microphone: ${selectedDeviceName || 'Default'} (ID: ${selectedDevice || 'default'})`);
            
            const constraints = {
                audio: {
                    deviceId: selectedDevice ? { exact: selectedDevice } : undefined,
                    echoCancellation: false,
                    noiseSuppression: false
                }
            };
            
            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            const track = stream.getAudioTracks()[0];
            const settings = track.getSettings();
            
            addLog('SUCCESS', `Microphone test successful! Device: ${settings.deviceId}, Sample Rate: ${settings.sampleRate}Hz`);
            
            // Stop the test stream
            stream.getTracks().forEach(track => track.stop());
            
            return true;
        } catch (error) {
            addLog('ERROR', `Microphone test failed: ${error.message}`);
            return false;
        }
    },
    
    // Refresh device list
    refreshDevices: async function() {
        addLog('INFO', 'Refreshing audio device list...');
        if (typeof AudioModule !== 'undefined' && AudioModule.loadAudioDevices) {
            await AudioModule.loadAudioDevices();
        }
    }
};

// Make selectModel available globally for inline onclick handlers
window.selectModel = window.WhisperApp.selectModel;
window.testMicrophone = window.WhisperApp.testMicrophone;
window.testDeviceSelection = window.WhisperApp.testDeviceSelection;
window.refreshDevices = window.WhisperApp.refreshDevices;

// Remove the duplicate audio visualization functions since AudioVisualization module handles this
// Keep only the fallback logging
function startAudioVisualization() {
    addLog('INFO', 'Fallback startAudioVisualization called - using AudioVisualization module instead');
    if (typeof AudioVisualization !== 'undefined' && AudioVisualization.startAudioVisualization) {
        AudioVisualization.startAudioVisualization();
    }
} 