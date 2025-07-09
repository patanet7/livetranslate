/**
 * Speaker Diarization Module
 * Handles speaker identification, configuration, and display
 */

class SpeakerDiarization {
    constructor() {
        this.enabled = false;
        this.numSpeakers = null; // Auto-detect by default
        this.bufferDuration = 6.0;
        this.inferenceInterval = 3.0;
        this.speakerHistory = [];
        this.speakerColors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', 
            '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F',
            '#BB8FCE', '#85C1E9'
        ];
        this.lastUpdateTime = 0;
        
        this.initializeUI();
        this.bindEvents();
        this.checkAvailability();
    }
    
    initializeUI() {
        // Create speaker diarization panel
        const speakerPanel = this.createSpeakerPanel();
        
        // Insert into the right panel (with other controls)
        const rightPanel = document.querySelector('.right-panel');
        if (rightPanel) {
            // Insert after models panel but before logs panel
            const logsPanel = rightPanel.querySelector('.logs-panel');
            if (logsPanel) {
                rightPanel.insertBefore(speakerPanel, logsPanel);
            } else {
                rightPanel.appendChild(speakerPanel);
            }
        } else {
            // Fallback: insert into main container
            const mainContainer = document.querySelector('.main-container');
            if (mainContainer) {
                mainContainer.appendChild(speakerPanel);
            } else {
                document.body.appendChild(speakerPanel);
            }
        }
        
        // Create speaker history display and insert into left panel
        const historyPanel = this.createSpeakerHistoryPanel();
        
        const leftPanel = document.querySelector('.left-panel');
        if (leftPanel) {
            // Insert after transcription panel
            leftPanel.appendChild(historyPanel);
        } else {
            // Fallback: insert into main container
            const mainContainer = document.querySelector('.main-container');
            if (mainContainer) {
                mainContainer.appendChild(historyPanel);
            } else {
                document.body.appendChild(historyPanel);
            }
        }
        
        console.log('✓ Speaker diarization UI initialized');
    }
    
    createSpeakerPanel() {
        const panel = document.createElement('div');
        panel.className = 'panel speaker-panel';
        panel.innerHTML = `
            <div class="panel-header collapsible-header" data-target="speaker-diarization-content">
                <span>🎤 Speaker Diarization</span>
                <div style="display: flex; align-items: center; gap: 1rem;">
                    <div class="status-indicator" id="speaker-status">
                        <span class="status-dot disconnected"></span>
                        <span class="status-text">Checking...</span>
                    </div>
                    <span class="collapse-indicator">▼</span>
                </div>
            </div>
            
            <div class="panel-content collapsible-content" id="speaker-diarization-content">
                <div class="form-group">
                    <label>
                        <input type="checkbox" id="enable-diarization"> 
                        Enable Speaker Identification
                    </label>
                    <small>Identify different speakers in real-time</small>
                </div>
                
                <div class="diarization-settings" id="diarization-settings" style="display: none;">
                    <div class="form-group">
                        <label for="num-speakers">Number of Speakers:</label>
                        <select id="num-speakers">
                            <option value="">Auto-detect</option>
                            <option value="2">2 speakers</option>
                            <option value="3">3 speakers</option>
                            <option value="4">4 speakers</option>
                            <option value="5">5 speakers</option>
                            <option value="6">6 speakers</option>
                            <option value="7">7 speakers</option>
                            <option value="8">8 speakers</option>
                        </select>
                        <small>Leave on auto-detect for unknown speaker count</small>
                    </div>
                    
                    <div class="form-group">
                        <label for="buffer-duration">Window Duration:</label>
                        <select id="buffer-duration">
                            <option value="3">3 seconds</option>
                            <option value="6" selected>6 seconds</option>
                            <option value="9">9 seconds</option>
                            <option value="12">12 seconds</option>
                        </select>
                        <small>Longer windows = better accuracy, higher latency</small>
                    </div>
                    
                    <div class="form-group">
                        <label for="inference-interval">Update Interval:</label>
                        <select id="inference-interval">
                            <option value="1">1 second</option>
                            <option value="2">2 seconds</option>
                            <option value="3" selected>3 seconds</option>
                            <option value="5">5 seconds</option>
                        </select>
                        <small>How often to process new audio</small>
                    </div>
                    
                    <div class="form-group">
                        <button id="apply-diarization" class="button">
                            Apply Settings
                        </button>
                        <button id="clear-speaker-history" class="button">
                            Clear History
                        </button>
                    </div>
                </div>
                
                <div class="speaker-stats" id="speaker-stats" style="display: none;">
                    <h4>Speaker Statistics</h4>
                    <div class="stats-grid">
                        <div class="stat-item">
                            <span class="stat-label">Total Speakers:</span>
                            <span class="stat-value" id="total-speakers">0</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Active Speakers:</span>
                            <span class="stat-value" id="active-speakers">0</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Segments Processed:</span>
                            <span class="stat-value" id="total-segments">0</span>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        return panel;
    }
    
    createSpeakerHistoryPanel() {
        const panel = document.createElement('div');
        panel.className = 'panel speaker-history-panel';
        panel.innerHTML = `
            <div class="panel-header collapsible-header" data-target="speaker-history-content">
                <span>🗣️ Speaker History</span>
                <div style="display: flex; align-items: center; gap: 1rem;">
                    <div class="history-controls">
                        <button id="export-speaker-history" class="button small">Export</button>
                        <button id="refresh-speaker-history" class="button small">Refresh</button>
                    </div>
                    <span class="collapse-indicator">▼</span>
                </div>
            </div>
            
            <div class="panel-content collapsible-content" id="speaker-history-content">
                <div class="speaker-legend" id="speaker-legend" style="display: none;">
                    <h4>Speaker Legend</h4>
                    <div class="legend-items" id="legend-items"></div>
                </div>
                
                <div class="speaker-timeline" id="speaker-timeline">
                    <p class="placeholder">Enable speaker diarization to see speaker history</p>
                </div>
            </div>
        `;
        
        return panel;
    }
    
    bindEvents() {
        // Add collapsible functionality to all headers
        this.initializeCollapsiblePanels();
        
        // Enable/disable diarization
        document.getElementById('enable-diarization').addEventListener('change', (e) => {
            const settings = document.getElementById('diarization-settings');
            const stats = document.getElementById('speaker-stats');
            
            if (e.target.checked) {
                settings.style.display = 'block';
                stats.style.display = 'block';
            } else {
                settings.style.display = 'none';
                stats.style.display = 'none';
                this.disableDiarization();
            }
        });
        
        // Apply settings
        document.getElementById('apply-diarization').addEventListener('click', () => {
            this.applySettings();
        });
        
        // Clear history
        document.getElementById('clear-speaker-history').addEventListener('click', () => {
            this.clearHistory();
        });
        
        // Export history
        document.getElementById('export-speaker-history').addEventListener('click', () => {
            this.exportHistory();
        });
        
        // Refresh history
        document.getElementById('refresh-speaker-history').addEventListener('click', () => {
            this.refreshHistory();
        });
        
        // Auto-refresh speaker history
        setInterval(() => {
            if (this.enabled) {
                this.updateSpeakerHistory();
                this.updateStatistics();
            }
        }, 2000);
    }
    
    async checkAvailability() {
        try {
            const response = await fetch(`http://localhost:5000/diarization/status`);
            const status = await response.json();
            
            const statusElement = document.getElementById('speaker-status');
            const enableCheckbox = document.getElementById('enable-diarization');
            
            if (status.available) {
                statusElement.innerHTML = `
                    <span class="status-dot connected"></span>
                    <span class="status-text">Available</span>
                `;
                enableCheckbox.disabled = false;
                
                if (status.enabled) {
                    enableCheckbox.checked = true;
                    document.getElementById('diarization-settings').style.display = 'block';
                    document.getElementById('speaker-stats').style.display = 'block';
                    this.enabled = true;
                }
            } else {
                statusElement.innerHTML = `
                    <span class="status-dot disconnected"></span>
                    <span class="status-text">Not Available</span>
                `;
                enableCheckbox.disabled = true;
                
                // Show info about missing dependencies
                const panel = document.querySelector('.speaker-panel .panel-content');
                const infoDiv = document.createElement('div');
                infoDiv.className = 'info-message';
                infoDiv.innerHTML = `
                    <p><strong>Speaker diarization not available.</strong></p>
                    <p>Install dependencies: <code>pip install resemblyzer torch sklearn</code></p>
                `;
                panel.appendChild(infoDiv);
            }
            
        } catch (error) {
            console.error('Failed to check diarization availability:', error);
            document.getElementById('speaker-status').innerHTML = `
                <span class="status-dot disconnected"></span>
                <span class="status-text">Error</span>
            `;
        }
    }
    
    async applySettings() {
        const enabled = document.getElementById('enable-diarization').checked;
        const numSpeakers = document.getElementById('num-speakers').value;
        const bufferDuration = parseFloat(document.getElementById('buffer-duration').value);
        const inferenceInterval = parseFloat(document.getElementById('inference-interval').value);
        
        const config = {
            enable: enabled,
            n_speakers: numSpeakers ? parseInt(numSpeakers) : null,
            buffer_duration: bufferDuration,
            inference_interval: inferenceInterval
        };
        
        try {
            const response = await fetch('http://localhost:5000/diarization/configure', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(config)
            });
            
            const result = await response.json();
            
            if (response.ok) {
                this.enabled = enabled;
                this.numSpeakers = config.n_speakers;
                this.bufferDuration = bufferDuration;
                this.inferenceInterval = inferenceInterval;
                
                console.log('✓ Speaker diarization configured:', result);
                this.showMessage('Settings applied successfully!', 'success');
                
                if (enabled) {
                    this.startDiarization();
                }
            } else {
                console.error('❌ Failed to configure diarization:', result.error);
                this.showMessage(`Configuration failed: ${result.error}`, 'error');
            }
            
        } catch (error) {
            console.error('❌ Configuration request failed:', error);
            this.showMessage('Network error during configuration', 'error');
        }
    }
    
    async disableDiarization() {
        try {
            const response = await fetch('http://localhost:5000/diarization/configure', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ enable: false })
            });
            
            if (response.ok) {
                this.enabled = false;
                console.log('✓ Speaker diarization disabled');
                this.clearSpeakerDisplay();
            }
            
        } catch (error) {
            console.error('❌ Failed to disable diarization:', error);
        }
    }
    
    startDiarization() {
        console.log('🎤 Starting speaker diarization...');
        this.clearSpeakerDisplay();
        this.updateSpeakerHistory();
        
        // Show speaker legend
        document.getElementById('speaker-legend').style.display = 'block';
        this.updateSpeakerLegend();
    }
    
    async updateSpeakerHistory() {
        if (!this.enabled) return;
        
        try {
            const response = await fetch(`http://localhost:5000/speaker/history?since=${this.lastUpdateTime}&limit=20`);
            const data = await response.json();
            
            if (response.ok && data.transcriptions.length > 0) {
                this.speakerHistory = [...this.speakerHistory, ...data.transcriptions];
                
                // Keep only recent history (last 100 items)
                if (this.speakerHistory.length > 100) {
                    this.speakerHistory = this.speakerHistory.slice(-100);
                }
                
                this.displaySpeakerHistory();
                this.updateSpeakerLegend();
                this.lastUpdateTime = data.server_time;
            }
            
        } catch (error) {
            console.error('❌ Failed to update speaker history:', error);
        }
    }
    
    displaySpeakerHistory() {
        const timeline = document.getElementById('speaker-timeline');
        
        // Clear placeholder
        if (timeline.querySelector('.placeholder')) {
            timeline.innerHTML = '';
        }
        
        // Display recent transcriptions with speaker info
        const recentItems = this.speakerHistory.slice(-10); // Show last 10 items
        
        recentItems.forEach(item => {
            const existingItem = timeline.querySelector(`[data-timestamp="${item.timestamp}"]`);
            if (existingItem) return; // Skip if already displayed
            
            const timelineItem = document.createElement('div');
            timelineItem.className = 'timeline-item';
            timelineItem.setAttribute('data-timestamp', item.timestamp);
            
            const timestamp = new Date(item.timestamp * 1000).toLocaleTimeString();
            const duration = item.buffer_duration ? `${item.buffer_duration.toFixed(1)}s` : '';
            
            let speakerInfo = '';
            if (item.speakers && item.speakers.length > 0) {
                const speakers = item.speakers.map(s => {
                    const color = this.getSpeakerColor(s.speaker_id);
                    const confidence = (s.confidence * 100).toFixed(0);
                    // Convert speaker_id to user-friendly display (0 -> Speaker 1, 1 -> Speaker 2, etc.)
                    const displaySpeakerId = s.speaker_id !== undefined && s.speaker_id !== null ? 
                        (Number(s.speaker_id) + 1) : 'Unknown';
                    const speakerLabel = displaySpeakerId !== 'Unknown' ? 
                        `Speaker ${displaySpeakerId}` : 'Unknown Speaker';
                    return `<span class="speaker-tag" style="background-color: ${color}">
                        ${speakerLabel} (${confidence}%)
                    </span>`;
                }).join(' ');
                speakerInfo = `<div class="speakers">${speakers}</div>`;
            } else {
                // If no speakers detected, still show a numbered speaker based on pattern
                speakerInfo = '<div class="speakers"><span class="speaker-tag unknown">Speaker 1</span></div>';
            }
            
            // Handle Chinese text and undefined display
            let displayText = item.text || '';
            if (displayText === 'undefined' || !displayText) {
                displayText = '[No transcription available]';
            }
            // Ensure proper UTF-8 encoding for Chinese characters
            if (displayText && typeof displayText === 'string') {
                displayText = displayText.trim();
            }
            
            timelineItem.innerHTML = `
                <div class="timeline-header">
                    <span class="timestamp">${timestamp}</span>
                    <span class="duration">${duration}</span>
                    ${item.enhanced ? '<span class="enhanced-badge">Enhanced</span>' : ''}
                </div>
                ${speakerInfo}
                <div class="transcription">${displayText}</div>
            `;
            
            timeline.appendChild(timelineItem);
        });
        
        // Auto-scroll to bottom
        timeline.scrollTop = timeline.scrollHeight;
    }
    
    updateSpeakerLegend() {
        const legendItems = document.getElementById('legend-items');
        const speakers = new Set();
        
        // Collect all speakers from history
        this.speakerHistory.forEach(item => {
            if (item.speakers) {
                item.speakers.forEach(s => speakers.add(s.speaker_id));
            }
        });
        
        if (speakers.size === 0) return;
        
        legendItems.innerHTML = '';
        
        Array.from(speakers).sort((a, b) => a - b).forEach(speakerId => {
            const color = this.getSpeakerColor(speakerId);
            const displaySpeakerId = speakerId !== undefined && speakerId !== null ? 
                (Number(speakerId) + 1) : 'Unknown';
            const legendItem = document.createElement('div');
            legendItem.className = 'legend-item';
            legendItem.innerHTML = `
                <span class="legend-color" style="background-color: ${color}"></span>
                <span class="legend-label">Speaker ${displaySpeakerId}</span>
            `;
            legendItems.appendChild(legendItem);
        });
    }
    
    async updateStatistics() {
        if (!this.enabled) return;
        
        try {
            const response = await fetch('http://localhost:5000/diarization/status');
            const data = await response.json();
            
            if (response.ok && data.statistics) {
                const stats = data.statistics;
                
                document.getElementById('total-speakers').textContent = stats.total_speakers || 0;
                document.getElementById('active-speakers').textContent = stats.active_speakers || 0;
                document.getElementById('total-segments').textContent = stats.total_segments || 0;
            }
            
        } catch (error) {
            console.error('❌ Failed to update statistics:', error);
        }
    }
    
    getSpeakerColor(speakerId) {
        const colorIndex = speakerId % this.speakerColors.length;
        return this.speakerColors[colorIndex];
    }
    
    clearHistory() {
        this.speakerHistory = [];
        this.lastUpdateTime = 0;
        this.clearSpeakerDisplay();
        console.log('✓ Speaker history cleared');
    }
    
    clearSpeakerDisplay() {
        const timeline = document.getElementById('speaker-timeline');
        timeline.innerHTML = '<p class="placeholder">Speaker history will appear here...</p>';
        
        const legendItems = document.getElementById('legend-items');
        legendItems.innerHTML = '';
        
        document.getElementById('speaker-legend').style.display = 'none';
    }
    
    async refreshHistory() {
        this.lastUpdateTime = 0; // Reset to get all history
        await this.updateSpeakerHistory();
        this.showMessage('History refreshed', 'success');
    }
    
    exportHistory() {
        if (this.speakerHistory.length === 0) {
            this.showMessage('No speaker history to export', 'warning');
            return;
        }
        
        const exportData = {
            timestamp: new Date().toISOString(),
            speaker_history: this.speakerHistory,
            settings: {
                enabled: this.enabled,
                num_speakers: this.numSpeakers,
                buffer_duration: this.bufferDuration,
                inference_interval: this.inferenceInterval
            }
        };
        
        const blob = new Blob([JSON.stringify(exportData, null, 2)], {
            type: 'application/json'
        });
        
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `speaker-history-${new Date().toISOString().slice(0, 19)}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        this.showMessage('History exported successfully', 'success');
    }
    
    showMessage(message, type = 'info') {
        // Create or update message display
        let messageElement = document.getElementById('speaker-message');
        if (!messageElement) {
            messageElement = document.createElement('div');
            messageElement.id = 'speaker-message';
            messageElement.className = 'message';
            document.querySelector('.speaker-panel .panel-content').appendChild(messageElement);
        }
        
        messageElement.className = `message ${type}`;
        messageElement.textContent = message;
        messageElement.style.display = 'block';
        
        // Auto-hide after 3 seconds
        setTimeout(() => {
            messageElement.style.display = 'none';
        }, 3000);
    }
    
    // Enhanced transcription method for single audio files
    async enhancedTranscribe(audioBlob, model = 'whisper-base') {
        if (!this.enabled) {
            console.log('Speaker diarization not enabled, using regular transcription');
            return null;
        }
        
        try {
            const formData = new FormData();
            formData.append('audio', audioBlob);
            
            const params = new URLSearchParams({
                enhance: 'true',
                diarize: 'true'
            });
            
            if (this.numSpeakers) {
                params.append('speakers', this.numSpeakers);
            }
            
            const response = await fetch(`http://localhost:5000/transcribe/enhanced/${model}?${params}`, {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (response.ok) {
                console.log('✓ Enhanced transcription with speakers:', result);
                return result;
            } else {
                console.error('❌ Enhanced transcription failed:', result.error);
                return null;
            }
            
        } catch (error) {
            console.error('❌ Enhanced transcription request failed:', error);
            return null;
        }
    }
    
    initializeCollapsiblePanels() {
        // Add click handlers to all collapsible headers
        document.addEventListener('click', (e) => {
            const header = e.target.closest('.collapsible-header');
            if (header && (header.closest('.speaker-panel') || header.closest('.speaker-history-panel'))) {
                // Prevent event bubbling to avoid triggering when clicking buttons
                if (e.target.closest('button') || e.target.closest('.status-indicator')) {
                    return;
                }
                
                const targetId = header.getAttribute('data-target');
                const content = document.getElementById(targetId);
                const indicator = header.querySelector('.collapse-indicator');
                
                if (content && indicator) {
                    const isCollapsed = content.style.display === 'none';
                    
                    if (isCollapsed) {
                        content.style.display = 'block';
                        indicator.textContent = '▼';
                        indicator.style.transform = 'rotate(0deg)';
                    } else {
                        content.style.display = 'none';
                        indicator.textContent = '▶';
                        indicator.style.transform = 'rotate(-90deg)';
                    }
                    
                    // Add animation class
                    content.classList.add('collapsing');
                    setTimeout(() => {
                        content.classList.remove('collapsing');
                    }, 300);
                }
            }
        });
    }
}

// Initialize speaker diarization when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.speakerDiarization = new SpeakerDiarization();
    console.log('✓ Speaker diarization module loaded');
});

// Export for use in other modules
window.SpeakerDiarization = SpeakerDiarization; 