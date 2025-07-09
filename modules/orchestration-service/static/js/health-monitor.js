// Health monitoring and service status management for the orchestration frontend
class HealthMonitor {
    constructor() {
        this.services = new Map();
        this.autoRefresh = true;
        this.refreshInterval = 10000; // 10 seconds
        this.refreshTimer = null;
        this.orchestrationApiUrl = '/api';
        this.systemStartTime = Date.now();
        
        this.initializeElements();
        this.setupEventListeners();
        this.startAutoRefresh();
    }
    
    initializeElements() {
        this.elements = {
            healthContent: document.getElementById('healthContent'),
            refreshHealth: document.getElementById('refreshHealth'),
            autoRefreshToggle: document.getElementById('autoRefreshToggle'),
            connectionCount: document.getElementById('connectionCount'),
            activeConnections: document.getElementById('activeConnections'),
            totalConnections: document.getElementById('totalConnections'),
            peakConnections: document.getElementById('peakConnections'),
            systemUptime: document.getElementById('systemUptime'),
            messageRate: document.getElementById('messageRate'),
            connectionContent: document.getElementById('connectionContent'),
            connectionDetails: document.getElementById('connectionDetails'),
            connectionDetailsList: document.getElementById('connectionDetailsList'),
            gatewayContent: document.getElementById('gatewayContent'),
            gatewayMetrics: document.getElementById('gatewayMetrics'),
            requestRate: document.getElementById('requestRate'),
            avgResponseTime: document.getElementById('avgResponseTime'),
            successRate: document.getElementById('successRate'),
            circuitStatus: document.getElementById('circuitStatus')
        };
    }
    
    setupEventListeners() {
        if (this.elements.refreshHealth) {
            this.elements.refreshHealth.addEventListener('click', () => {
                this.refreshHealthStatus();
            });
        }
        
        if (this.elements.autoRefreshToggle) {
            this.elements.autoRefreshToggle.addEventListener('click', () => {
                this.toggleAutoRefresh();
            });
        }
        
        if (this.elements.connectionDetails) {
            this.elements.connectionDetails.addEventListener('click', () => {
                this.toggleConnectionDetails();
            });
        }
        
        if (this.elements.gatewayMetrics) {
            this.elements.gatewayMetrics.addEventListener('click', () => {
                this.showGatewayMetrics();
            });
        }
        
        // Update uptime every second
        setInterval(() => {
            this.updateUptime();
        }, 1000);
    }
    
    async refreshHealthStatus() {
        try {
            // Get orchestration service health
            const healthResponse = await fetch(`${this.orchestrationApiUrl}/health`);
            const healthData = await healthResponse.json();
            
            // Get backend services status
            const servicesResponse = await fetch(`${this.orchestrationApiUrl}/services`);
            const servicesData = await servicesResponse.json();
            
            // Get metrics and connection info
            const metricsResponse = await fetch(`${this.orchestrationApiUrl}/metrics`);
            const metricsData = await metricsResponse.json();
            
            this.updateHealthDisplay(healthData, servicesData, metricsData);
            this.updateConnectionStats(metricsData);
            this.updateGatewayStats(metricsData);
            
        } catch (error) {
            console.error('Failed to refresh health status:', error);
            this.showHealthError(error.message);
        }
    }
    
    updateHealthDisplay(healthData, servicesData, metricsData) {
        if (!this.elements.healthContent) return;
        
        // Create system health summary
        const systemHealth = this.determineSystemHealth(healthData, servicesData);
        
        let html = `
            <div class="system-health-summary ${systemHealth.status}">
                <div class="system-status-text">${systemHealth.text}</div>
                <div class="system-uptime">${this.formatUptime(metricsData.uptime || 0)}</div>
            </div>
        `;
        
        // Add orchestration service status
        html += this.createServiceHealthItem({
            name: 'Orchestration Service',
            url: window.location.origin,
            status: healthData.status || 'unknown',
            uptime: healthData.uptime || 0,
            components: healthData.components || {}
        });
        
        // Add backend services status
        if (servicesData && servicesData.services) {
            Object.entries(servicesData.services).forEach(([serviceName, serviceData]) => {
                html += this.createServiceHealthItem({
                    name: this.getServiceDisplayName(serviceName),
                    url: serviceData.url || 'Unknown',
                    status: serviceData.status || 'unknown',
                    uptime: serviceData.uptime_percentage || 0,
                    responseTime: serviceData.average_response_time || 0,
                    lastCheck: serviceData.last_check || 0
                });
            });
        }
        
        this.elements.healthContent.innerHTML = html;
        this.logActivity('Health status refreshed');
    }
    
    createServiceHealthItem(service) {
        const statusClass = this.getStatusClass(service.status);
        const responseTime = service.responseTime ? `${service.responseTime}ms` : 'N/A';
        const uptime = service.uptime ? `${service.uptime.toFixed(1)}%` : 'N/A';
        
        return `
            <div class="service-health-item">
                <div class="service-info">
                    <div class="service-name">${service.name}</div>
                    <div class="service-url">${service.url}</div>
                    <div class="service-metrics">
                        Response: ${responseTime} | Uptime: ${uptime}
                    </div>
                </div>
                <div class="service-status">
                    <div class="status-indicator ${statusClass}"></div>
                    <span class="status-text ${statusClass}">${this.getStatusText(service.status)}</span>
                </div>
            </div>
        `;
    }
    
    getServiceDisplayName(serviceName) {
        const names = {
            'whisper': 'Audio Service',
            'audio': 'Audio Service',
            'speaker': 'Speaker Service',
            'translation': 'Translation Service',
            'frontend': 'Frontend Service',
            'websocket': 'WebSocket Service',
            'monitoring': 'Monitoring Service'
        };
        return names[serviceName] || serviceName.charAt(0).toUpperCase() + serviceName.slice(1);
    }
    
    getStatusClass(status) {
        switch (status?.toLowerCase()) {
            case 'healthy':
            case 'running':
            case 'active':
                return 'healthy';
            case 'degraded':
            case 'warning':
                return 'degraded';
            case 'unhealthy':
            case 'error':
            case 'failed':
            case 'critical':
                return 'unhealthy';
            default:
                return 'unknown';
        }
    }
    
    getStatusText(status) {
        switch (status?.toLowerCase()) {
            case 'healthy':
                return 'Healthy';
            case 'degraded':
                return 'Degraded';
            case 'unhealthy':
                return 'Unhealthy';
            case 'running':
                return 'Running';
            case 'error':
                return 'Error';
            case 'unknown':
            default:
                return 'Unknown';
        }
    }
    
    determineSystemHealth(healthData, servicesData) {
        let healthyServices = 0;
        let degradedServices = 0;
        let unhealthyServices = 0;
        let totalServices = 1; // Include orchestration service
        
        // Check orchestration service
        if (healthData.status === 'healthy') {
            healthyServices++;
        } else {
            unhealthyServices++;
        }
        
        // Check backend services
        if (servicesData && servicesData.services) {
            Object.values(servicesData.services).forEach(service => {
                totalServices++;
                const status = this.getStatusClass(service.status);
                if (status === 'healthy') {
                    healthyServices++;
                } else if (status === 'degraded') {
                    degradedServices++;
                } else {
                    unhealthyServices++;
                }
            });
        }
        
        if (unhealthyServices > 0) {
            return {
                status: 'critical',
                text: `System Critical (${unhealthyServices}/${totalServices} services down)`
            };
        } else if (degradedServices > 0) {
            return {
                status: 'degraded',
                text: `System Degraded (${degradedServices}/${totalServices} services degraded)`
            };
        } else {
            return {
                status: 'healthy',
                text: `All Systems Operational (${healthyServices}/${totalServices} services healthy)`
            };
        }
    }
    
    updateConnectionStats(metricsData) {
        if (!metricsData) return;
        
        const websocketStats = metricsData.websocket_stats || {};
        const gatewayStats = metricsData.gateway_stats || {};
        
        const activeConnections = websocketStats.active_connections || 0;
        const totalConnections = websocketStats.total_connections || 0;
        const peakConnections = websocketStats.peak_connections || 0;
        const messageRate = websocketStats.messages_per_minute || 0;
        const uptime = metricsData.uptime || 0;
        
        if (this.elements.connectionCount) {
            this.elements.connectionCount.textContent = activeConnections;
        }
        
        if (this.elements.activeConnections) {
            this.elements.activeConnections.textContent = activeConnections;
        }
        
        if (this.elements.totalConnections) {
            this.elements.totalConnections.textContent = totalConnections;
        }
        
        if (this.elements.peakConnections) {
            this.elements.peakConnections.textContent = peakConnections;
        }
        
        if (this.elements.messageRate) {
            this.elements.messageRate.textContent = Math.round(messageRate);
        }
        
        if (this.elements.systemUptime) {
            this.elements.systemUptime.textContent = this.formatUptime(uptime);
        }
    }
    
    formatUptime(seconds) {
        if (!seconds || seconds < 0) return '00:00:00';
        
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);
        
        return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    
    updateUptime() {
        const uptime = (Date.now() - this.systemStartTime) / 1000;
        if (this.elements.systemUptime && !this.elements.systemUptime.textContent.includes(':')) {
            this.elements.systemUptime.textContent = this.formatUptime(uptime);
        }
    }
    
    showHealthError(error) {
        if (this.elements.healthContent) {
            this.elements.healthContent.innerHTML = `
                <div class="system-health-summary critical">
                    <div class="system-status-text">Health Check Failed</div>
                    <div class="system-uptime">Error: ${error}</div>
                </div>
                <div class="service-health-item">
                    <div class="service-info">
                        <div class="service-name">Orchestration Service</div>
                        <div class="service-url">Connection Error</div>
                    </div>
                    <div class="service-status">
                        <div class="status-indicator unhealthy"></div>
                        <span class="status-text unhealthy">Error</span>
                    </div>
                </div>
            `;
        }
    }
    
    toggleAutoRefresh() {
        this.autoRefresh = !this.autoRefresh;
        
        if (this.autoRefresh) {
            this.startAutoRefresh();
            if (this.elements.autoRefreshToggle) {
                this.elements.autoRefreshToggle.textContent = 'Auto: ON';
                this.elements.autoRefreshToggle.className = 'button small auto-on';
            }
            this.logActivity('Auto-refresh enabled');
        } else {
            this.stopAutoRefresh();
            if (this.elements.autoRefreshToggle) {
                this.elements.autoRefreshToggle.textContent = 'Auto: OFF';
                this.elements.autoRefreshToggle.className = 'button small auto-off';
            }
            this.logActivity('Auto-refresh disabled');
        }
    }
    
    startAutoRefresh() {
        if (this.refreshTimer) {
            clearInterval(this.refreshTimer);
        }
        
        if (this.autoRefresh) {
            // Initial refresh
            this.refreshHealthStatus();
            
            // Set up interval
            this.refreshTimer = setInterval(() => {
                if (this.autoRefresh) {
                    this.refreshHealthStatus();
                }
            }, this.refreshInterval);
        }
    }
    
    stopAutoRefresh() {
        if (this.refreshTimer) {
            clearInterval(this.refreshTimer);
            this.refreshTimer = null;
        }
    }
    
    logActivity(message) {
        // Integration with existing logging system
        if (window.WhisperApp && window.WhisperApp.logActivity) {
            window.WhisperApp.logActivity('HEALTH', message);
        } else {
            console.log('[HEALTH]', message);
        }
    }
    
    // Get health statistics
    getStatistics() {
        return {
            servicesMonitored: this.services.size,
            autoRefreshEnabled: this.autoRefresh,
            refreshInterval: this.refreshInterval,
            lastRefresh: new Date().toISOString()
        };
    }
    
    updateGatewayStats(metricsData) {
        if (!metricsData) return;
        
        const gatewayStats = metricsData.gateway_stats || {};
        
        const requestRate = gatewayStats.requests_per_minute || 0;
        const avgResponseTime = gatewayStats.avg_response_time || 0;
        const successRate = gatewayStats.success_rate || 1.0;
        const circuitStatus = gatewayStats.circuit_status || 'closed';
        
        if (this.elements.requestRate) {
            this.elements.requestRate.textContent = Math.round(requestRate);
        }
        
        if (this.elements.avgResponseTime) {
            this.elements.avgResponseTime.textContent = `${Math.round(avgResponseTime)}ms`;
        }
        
        if (this.elements.successRate) {
            this.elements.successRate.textContent = `${Math.round(successRate * 100)}%`;
        }
        
        if (this.elements.circuitStatus) {
            this.elements.circuitStatus.textContent = circuitStatus.charAt(0).toUpperCase() + circuitStatus.slice(1);
            this.elements.circuitStatus.className = `circuit-${circuitStatus.replace('_', '-')}`;
        }
    }
    
    async toggleConnectionDetails() {
        if (!this.elements.connectionDetailsList) return;
        
        const isVisible = this.elements.connectionDetailsList.style.display !== 'none';
        
        if (isVisible) {
            this.elements.connectionDetailsList.style.display = 'none';
            if (this.elements.connectionDetails) {
                this.elements.connectionDetails.textContent = 'Details';
            }
        } else {
            await this.loadConnectionDetails();
            this.elements.connectionDetailsList.style.display = 'block';
            if (this.elements.connectionDetails) {
                this.elements.connectionDetails.textContent = 'Hide';
            }
        }
    }
    
    async loadConnectionDetails() {
        if (!this.elements.connectionDetailsList) return;
        
        try {
            this.elements.connectionDetailsList.innerHTML = '<div class="connection-loading">Loading connection details...</div>';
            
            // Mock connection details (replace with actual API call)
            const connections = [
                { id: 'conn-001', duration: 120, ip: '192.168.1.100', userAgent: 'Chrome' },
                { id: 'conn-002', duration: 45, ip: '192.168.1.101', userAgent: 'Firefox' },
                { id: 'conn-003', duration: 300, ip: '192.168.1.102', userAgent: 'Safari' }
            ];
            
            let html = '';
            connections.forEach(conn => {
                html += `
                    <div class="connection-item">
                        <div class="connection-item-header">
                            <span class="connection-id">${conn.id}</span>
                            <span class="connection-duration">${this.formatUptime(conn.duration)}</span>
                        </div>
                        <div class="connection-meta">
                            <span>IP: ${conn.ip}</span>
                            <span>Agent: ${conn.userAgent}</span>
                        </div>
                    </div>
                `;
            });
            
            this.elements.connectionDetailsList.innerHTML = html || '<div class="connection-loading">No active connections</div>';
            
        } catch (error) {
            console.error('Failed to load connection details:', error);
            this.elements.connectionDetailsList.innerHTML = '<div class="connection-loading">Failed to load connection details</div>';
        }
    }
    
    async showGatewayMetrics() {
        this.logActivity('Gateway metrics requested');
        // In a real implementation, this could show a detailed metrics modal
        // For now, just refresh the gateway stats
        await this.refreshHealthStatus();
    }

    // Cleanup
    destroy() {
        this.stopAutoRefresh();
    }
}

// Initialize health monitor when DOM is loaded
let healthMonitor;
document.addEventListener('DOMContentLoaded', function() {
    healthMonitor = new HealthMonitor();
    
    // Make it globally accessible
    window.HealthMonitor = healthMonitor;
});

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = HealthMonitor;
}