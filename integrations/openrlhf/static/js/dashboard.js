// OpenRLHF Dashboard JavaScript

class OpenRLHFDashboard {
    constructor() {
        this.refreshInterval = 2000; // 2 seconds
        this.isRefreshing = false;
        this.charts = {};
        this.lastUpdateTime = null;
        
        this.init();
    }
    
    init() {
        console.log('🚀 Initializing OpenRLHF Dashboard...');
        
        // Load initial data
        this.loadAllData();
        
        // Set up auto-refresh
        this.setupAutoRefresh();
        
        // Set up event listeners
        this.setupEventListeners();
        
        console.log('✅ Dashboard initialized successfully');
    }
    
    setupAutoRefresh() {
        setInterval(() => {
            if (!this.isRefreshing) {
                this.refreshData();
            }
        }, this.refreshInterval);
    }
    
    setupEventListeners() {
        // Add any custom event listeners here
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                console.log('Dashboard hidden, pausing auto-refresh');
            } else {
                console.log('Dashboard visible, resuming auto-refresh');
                this.refreshData();
            }
        });
    }
    
    async loadAllData() {
        try {
            this.isRefreshing = true;
            
            // Load data in parallel
            const [summary, metrics, health, resources, alerts] = await Promise.all([
                this.fetchData('/api/summary'),
                this.fetchData('/api/metrics'),
                this.fetchData('/api/health'),
                this.fetchData('/api/resources'),
                this.fetchData('/api/alerts')
            ]);
            
            // Update UI with loaded data
            this.updateSummary(summary);
            this.updateMetricsTable(metrics);
            this.updateAlerts(alerts);
            
            // Update charts
            await this.updateCharts();
            
            // Update timestamp
            this.updateTimestamp();
            
        } catch (error) {
            console.error('Error loading dashboard data:', error);
            this.showError('Failed to load dashboard data');
        } finally {
            this.isRefreshing = false;
        }
    }
    
    async refreshData() {
        if (this.isRefreshing) return;
        
        try {
            this.isRefreshing = true;
            
            // Load summary and alerts (lightweight)
            const [summary, alerts] = await Promise.all([
                this.fetchData('/api/summary'),
                this.fetchData('/api/alerts')
            ]);
            
            this.updateSummary(summary);
            this.updateAlerts(alerts);
            this.updateTimestamp();
            
        } catch (error) {
            console.error('Error refreshing data:', error);
        } finally {
            this.isRefreshing = false;
        }
    }
    
    async fetchData(endpoint) {
        try {
            const response = await fetch(endpoint);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            return await response.json();
        } catch (error) {
            console.error(`Error fetching ${endpoint}:`, error);
            throw error;
        }
    }
    
    updateSummary(summary) {
        if (!summary) return;
        
        // Update summary cards
        this.updateElement('current-step', summary.latest_step || '-');
        this.updateElement('current-loss', this.formatNumber(summary.latest_loss, 4));
        this.updateElement('current-reward', this.formatNumber(summary.latest_reward, 3));
        this.updateElement('health-score', this.formatNumber(summary.overall_health, 2));
        
        // Add status indicators
        this.addStatusIndicator('current-step', summary.latest_step > 0 ? 'good' : 'neutral');
        this.addStatusIndicator('current-loss', this.getLossStatus(summary.latest_loss));
        this.addStatusIndicator('current-reward', this.getRewardStatus(summary.latest_reward));
        this.addStatusIndicator('health-score', this.getHealthStatus(summary.overall_health));
    }
    
    updateMetricsTable(metrics) {
        if (!metrics || !Array.isArray(metrics)) return;
        
        const tbody = document.getElementById('metrics-table');
        if (!tbody) return;
        
        // Show only last 10 metrics
        const recentMetrics = metrics.slice(-10).reverse();
        
        if (recentMetrics.length === 0) {
            tbody.innerHTML = '<tr><td colspan="8" class="text-center">No metrics available</td></tr>';
            return;
        }
        
        tbody.innerHTML = recentMetrics.map(metric => `
            <tr class="fade-in">
                <td>${metric.step || '-'}</td>
                <td class="metric-value ${this.getLossClass(metric.loss)}">${this.formatNumber(metric.loss, 4)}</td>
                <td class="metric-value ${this.getRewardClass(metric.reward_mean)}">${this.formatNumber(metric.reward_mean, 3)}</td>
                <td class="metric-value ${this.getKLClass(metric.kl_mean)}">${this.formatNumber(metric.kl_mean, 4)}</td>
                <td>${this.formatNumber(metric.learning_rate, 6)}</td>
                <td>${this.formatNumber(metric.gpu_memory_used, 2)} GB</td>
                <td>${this.formatNumber(metric.step_time, 3)}s</td>
                <td>${this.formatTimestamp(metric.timestamp)}</td>
            </tr>
        `).join('');
    }
    
    updateAlerts(alerts) {
        const container = document.getElementById('alerts-container');
        const alertsList = document.getElementById('alerts-list');
        
        if (!container || !alertsList) return;
        
        if (!alerts || alerts.length === 0) {
            container.style.display = 'none';
            return;
        }
        
        container.style.display = 'block';
        alertsList.innerHTML = alerts.map(alert => `
            <div class="alert-item mb-2">
                <strong>${alert.type.toUpperCase()}:</strong> ${alert.message}
                <small class="text-muted ms-2">${this.formatTimestamp(alert.timestamp)}</small>
            </div>
        `).join('');
    }
    
    async updateCharts() {
        try {
            // Update charts in parallel
            await Promise.all([
                this.updateChart('loss-plot', 'loss'),
                this.updateChart('reward-plot', 'reward'),
                this.updateChart('kl-plot', 'kl'),
                this.updateChart('resources-plot', 'resources'),
                this.updateChart('health-plot', 'health')
            ]);
        } catch (error) {
            console.error('Error updating charts:', error);
        }
    }
    
    async updateChart(containerId, plotType) {
        try {
            const plotData = await this.fetchData(`/api/plots/${plotType}`);
            
            if (plotData.error) {
                console.warn(`Chart ${plotType} error:`, plotData.error);
                return;
            }
            
            const plotConfig = JSON.parse(plotData);
            Plotly.react(containerId, plotConfig.data, plotConfig.layout, {responsive: true});
            
        } catch (error) {
            console.error(`Error updating chart ${plotType}:`, error);
        }
    }
    
    updateTimestamp() {
        const now = new Date();
        const timeString = now.toLocaleTimeString();
        this.updateElement('update-time', timeString);
        this.lastUpdateTime = now;
    }
    
    // Utility methods
    updateElement(id, value) {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = value;
        }
    }
    
    formatNumber(value, decimals = 2) {
        if (value === null || value === undefined || isNaN(value)) {
            return '-';
        }
        return Number(value).toFixed(decimals);
    }
    
    formatTimestamp(timestamp) {
        if (!timestamp) return '-';
        const date = new Date(timestamp * 1000);
        return date.toLocaleString();
    }
    
    addStatusIndicator(elementId, status) {
        const element = document.getElementById(elementId);
        if (!element) return;
        
        // Remove existing status indicator
        const existing = element.parentElement.querySelector('.status-indicator');
        if (existing) {
            existing.remove();
        }
        
        // Add new status indicator
        const indicator = document.createElement('span');
        indicator.className = `status-indicator ${status}`;
        element.parentElement.insertBefore(indicator, element);
    }
    
    getLossStatus(loss) {
        if (loss === null || loss === undefined) return 'neutral';
        if (loss < 0.1) return 'good';
        if (loss < 1.0) return 'warning';
        return 'danger';
    }
    
    getRewardStatus(reward) {
        if (reward === null || reward === undefined) return 'neutral';
        if (reward > 0.5) return 'good';
        if (reward > 0) return 'warning';
        return 'danger';
    }
    
    getHealthStatus(health) {
        if (health === null || health === undefined) return 'neutral';
        if (health > 0.7) return 'good';
        if (health > 0.4) return 'warning';
        return 'danger';
    }
    
    getLossClass(loss) {
        const status = this.getLossStatus(loss);
        return status === 'good' ? 'positive' : status === 'danger' ? 'negative' : 'neutral';
    }
    
    getRewardClass(reward) {
        const status = this.getRewardStatus(reward);
        return status === 'good' ? 'positive' : status === 'danger' ? 'negative' : 'neutral';
    }
    
    getKLClass(kl) {
        if (kl === null || kl === undefined) return 'neutral';
        if (kl < 0.1) return 'positive';
        if (kl < 1.0) return 'neutral';
        return 'negative';
    }
    
    showError(message) {
        // Create a toast notification or alert
        const alertDiv = document.createElement('div');
        alertDiv.className = 'alert alert-danger alert-dismissible fade show position-fixed';
        alertDiv.style.top = '20px';
        alertDiv.style.right = '20px';
        alertDiv.style.zIndex = '9999';
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        document.body.appendChild(alertDiv);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentElement) {
                alertDiv.remove();
            }
        }, 5000);
    }
}

// Global functions for button clicks
function refreshData() {
    if (window.dashboard) {
        window.dashboard.loadAllData();
    }
}

function exportData(dataType) {
    const url = `/api/export/${dataType}`;
    const link = document.createElement('a');
    link.href = url;
    link.download = `${dataType}_export.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new OpenRLHFDashboard();
});

// Handle page visibility changes
document.addEventListener('visibilitychange', () => {
    if (window.dashboard) {
        if (document.hidden) {
            console.log('Page hidden, pausing dashboard updates');
        } else {
            console.log('Page visible, resuming dashboard updates');
            window.dashboard.refreshData();
        }
    }
});

// Handle window resize for responsive charts
window.addEventListener('resize', () => {
    if (window.dashboard && window.dashboard.charts) {
        Object.values(window.dashboard.charts).forEach(chart => {
            if (chart && typeof chart.resize === 'function') {
                chart.resize();
            }
        });
    }
});

// Export functions for global access
window.refreshData = refreshData;
window.exportData = exportData;