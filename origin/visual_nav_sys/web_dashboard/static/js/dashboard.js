// Visual Navigation System - Dashboard JavaScript

// Global state
const state = {
    socket: null,
    connected: false,
    hasReference: false,
    currentData: {},
    history: {
        fps: [],
        matches: [],
        confidence: [],
        velocity: [],
        timestamps: []
    },
    maxHistoryLength: 100
};

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    initSocket();
    initTabs();
    initCharts();
    initGauges();
    startVideoStream();
});

// Socket.IO Connection - FORCE WEBSOCKET for low latency
function initSocket() {
    // Force WebSocket transport (polling is slow!)
    state.socket = io({
        transports: ['websocket'],
        upgrade: false,
        reconnection: true,
        reconnectionDelay: 100
    });
    
    state.socket.on('connect', () => {
        console.log('Connected via WebSocket - LOW LATENCY MODE');
        state.connected = true;
        updateConnectionStatus('Connected (WS)', 'success');
    });
    
    state.socket.on('disconnect', () => {
        console.log('Disconnected from server');
        state.connected = false;
        updateConnectionStatus('Disconnected', 'error');
    });
    
    state.socket.on('status', (data) => {
        console.log('Status:', data.message);
    });
    
    state.socket.on('frame_data', (data) => {
        updateDashboard(data);
    });

    // =================================================================
    // PUB/SUB PATTERN: Subscribe to data channels
    // =================================================================
    
    console.log('[SUB] Subscribing to channels: telemetry, health, frame_data');
    
    // SUBSCRIBE: Telemetry Channel (150Hz)
    let telemetryCount = 0;
    let lastTelemetryLog = Date.now();
    
    state.socket.on('telemetry', (d) => {
        if (!Array.isArray(d) || d.length < 6) return;
        
        const data = {
            roll: d[0], 
            pitch: d[1], 
            yaw: d[2],
            pos_x: d[3], 
            pos_y: d[4], 
            pos_z: d[5],
            connected: true
        };
        
        // Store for other uses
        state.telemetry = data;
        telemetryCount++;
        
        // Debug log every 2 seconds
        if (Date.now() - lastTelemetryLog > 2000) {
            console.log(`[SUB/TELEM] R:${data.roll.toFixed(2)} P:${data.pitch.toFixed(2)} @ ${telemetryCount/2} Hz`);
            telemetryCount = 0;
            lastTelemetryLog = Date.now();
        }
        
        // INSTANT updates - no buffering!
        if (typeof update3DVisualization === 'function') {
            update3DVisualization(data);
        }
        if (typeof updateArtificialHorizon === 'function') {
            updateArtificialHorizon(data);
        }
        if (typeof updateIMUDisplay === 'function') {
            updateIMUDisplay(data);
        }
    });
    
    // SUBSCRIBE: Health Channel (2Hz)
    state.socket.on('health', (data) => {
        if (typeof updateSystemHealth === 'function') {
            updateSystemHealth(data);
        }
    });
    
    state.socket.on('reference_captured', (data) => {
        state.hasReference = true;
        document.getElementById('referenceFrame').src = 'data:image/jpeg;base64,' + data.frame;
        document.getElementById('refFeatures').textContent = `📍 ${data.num_features} features`;
        
        // Hide overlay
        const refOverlay = document.getElementById('refOverlay');
        if (refOverlay) refOverlay.style.display = 'none';
        
        showNotification('Reference frame captured!', 'success');
    });
    
    state.socket.on('reference_reset', () => {
        state.hasReference = false;
        document.getElementById('referenceFrame').src = '';
        document.getElementById('refFeatures').textContent = '📍 0 features';
        
        // Show overlay again
        const refOverlay = document.getElementById('refOverlay');
        if (refOverlay) refOverlay.style.display = 'flex';
        
        showNotification('Reference reset', 'info');
    });
    
    state.socket.on('error', (data) => {
        showNotification(data.message, 'error');
    });
}

// Update connection status
function updateConnectionStatus(message, status) {
    const statusEl = document.getElementById('connectionStatus');
    const dotEl = statusEl.querySelector('.status-dot');
    const textEl = statusEl.querySelector('span:last-child');
    
    textEl.textContent = message;
    
    dotEl.style.background = status === 'success' ? 'var(--success)' : 
                             status === 'error' ? 'var(--error)' : 
                             'var(--warning)';
}

// Tab Navigation
function initTabs() {
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tabId = button.getAttribute('data-tab');
            
            // Remove active class from all
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));
            
            // Add active class to clicked tab
            button.classList.add('active');
            document.getElementById(tabId).classList.add('active');
        });
    });
}

// Video Stream
function startVideoStream() {
    const currentFrame = document.getElementById('currentFrame');
    const visionFrame = document.getElementById('visionFrame');
    const liveOverlay = document.getElementById('liveOverlay');
    
    // Hide overlay once first frame arrives
    if (currentFrame) {
        currentFrame.onload = function() {
            if (liveOverlay) liveOverlay.style.display = 'none';
        };
    }
    
    // Use video feed endpoint
    currentFrame.src = '/video_feed';
    visionFrame.src = '/video_feed';
}

// Update Dashboard with new data
function updateDashboard(data) {
    state.currentData = data;
    
    // Update FPS
    const fpsEl = document.getElementById('fpsValue');
    if (fpsEl && data.fps !== undefined) {
        fpsEl.textContent = data.fps.toFixed(1);
    }
    
    // Update matches
    const matchesEl = document.getElementById('matchesValue');
    if (matchesEl) {
        const matchesText = state.hasReference ? 
            `${data.num_matches || 0}/${data.max_matches || 0}` : '0/0';
        matchesEl.textContent = matchesText;
    }
    
    // Update confidence
    const confEl = document.getElementById('confidenceValue');
    if (confEl && data.confidence !== undefined) {
        confEl.textContent = data.confidence.toFixed(0) + '%';
    }
    
    // Update source
    const sourceEl = document.getElementById('sourceValue');
    if (sourceEl && data.guidance_source) {
        sourceEl.textContent = data.guidance_source.toUpperCase();
    }
    
    // Update guidance vector
    const dxEl = document.getElementById('dxValue');
    const dyEl = document.getElementById('dyValue');
    const dzEl = document.getElementById('dzValue');
    if (dxEl && data.dx !== undefined) dxEl.textContent = data.dx.toFixed(1) + ' px';
    if (dyEl && data.dy !== undefined) dyEl.textContent = data.dy.toFixed(1) + ' px';
    if (dzEl && data.dz_scale !== undefined) dzEl.textContent = data.dz_scale.toFixed(2) + 'x';
    
    // Update alignment status
    updateAlignmentStatus(data);
    
    // Update gauges
    if (typeof updateGauges === 'function') {
        updateGauges(data);
    }
    
    // Update detections
    if (data.yolo_detections) {
        updateDetections(data.yolo_detections);
    }
    
    // Update camera info
    const currFeatEl = document.getElementById('currFeatures');
    const yoloDetEl = document.getElementById('yoloDetections');
    if (currFeatEl) currFeatEl.textContent = `🎯 ${data.num_matches || 0} matches`;
    if (yoloDetEl && data.yolo_detections) yoloDetEl.textContent = `🔍 ${data.yolo_detections.length} objects`;
    
    // Update communication status - Camera FPS
    const cameraFPSEl = document.getElementById('cameraFPS');
    const cameraStatusEl = document.getElementById('cameraStatus');
    if (cameraFPSEl && data.fps !== undefined) {
        cameraFPSEl.textContent = `${data.fps.toFixed(1)} FPS`;
        // Add active class if camera is running
        if (cameraStatusEl && data.fps > 5) {
            cameraStatusEl.classList.add('active');
        }
    }
    
    // Update communication status - Pixhawk rate (from telemetry count)
    if (state.telemetry && state.telemetry.connected) {
        const pixhawkStatusEl = document.getElementById('pixhawkStatus');
        const pixhawkRateEl = document.getElementById('pixhawkRate');
        if (pixhawkStatusEl) pixhawkStatusEl.classList.add('active');
        if (pixhawkRateEl) pixhawkRateEl.textContent = '~150 Hz';
    }
    
    // Update history for charts
    updateHistory(data);
    if (typeof updateCharts === 'function') {
        updateCharts();
    }
}

// Update alignment status badge
function updateAlignmentStatus(data) {
    const statusEl = document.getElementById('alignmentStatus');
    const badge = statusEl.querySelector('.status-badge');
    
    const dx = Math.abs(data.dx);
    const dy = Math.abs(data.dy);
    const dz = Math.abs(data.dz_scale - 1.0);
    
    if (dx < 5 && dy < 5 && dz < 0.05) {
        badge.textContent = 'ALIGNED';
        badge.className = 'status-badge aligned';
    } else {
        badge.textContent = 'ADJUSTING';
        badge.className = 'status-badge adjusting';
    }
}

// Update detections list
function updateDetections(detections) {
    const listEl = document.getElementById('detectionList');
    
    if (detections.length === 0) {
        listEl.innerHTML = '<div style="text-align: center; color: var(--text-dim); padding: 20px;">No detections</div>';
        return;
    }
    
    listEl.innerHTML = detections.map(det => `
        <div class="detection-item">
            <span class="detection-class">${det.class}</span>
            <span class="detection-conf">${(det.confidence * 100).toFixed(1)}%</span>
        </div>
    `).join('');
}

// Update history data
function updateHistory(data) {
    const now = Date.now();
    
    state.history.timestamps.push(now);
    state.history.fps.push(data.fps || 0);
    state.history.matches.push(data.num_matches || 0);
    state.history.confidence.push(data.confidence || 0);
    
    // Calculate velocity magnitude (handle missing data)
    let velMag = 0;
    if (data.ekf_data) {
        velMag = Math.sqrt(
            (data.ekf_data.vel_x || 0) ** 2 + 
            (data.ekf_data.vel_y || 0) ** 2 + 
            (data.ekf_data.vel_z || 0) ** 2
        );
    }
    state.history.velocity.push(velMag);
    
    // Keep only last N entries
    if (state.history.timestamps.length > state.maxHistoryLength) {
        state.history.timestamps.shift();
        state.history.fps.shift();
        state.history.matches.shift();
        state.history.confidence.shift();
        state.history.velocity.shift();
    }
}

// Capture reference frame
function captureReference() {
    if (!state.connected) {
        showNotification('Not connected to server', 'error');
        return;
    }
    
    state.socket.emit('capture_reference');
    showNotification('Capturing reference...', 'info');
}

// Reset reference frame
function resetReference() {
    if (!state.connected) {
        showNotification('Not connected to server', 'error');
        return;
    }
    
    state.socket.emit('reset_reference');
}

// Show notification
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    
    // Style
    Object.assign(notification.style, {
        position: 'fixed',
        top: '80px',
        right: '20px',
        padding: '15px 25px',
        borderRadius: '8px',
        background: type === 'success' ? 'var(--success)' :
                    type === 'error' ? 'var(--error)' :
                    'var(--accent-blue)',
        color: 'white',
        fontWeight: '600',
        boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)',
        zIndex: '10000',
        animation: 'slideIn 0.3s ease'
    });
    
    document.body.appendChild(notification);
    
    // Remove after 3 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// Update IMU tab data
function updateIMUDisplay(data) {
    // Raw IMU data
    const rad2deg = 180 / Math.PI;
    document.getElementById('imuRoll').textContent = `${(data.roll * rad2deg).toFixed(2)}°`;
    document.getElementById('imuPitch').textContent = `${(data.pitch * rad2deg).toFixed(2)}°`;
    document.getElementById('imuYaw').textContent = `${(data.yaw * rad2deg).toFixed(2)}°`;
    document.getElementById('imuRollRate').textContent = '0.00 rad/s';
    document.getElementById('imuPitchRate').textContent = '0.00 rad/s';
    document.getElementById('imuYawRate').textContent = '0.00 rad/s';
    
    // EKF position data
    document.getElementById('ekfPosX').textContent = `${data.pos_x.toFixed(2)} m`;
    document.getElementById('ekfPosY').textContent = `${data.pos_y.toFixed(2)} m`;
    document.getElementById('ekfPosZ').textContent = `${data.pos_z.toFixed(2)} m`;
    document.getElementById('ekfVelX').textContent = '0.00 m/s';
    document.getElementById('ekfVelY').textContent = '0.00 m/s';
    document.getElementById('ekfVelZ').textContent = '0.00 m/s';
    
    // Fusion status
    document.getElementById('fusionIMU').textContent = data.connected ? 'Active' : 'Disconnected';
    document.getElementById('fusionIMU').style.color = data.connected ? 'var(--sea-green)' : 'var(--coral)';
    document.getElementById('fusionEKF').textContent = data.connected ? 'Healthy' : 'N/A';
    document.getElementById('fusionEKF').style.color = data.connected ? 'var(--sea-green)' : 'var(--text-dim)';
}

// Update system health display
function updateSystemHealth(data) {
    // CPU
    document.getElementById('healthCPU').textContent = `${data.cpu.percent}%`;
    const cpuBar = document.getElementById('healthCPUBar');
    cpuBar.style.width = `${data.cpu.percent}%`;
    cpuBar.className = 'health-bar-fill ' + data.cpu.status;
    
    // Memory
    document.getElementById('healthMem').textContent = `${data.memory.percent}%`;
    const memBar = document.getElementById('healthMemBar');
    memBar.style.width = `${data.memory.percent}%`;
    memBar.className = 'health-bar-fill ' + data.memory.status;
    
    // Disk
    document.getElementById('healthDisk').textContent = `${data.disk.percent}%`;
    const diskBar = document.getElementById('healthDiskBar');
    diskBar.style.width = `${data.disk.percent}%`;
    diskBar.className = 'health-bar-fill ' + data.disk.status;
    
    // Temperature
    if (data.temperature.celsius) {
        document.getElementById('healthTemp').textContent = `${data.temperature.celsius}°C`;
        const tempStatus = document.getElementById('healthTempStatus');
        tempStatus.textContent = data.temperature.status.toUpperCase();
        tempStatus.style.background = data.temperature.status === 'critical' ? 'rgba(255, 107, 107, 0.3)' :
                                      data.temperature.status === 'warning' ? 'rgba(237, 137, 54, 0.3)' :
                                      'rgba(32, 191, 107, 0.2)';
        tempStatus.style.color = data.temperature.status === 'critical' ? '#ff6b6b' :
                                data.temperature.status === 'warning' ? '#ed8936' :
                                'var(--sea-green)';
    }
    
    // Uptime
    document.getElementById('healthUptime').textContent = `${data.uptime_hours.toFixed(1)}h`;
    
    // Network
    document.getElementById('networkRate').textContent = `↓ ${data.network.recv_mb.toFixed(1)}MB ↑ ${data.network.sent_mb.toFixed(1)}MB`;
}

// Add animation styles
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(400px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(400px); opacity: 0; }
    }
`;
document.head.appendChild(style);

// Handle window resize
window.addEventListener('resize', () => {
    if (state.charts) {
        Object.values(state.charts).forEach(chart => {
            if (chart) chart.resize();
        });
    }
});

