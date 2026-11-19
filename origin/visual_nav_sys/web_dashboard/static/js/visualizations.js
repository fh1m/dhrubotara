// Visualizations: Charts, Gauges, 3D, and Horizon
// Uses RAW IMU data for all visualizations

// Chart.js configurations (Stabilized)
state.charts = {};

function initCharts() {
    const chartConfig = {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: {
            legend: { display: true, labels: { color: '#a0aec0' } }
        },
        scales: {
            x: {
                grid: { color: '#2d3748' },
                ticks: { color: '#718096', maxTicksLimit: 10 }
            },
            y: {
                grid: { color: '#2d3748' },
                ticks: { color: '#718096' }
            }
        },
        elements: {
            point: { radius: 0 }
        }
    };
    
    // FPS Chart
    const fpsCtx = document.getElementById('fpsChart');
    if (fpsCtx) {
        state.charts.fps = new Chart(fpsCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'FPS',
                    data: [],
                    borderColor: '#4a90e2',
                    backgroundColor: 'rgba(74, 144, 226, 0.1)',
                    fill: true,
                    tension: 0.1
                }]
            },
            options: { 
                ...chartConfig, 
                scales: { 
                    ...chartConfig.scales, 
                    y: { min: 0, max: 60, grid: { color: '#2d3748' } }
                }
            }
        });
    }
    
    // Confidence Chart
    const confCtx = document.getElementById('confidenceChart');
    if (confCtx) {
        state.charts.confidence = new Chart(confCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Confidence %',
                    data: [],
                    borderColor: '#48bb78',
                    backgroundColor: 'rgba(72, 187, 120, 0.1)',
                    fill: true,
                    tension: 0.1
                }]
            },
            options: { 
                ...chartConfig, 
                scales: { 
                    ...chartConfig.scales, 
                    y: { min: 0, max: 100, grid: { color: '#2d3748' } }
                }
            }
        });
    }
    
    // Matches Chart
    const matchCtx = document.getElementById('matchesChart');
    if (matchCtx) {
        state.charts.matches = new Chart(matchCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Feature Matches',
                    data: [],
                    borderColor: '#5bc0de',
                    backgroundColor: 'rgba(91, 192, 222, 0.1)',
                    fill: true,
                    tension: 0.1
                }]
            },
            options: chartConfig
        });
    }
    
    // Velocity Chart
    const velCtx = document.getElementById('velocityChart');
    if (velCtx) {
        state.charts.velocity = new Chart(velCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Velocity (m/s)',
                    data: [],
                    borderColor: '#ed8936',
                    backgroundColor: 'rgba(237, 131, 54, 0.1)',
                    fill: true,
                    tension: 0.1
                }]
            },
            options: chartConfig
        });
    }
    
    // Match History Chart (in Vision tab)
    const matchHistCtx = document.getElementById('matchHistoryChart');
    if (matchHistCtx) {
        state.charts.matchHistory = new Chart(matchHistCtx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Matches',
                    data: [],
                    backgroundColor: 'rgba(74, 144, 226, 0.6)',
                    borderColor: '#4a90e2',
                    borderWidth: 2
                }]
            },
            options: {
                ...chartConfig,
                animation: { duration: 500 }
            }
        });
    }
}

function updateCharts() {
    const labels = state.history.timestamps.map((t, i) => i);
    
    if (state.charts.fps) {
        state.charts.fps.data.labels = labels;
        state.charts.fps.data.datasets[0].data = state.history.fps;
        state.charts.fps.update('none');
    }
    
    if (state.charts.confidence) {
        state.charts.confidence.data.labels = labels;
        state.charts.confidence.data.datasets[0].data = state.history.confidence;
        state.charts.confidence.update('none');
    }
    
    if (state.charts.matches) {
        state.charts.matches.data.labels = labels;
        state.charts.matches.data.datasets[0].data = state.history.matches;
        state.charts.matches.update('none');
    }
    
    if (state.charts.velocity) {
        state.charts.velocity.data.labels = labels;
        state.charts.velocity.data.datasets[0].data = state.history.velocity;
        state.charts.velocity.update('none');
    }
    
    if (state.charts.matchHistory) {
        state.charts.matchHistory.data.labels = labels.slice(-20);
        state.charts.matchHistory.data.datasets[0].data = state.history.matches.slice(-20);
        state.charts.matchHistory.update('none');
    }
}

// Circular Gauges
state.gauges = {};

function initGauges() {
    state.gauges.speedX = new CircularGauge('gaugeSpeedX', -50, 50, 'px/s');
    state.gauges.speedY = new CircularGauge('gaugeSpeedY', -50, 50, 'px/s');
}

function updateGauges(data) {
    if (state.gauges.speedX) {
        state.gauges.speedX.update(data.dx);
    }
    if (state.gauges.speedY) {
        state.gauges.speedY.update(data.dy);
    }
}

class CircularGauge {
    constructor(canvasId, minVal, maxVal, unit) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) return;
        this.ctx = this.canvas.getContext('2d');
        this.minVal = minVal;
        this.maxVal = maxVal;
        this.unit = unit;
        this.value = 0;
        this.targetValue = 0;
        this.animationFrame = null;
        this.animate();
    }
    
    update(value) {
        // INSTANT update - no smoothing
        this.value = Math.max(this.minVal, Math.min(this.maxVal, value));
        this.draw();
    }
    
    animate() {
        // Just redraw at screen refresh rate
        this.draw();
        this.animationFrame = requestAnimationFrame(() => this.animate());
    }
    
    draw() {
        if (!this.ctx) return;
        const ctx = this.ctx;
        const centerX = this.canvas.width / 2;
        const centerY = this.canvas.height / 2;
        const radius = 70;
        
        ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
        ctx.strokeStyle = '#2d3748';
        ctx.lineWidth = 12;
        ctx.stroke();
        
        const normalized = (this.value - this.minVal) / (this.maxVal - this.minVal);
        const startAngle = -Math.PI / 2;
        const endAngle = startAngle + (normalized * 2 * Math.PI);
        
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, startAngle, endAngle);
        ctx.strokeStyle = this.getColor(normalized);
        ctx.lineWidth = 12;
        ctx.lineCap = 'round';
        ctx.stroke();
        
        ctx.fillStyle = '#e8eef5';
        ctx.font = 'bold 28px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(this.value.toFixed(0), centerX, centerY - 5);
        
        ctx.fillStyle = '#718096';
        ctx.font = '12px Arial';
        ctx.fillText(this.unit, centerX, centerY + 20);
    }
    
    getColor(normalized) {
        const absNorm = Math.abs(normalized - 0.5) * 2;
        if (absNorm < 0.3) return '#48bb78';
        if (absNorm < 0.7) return '#ed8936';
        return '#f56565';
    }
}

// 3D Visualization with Three.js
let scene, camera, renderer, drone;

function init3DVisualization() {
    const container = document.getElementById('threejs-container');
    if (!container) return;
    
    // Clear existing
    if (renderer) {
        container.innerHTML = '';
    }
    
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a1628);
    // Add depth fog for professional look
    scene.fog = new THREE.Fog(0x0a1628, 5, 25);
    
    // ULTRA ZOOMED IN camera - fills the view!
    camera = new THREE.PerspectiveCamera(
        65,  // Wide FOV for immersion
        container.clientWidth / container.clientHeight,
        0.1,
        1000
    );
    camera.position.set(3, 2.5, 3);  // VERY close!
    camera.lookAt(0, 1.5, 0);
    
    // High quality renderer with shadows
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);
    
    // Enhanced lighting for depth
    const ambientLight = new THREE.AmbientLight(0x4a90e2, 0.4);
    scene.add(ambientLight);
    
    const keyLight = new THREE.DirectionalLight(0xffffff, 1.2);
    keyLight.position.set(5, 10, 5);
    keyLight.castShadow = true;
    keyLight.shadow.mapSize.width = 2048;
    keyLight.shadow.mapSize.height = 2048;
    scene.add(keyLight);
    
    const fillLight = new THREE.DirectionalLight(0x00d4ff, 0.5);
    fillLight.position.set(-5, 5, -5);
    scene.add(fillLight);
    
    // Professional grid - underwater style
    const gridHelper = new THREE.GridHelper(30, 30, 0x0077BE, 0x1a3a52);
    gridHelper.material.opacity = 0.7;
    gridHelper.material.transparent = true;
    gridHelper.receiveShadow = true;
    scene.add(gridHelper);
    
    // Enhanced AUV/Drone Model - BRACU DUBURI style
    drone = new THREE.Group();
    drone.castShadow = true;
    
    // Main body - cylindrical like an AUV
    const bodyGeo = new THREE.CylinderGeometry(0.3, 0.3, 1.2, 16);
    const bodyMat = new THREE.MeshPhongMaterial({ 
        color: 0x1a1a1a,
        shininess: 80,
        specular: 0x222222
    });
    const body = new THREE.Mesh(bodyGeo, bodyMat);
    body.rotation.z = Math.PI / 2;
    body.castShadow = true;
    drone.add(body);
    
    // Front nose cone (forward direction indicator)
    const noseGeo = new THREE.ConeGeometry(0.3, 0.5, 16);
    const noseMat = new THREE.MeshPhongMaterial({ 
        color: 0x0077BE,
        emissive: 0x0077BE,
        emissiveIntensity: 0.3
    });
    const nose = new THREE.Mesh(noseGeo, noseMat);
    nose.position.x = 0.85;
    nose.rotation.z = -Math.PI / 2;
    nose.castShadow = true;
    drone.add(nose);
    
    // Thruster pods (4 corners)
    const thrusterGeo = new THREE.BoxGeometry(0.15, 0.15, 0.4);
    const thrusterMat = new THREE.MeshPhongMaterial({ color: 0x444444 });
    
    [
        { x: 0.4, y: 0.25, z: 0.25 },
        { x: 0.4, y: -0.25, z: 0.25 },
        { x: 0.4, y: 0.25, z: -0.25 },
        { x: 0.4, y: -0.25, z: -0.25 }
    ].forEach(p => {
        const thruster = new THREE.Mesh(thrusterGeo, thrusterMat);
        thruster.position.set(p.x, p.y, p.z);
        thruster.castShadow = true;
        drone.add(thruster);
        
        // Propeller glow
        const propGlow = new THREE.Mesh(
            new THREE.CircleGeometry(0.2, 8),
            new THREE.MeshBasicMaterial({ 
                color: 0x00d4ff, 
                transparent: true, 
                opacity: 0.6,
                side: THREE.DoubleSide
            })
        );
        propGlow.position.set(p.x + 0.25, p.y, p.z);
        propGlow.rotation.y = Math.PI / 2;
        drone.add(propGlow);
    });
    
    // Reference axes with labels
    const axisLength = 1.5;
    const axes = new THREE.Group();
    
    // X axis (Red - Roll)
    const xAxis = new THREE.ArrowHelper(
        new THREE.Vector3(1, 0, 0), 
        new THREE.Vector3(0, 0, 0), 
        axisLength, 
        0xff0000, 
        0.2, 
        0.2
    );
    axes.add(xAxis);
    
    // Y axis (Green - Pitch) 
    const yAxis = new THREE.ArrowHelper(
        new THREE.Vector3(0, 1, 0), 
        new THREE.Vector3(0, 0, 0), 
        axisLength, 
        0x00ff00, 
        0.2, 
        0.2
    );
    axes.add(yAxis);
    
    // Z axis (Blue - Yaw)
    const zAxis = new THREE.ArrowHelper(
        new THREE.Vector3(0, 0, 1), 
        new THREE.Vector3(0, 0, 0), 
        axisLength, 
        0x0077BE, 
        0.2, 
        0.2
    );
    axes.add(zAxis);
    
    drone.add(axes);
    
    drone.position.set(0, 1.5, 0); // Start above grid
    scene.add(drone);
    
    // Start animation loop
    animate3D();
    
    window.addEventListener('resize', () => {
        if (!container) return;
        camera.aspect = container.clientWidth / container.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(container.clientWidth, container.clientHeight);
    });
}

function animate3D() {
    requestAnimationFrame(animate3D);
    if (renderer && scene && camera) {
        renderer.render(scene, camera);
    }
}

// MAIN UPDATE FUNCTION - Called from telemetry event
// INSTANT updates - no smoothing for real-time response
function update3DVisualization(data) {
    if (!drone) {
        init3DVisualization();
        return;
    }
    
    // Get rotation values (in radians from Pixhawk)
    const roll = data.roll || 0;
    const pitch = data.pitch || 0;
    const yaw = data.yaw || 0;
    
    // DIRECT assignment for INSTANT response
    // Pixhawk sends in NED frame, Three.js uses different conventions
    drone.rotation.order = 'ZYX';  // Yaw, Pitch, Roll order
    drone.rotation.x = pitch;      // Pitch
    drone.rotation.z = -roll;      // Roll (inverted)
    drone.rotation.y = -yaw;       // Yaw (inverted)
    
    // Position from dead reckoning (or EKF if available)
    const posX = data.pos_x || 0;
    const posY = data.pos_y || 0;
    const posZ = data.pos_z || 0;
    
    const scale = 2.0;
    
    // NED to Three.js coordinate mapping
    drone.position.x = posY * scale;   // East -> X
    drone.position.z = -posX * scale;  // North -> -Z
    drone.position.y = Math.max(1.0, 1.5 - posZ * scale); // Keep above ground
    
    // ULTRA ZOOMED IN camera - fills the view like a cockpit!
    const cameraDistance = 3;  // VERY close!
    const cameraHeight = 2.5;  // Close eye level
    
    camera.position.set(
        drone.position.x + cameraDistance * 0.7,
        drone.position.y + cameraHeight,
        drone.position.z + cameraDistance * 0.7
    );
    camera.lookAt(drone.position);
}

// Artificial Horizon (PFD Style)
let horizonInitialized = false;
function updateArtificialHorizon(data) {
    const canvas = document.getElementById('horizonCanvas');
    if (!canvas) {
        console.warn('[Horizon] Canvas not found!');
        return;
    }
    
    if (!horizonInitialized) {
        console.log('[Horizon] Initialized canvas:', canvas.width, 'x', canvas.height);
        horizonInitialized = true;
    }
    
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    const centerX = w / 2;
    const centerY = h / 2;
    
    // Get RAW attitude data
    const roll = data.roll || 0;
    const pitch = data.pitch || 0;
    const yaw = data.yaw || 0;
    
    // Clear
    ctx.clearRect(0, 0, w, h);
    
    // --- Sky/Ground Background ---
    ctx.save();
    ctx.translate(centerX, centerY);
    ctx.rotate(-roll);
    
    const pitchPx = pitch * (180 / Math.PI) * 4; // 4 pixels per degree
    ctx.translate(0, pitchPx);
    
    // Sky
    ctx.fillStyle = '#0077BE';
    ctx.fillRect(-w, -h*2, w*2, h*2);
    
    // Ground
    ctx.fillStyle = '#8B4513';
    ctx.fillRect(-w, 0, w*2, h*2);
    
    // Horizon Line
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(-w, 0);
    ctx.lineTo(w, 0);
    ctx.stroke();
    
    // --- Pitch Ladder ---
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 1;
    ctx.font = '12px sans-serif';
    ctx.fillStyle = 'white';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    
    for (let i = -90; i <= 90; i += 10) {
        if (i === 0) continue;
        const y = -i * 4;
        
        if (y + pitchPx > -h/2 && y + pitchPx < h/2) {
            const width = (Math.abs(i) % 20 === 0) ? 60 : 30;
            ctx.beginPath();
            ctx.moveTo(-width, y);
            ctx.lineTo(width, y);
            ctx.stroke();
            
            if (Math.abs(i) % 20 === 0) {
                ctx.fillText(Math.abs(i).toString(), -width - 15, y);
                ctx.fillText(Math.abs(i).toString(), width + 15, y);
            }
        }
    }
    
    ctx.restore();
    
    // --- Aircraft Reference Symbol ---
    ctx.strokeStyle = '#FFFF00';
    ctx.lineWidth = 3;
    ctx.shadowColor = 'black';
    ctx.shadowBlur = 4;
    
    ctx.beginPath();
    ctx.moveTo(centerX - 40, centerY);
    ctx.lineTo(centerX - 10, centerY);
    ctx.lineTo(centerX - 10, centerY + 10);
    ctx.moveTo(centerX + 40, centerY);
    ctx.lineTo(centerX + 10, centerY);
    ctx.lineTo(centerX + 10, centerY + 10);
    ctx.moveTo(centerX - 2, centerY);
    ctx.lineTo(centerX + 2, centerY);
    ctx.stroke();
    ctx.shadowBlur = 0;
    
    // --- Roll Scale ---
    ctx.save();
    ctx.translate(centerX, centerY);
    const r = Math.min(centerY, centerX) - 30;  // Adaptive radius
    
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(0, 0, r, -Math.PI*0.8, -Math.PI*0.2);
    ctx.stroke();
    
    [-60, -45, -30, -20, -10, 0, 10, 20, 30, 45, 60].forEach(deg => {
        const rad = (deg - 90) * Math.PI / 180;
        const tickLen = (deg % 30 === 0) ? 10 : 5;
        
        const x1 = Math.cos(rad) * r;
        const y1 = Math.sin(rad) * r;
        const x2 = Math.cos(rad) * (r + tickLen);
        const y2 = Math.sin(rad) * (r + tickLen);
        
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
    });
    
    // Roll Pointer
    ctx.rotate(-roll);
    ctx.fillStyle = 'white';
    ctx.beginPath();
    ctx.moveTo(0, -r + 2);
    ctx.lineTo(-5, -r + 12);
    ctx.lineTo(5, -r + 12);
    ctx.closePath();
    ctx.fill();
    
    ctx.restore();
    
    // --- Text Data ---
    ctx.font = 'bold 14px monospace';
    ctx.fillStyle = '#00FF00';
    
    ctx.textAlign = 'left';
    ctx.fillText(`PIT: ${(pitch * 180 / Math.PI).toFixed(1)}°`, 10, 20);
    ctx.fillText(`ROL: ${(roll * 180 / Math.PI).toFixed(1)}°`, 10, 40);
    
    ctx.textAlign = 'right';
    const hdg = ((yaw * 180 / Math.PI + 360) % 360).toFixed(1);
    ctx.fillText(`HDG: ${hdg}°`, w - 10, 20);
    
    // Connection indicator
    if (data.connected) {
        ctx.fillStyle = '#00FF00';
        ctx.fillText('IMU OK', w - 10, 40);
    } else {
        ctx.fillStyle = '#FF0000';
        ctx.fillText('NO IMU', w - 10, 40);
    }
}

// Initialize 3D on load
document.addEventListener('DOMContentLoaded', () => {
    setTimeout(() => {
        init3DVisualization();
    }, 500);
});
