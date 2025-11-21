/**
 * Feature Extraction Module for PARIMA
 * Collects visibility metrics, device metrics, and viewport trajectory
 */

let camera = null;
let renderer = null;
let currentActor = null;
let viewportHistory = [];
let fpsHistory = [];
let lastFrameTime = null;
let viewportHistorySize = 10;

// GPU metrics caching
let lastGPUMetrics = null;
let gpuMetricsCacheTime = 0;
const GPU_CACHE_DURATION = 1000; // Cache for 1 second
let gpuMetricsAPIUrl = null;

/**
 * Initialize feature extractor with required VTK.js objects
 * @param {Object} vtkObjects - Object containing camera, renderer, actor references
 * @param {Object} config - Configuration object with viewportHistorySize
 */
export function initializeFeatureExtractor(vtkObjects, config = {}) {
  camera = vtkObjects.camera;
  renderer = vtkObjects.renderer;
  currentActor = vtkObjects.actor;
  viewportHistorySize = config.viewportHistorySize || 10;
  lastFrameTime = performance.now();
  
  // Initialize FPS tracking
  fpsHistory = [];
}

/**
 * Compute frustum coverage - percentage of mesh points within camera frustum
 * @returns {number} Coverage ratio (0-1)
 */
function computeFrustumCoverage() {
  if (!currentActor || !renderer || !camera) {
    return 0.5; // Default if no actor loaded
  }
  
  try {
    const mapper = currentActor.getMapper();
    const inputData = mapper.getInputData();
    
    if (!inputData || !inputData.getPoints()) {
      return 0.5;
    }
    
    const points = inputData.getPoints();
    const numPoints = points.getNumberOfPoints();
    if (numPoints === 0) return 0.0;
    
    const bounds = renderer.computeVisiblePropBounds();
    const pointData = points.getData();
    
    let visibleCount = 0;
    const sampleSize = Math.min(numPoints, 1000); // Sample for performance
    const step = Math.max(1, Math.floor(numPoints / sampleSize));
    
    for (let i = 0; i < numPoints; i += step) {
      const x = pointData[i * 3];
      const y = pointData[i * 3 + 1];
      const z = pointData[i * 3 + 2];
      
      // Check if point is within visible bounds
      if (x >= bounds[0] && x <= bounds[1] &&
          y >= bounds[2] && y <= bounds[3] &&
          z >= bounds[4] && z <= bounds[5]) {
        visibleCount++;
      }
    }
    
    return visibleCount / (numPoints / step);
  } catch (error) {
    console.warn('Frustum coverage computation failed:', error);
    return 0.5;
  }
}

/**
 * Compute occlusion ratio - estimate of occluded vs visible geometry
 * This is a simplified estimation based on depth complexity
 * @returns {number} Occlusion ratio (0-1, higher = more occluded)
 */
function computeOcclusionRatio() {
  if (!currentActor || !renderer) {
    return 0.3; // Default
  }
  
  try {
    // Simplified occlusion estimation based on geometry complexity
    const mapper = currentActor.getMapper();
    const inputData = mapper.getInputData();
    
    if (!inputData) return 0.3;
    
    const cells = inputData.getPolys();
    const points = inputData.getPoints();
    
    if (!cells || !points) return 0.3;
    
    const numCells = cells.getNumberOfCells();
    const numPoints = points.getNumberOfPoints();
    
    // Estimate occlusion based on geometry density
    const density = numCells / Math.max(1, numPoints);
    // Normalize to 0-1 range (higher density = more potential occlusion)
    return Math.min(1.0, density / 10.0);
  } catch (error) {
    console.warn('Occlusion ratio computation failed:', error);
    return 0.3;
  }
}

/**
 * Compute mean visible distance from camera
 * @returns {number} Mean distance in world units
 */
function computeMeanVisibleDistance() {
  if (!camera || !currentActor || !renderer) {
    return 100.0; // Default distance
  }
  
  try {
    const cameraPos = camera.getPosition();
    const focalPoint = camera.getFocalPoint();
    
    // Distance to focal point
    const dx = cameraPos[0] - focalPoint[0];
    const dy = cameraPos[1] - focalPoint[1];
    const dz = cameraPos[2] - focalPoint[2];
    
    const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
    
    // Also consider bounds distance
    const bounds = renderer.computeVisiblePropBounds();
    if (bounds && bounds.length === 6) {
      const centerX = (bounds[0] + bounds[1]) / 2;
      const centerY = (bounds[2] + bounds[3]) / 2;
      const centerZ = (bounds[4] + bounds[5]) / 2;
      
      const centerDx = cameraPos[0] - centerX;
      const centerDy = cameraPos[1] - centerY;
      const centerDz = cameraPos[2] - centerZ;
      
      const centerDist = Math.sqrt(centerDx * centerDx + centerDy * centerDy + centerDz * centerDz);
      
      return (distance + centerDist) / 2;
    }
    
    return Math.max(1.0, distance);
  } catch (error) {
    console.warn('Mean visible distance computation failed:', error);
    return 100.0;
  }
}

/**
 * Get current device FPS
 * @returns {number} Frames per second
 */
function getDeviceFPS() {
  const now = performance.now();
  
  if (lastFrameTime === null) {
    lastFrameTime = now;
    return 60.0; // Default
  }
  
  const deltaTime = now - lastFrameTime;
  lastFrameTime = now;
  
  if (deltaTime === 0) return 60.0;
  
  const fps = 1000.0 / deltaTime;
  
  // Maintain history for smoothing
  fpsHistory.push(fps);
  if (fpsHistory.length > 30) {
    fpsHistory.shift();
  }
  
  // Return average of recent FPS
  const avgFps = fpsHistory.reduce((sum, val) => sum + val, 0) / fpsHistory.length;
  //console.log('Raw FPS:', fps, 'Average FPS:', avgFps, 'Delta:', deltaTime);
  return Math.max(1.0, avgFps); // Clamp to reasonable range
}

/**
 * Get device CPU load estimate
 * @returns {number} CPU load estimate (0-1)
 */
function getDeviceCPULoad() {
  try {
    // Use Performance API if available
    if (performance.memory) {
      const memInfo = performance.memory;
      const usedRatio = memInfo.usedJSHeapSize / memInfo.jsHeapSizeLimit;
      // Use memory pressure as proxy for CPU load
      return Math.min(1.0, usedRatio * 1.5); // Scale slightly
    }
    
    // Estimate based on FPS (lower FPS = higher load)
    const fps = getDeviceFPS();
    const loadEstimate = Math.max(0.0, Math.min(1.0, 1.0 - (fps / 60.0)));
    return loadEstimate;
  } catch (error) {
    console.warn('CPU load estimation failed:', error);
    return 0.5; // Default
  }
}

/**
 * Initialize GPU metrics API URL
 * @param {string} apiUrl - Base API URL (e.g., from PARIMA config)
 */
export function initializeGPUMetrics(apiUrl) {
  // Extract base URL from PARIMA API URL
  if (apiUrl) {
    try {
      const url = new URL(apiUrl);
      gpuMetricsAPIUrl = `${url.protocol}//${url.host}/api/system/gpu`;
    } catch (error) {
      console.warn('Failed to parse API URL for GPU metrics:', error);
      gpuMetricsAPIUrl = null;
    }
  }
}

/**
 * Get GPU load from system (macOS) via backend API
 * @returns {Promise<number|null>} GPU usage percentage or null
 */
async function getGPULoadFromSystem() {
  // Check cache
  const now = Date.now();
  if (now - gpuMetricsCacheTime < GPU_CACHE_DURATION && lastGPUMetrics !== null) {
    return lastGPUMetrics;
  }
  
  if (!gpuMetricsAPIUrl) {
    return null;
  }
  
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 500); // 500ms timeout
    
    const response = await fetch(gpuMetricsAPIUrl, {
      method: 'GET',
      signal: controller.signal
    });
    
    clearTimeout(timeoutId);
    
    if (response.ok) {
      const data = await response.json();
      
      if (data.success && data.available && data.gpu_usage !== null) {
        lastGPUMetrics = data.gpu_usage;
        gpuMetricsCacheTime = now;
        return data.gpu_usage;
      }
    }
  } catch (error) {
    // Backend not available, timeout, or error
    // Silently fail and use fallback estimation
    if (error.name !== 'AbortError') {
      console.debug('GPU metrics API not available:', error.message);
    }
  }
  
  return null;
}

/**
 * Get GPU load with fallback to estimation
 * @returns {Promise<number>} GPU load percentage (0-100)
 */
export async function getDeviceGPULoad() {
  // Try to get real GPU metrics from system
  const systemGPU = await getGPULoadFromSystem();
  if (systemGPU !== null) {
    return systemGPU;
  }
  
  // Fallback to estimation based on FPS and CPU
  const cpuLoad = getDeviceCPULoad();
  const fps = getDeviceFPS();
  const fpsRatio = fps / 60.0;
  const estimatedGPU = Math.min(100, Math.max(0, (1.0 - fpsRatio) * 100 + (cpuLoad * 30)));
  
  return estimatedGPU;
}

/**
 * Update viewport trajectory history
 */
function updateViewportTrajectory() {
  if (!camera) return;
  
  try {
    const cameraPos = camera.getPosition();
    const focalPoint = camera.getFocalPoint();
    
    const currentState = {
      timestamp: performance.now(),
      position: [...cameraPos],
      focalPoint: [...focalPoint]
    };
    
    // Compute velocity and acceleration from history
    if (viewportHistory.length > 0) {
      const lastState = viewportHistory[viewportHistory.length - 1];
      const dt = (currentState.timestamp - lastState.timestamp) / 1000.0; // Convert to seconds
      
      if (dt > 0) {
        // Linear velocity
        const dx = currentState.position[0] - lastState.position[0];
        const dy = currentState.position[1] - lastState.position[1];
        const dz = currentState.position[2] - lastState.position[2];
        currentState.velocity = Math.sqrt(dx * dx + dy * dy + dz * dz) / dt;
        
        // Angular velocity (based on focal point change)
        const focalDx = currentState.focalPoint[0] - lastState.focalPoint[0];
        const focalDy = currentState.focalPoint[1] - lastState.focalPoint[1];
        const focalDz = currentState.focalPoint[2] - lastState.focalPoint[2];
        currentState.angularVelocity = Math.sqrt(focalDx * focalDx + focalDy * focalDy + focalDz * focalDz) / dt;
        
        // Acceleration (if we have previous velocity)
        if (lastState.velocity !== undefined) {
          currentState.acceleration = (currentState.velocity - lastState.velocity) / dt;
        } else {
          currentState.acceleration = 0.0;
        }
      } else {
        currentState.velocity = 0.0;
        currentState.angularVelocity = 0.0;
        currentState.acceleration = 0.0;
      }
    } else {
      currentState.velocity = 0.0;
      currentState.angularVelocity = 0.0;
      currentState.acceleration = 0.0;
    }
    
    viewportHistory.push(currentState);
    
    // Maintain history size
    if (viewportHistory.length > viewportHistorySize) {
      viewportHistory.shift();
    }
  } catch (error) {
    console.warn('Viewport trajectory update failed:', error);
  }
}

/**
 * Collect all features for PARIMA prediction
 * @returns {Promise<Object>} Feature dictionary
 */
export async function collectFeatures() {
  // Update viewport trajectory
  updateViewportTrajectory();
  
  // Collect visibility metrics
  const frustumCoverage = computeFrustumCoverage();
  const occlusionRatio = computeOcclusionRatio();
  const meanVisibleDistance = computeMeanVisibleDistance();
  
  // Collect device metrics
  const deviceFPS = getDeviceFPS();
  const deviceCPULoad = getDeviceCPULoad();
  const deviceGPULoad = await getDeviceGPULoad(); // Now async!
  
  // Prepare viewport trajectory
  const trajectory = viewportHistory.map(state => ({
    velocity: state.velocity || 0.0,
    acceleration: state.acceleration || 0.0,
    angularVelocity: state.angularVelocity || 0.0
  }));
  
  return {
    frustumCoverage,
    occlusionRatio,
    meanVisibleDistance,
    deviceFPS,
    deviceCPULoad,
    deviceGPULoad, // Add GPU load
    viewportTrajectory: trajectory
  };
}

/**
 * Reset feature extractor state
 */
export function resetFeatureExtractor() {
  viewportHistory = [];
  fpsHistory = [];
  lastFrameTime = performance.now();
}

