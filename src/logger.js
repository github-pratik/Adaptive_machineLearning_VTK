/**
 * Logger Module for PARIMA Decision Logging
 * Logs features, decisions, and performance outcomes to CSV format
 */

export class Logger {
  constructor(logFilePath = 'parima_decisions_log.csv') {
    this.logFilePath = logFilePath;
    this.logEntries = [];
    this.initialized = false;
    this.maxBufferSize = 100; // Buffer entries before download
    // Memory peak tracking
    this.memoryPeakMB = 0;
    this.jsHeapPeakMB = 0;
    // LOD change tracking for adaptation metrics
    this.previousLOD = null;
    this.lodChangeCount = 0;
    // FPS history for stability calculation
    this.fpsHistory = [];
  }

  /**
   * Initialize CSV header
   */
  initializeCSV() {
    if (this.initialized) return;
    
    const header = [
      'timestamp',
      'frustumCoverage',
      'occlusionRatio',
      'meanVisibleDistance',
      'deviceFPS',
      'deviceCPULoad',
      'deviceGPULoad',
      'viewportVelocity',
      'viewportAcceleration',
      'viewportAngularVelocity',
      'decisionLOD',
      'decisionTiles',
      'latencyMs',
      'fpsAfterDecision',
      'memoryMB',           // Current tile memory
      'jsHeapMemoryMB',     // Current JS heap memory
      'memoryPeakMB',       // Peak tile memory seen
      'jsHeapPeakMB',       // Peak JS heap memory seen
      'fpsImprovement',     // (fpsAfterDecision - deviceFPS) / deviceFPS * 100
      'qualityScore',       // (5 - decisionLOD) / 5.0 (0-1, higher = better quality)
      'fpsPerGPUPercent',   // deviceFPS / deviceGPULoad (efficiency metric)
      'fpsPerMemoryMB',     // deviceFPS / memoryMB (efficiency metric)
      'inAcceptableRange',  // 1 if fps >= 30, else 0
      'lodChangeFromPrevious' // 1 if LOD changed, 0 if same
    ].join(',');
    
    this.logEntries.push(header);
    this.initialized = true;
  }

  /**
   * Get current JavaScript heap memory
   * @returns {number} Memory in MB, or 0 if unavailable
   */
  getJSHeapMemory() {
    try {
      if (performance.memory) {
        const memInfo = performance.memory;
        return memInfo.usedJSHeapSize / (1024 * 1024); // Convert to MB
      }
    } catch (error) {
      // Performance API not available
    }
    return 0;
  }

  /**
   * Log a decision entry
   * @param {Object} entryObj - Entry object containing features, decision, and performance metrics
   */
  logEntry(entryObj) {
    this.initializeCSV();
    
    const timestamp = entryObj.timestamp || Date.now();
    
    // Extract viewport trajectory summary
    const trajectory = entryObj.viewportTrajectory || [];
    const lastTrajectory = trajectory.length > 0 ? trajectory[trajectory.length - 1] : {};
    
    const decision = entryObj.decision || {};
    const decisionLOD = decision.lod !== undefined ? decision.lod : -1;
    const decisionTiles = decision.tiles ? decision.tiles.join(';') : '';
    
    // Get current JS heap memory
    const jsHeapMemoryMB = this.getJSHeapMemory();
    
    // Track memory peaks
    const currentMemoryMB = entryObj.memoryMB || 0;
    if (currentMemoryMB > this.memoryPeakMB) {
      this.memoryPeakMB = currentMemoryMB;
    }
    if (jsHeapMemoryMB > this.jsHeapPeakMB) {
      this.jsHeapPeakMB = jsHeapMemoryMB;
    }
    
    // Calculate PARIMA-supporting metrics
    const deviceFPS = entryObj.deviceFPS || 0;
    const fpsAfterDecision = entryObj.fpsAfterDecision || 0;
    const deviceGPULoad = entryObj.deviceGPULoad || 0;
    
    // FPS Improvement: percentage change after decision
    const fpsImprovement = (deviceFPS > 0 && fpsAfterDecision > 0)
      ? ((fpsAfterDecision - deviceFPS) / deviceFPS * 100).toFixed(2)
      : '';
    
    // Quality Score: (5 - LOD) / 5.0, where LOD 0 = highest quality (score 1.0)
    const qualityScore = (decisionLOD !== -1 && decisionLOD >= 0 && decisionLOD <= 5)
      ? ((5 - decisionLOD) / 5.0).toFixed(4)
      : '';
    
    // FPS per GPU Percent: efficiency metric (higher = better)
    const fpsPerGPUPercent = (deviceGPULoad > 0)
      ? (deviceFPS / deviceGPULoad).toFixed(2)
      : '';
    
    // FPS per Memory MB: efficiency metric (higher = better)
    const fpsPerMemoryMB = (currentMemoryMB > 0)
      ? (deviceFPS / currentMemoryMB).toFixed(2)
      : '';
    
    // In Acceptable Range: 1 if FPS >= 30, else 0
    const inAcceptableRange = (deviceFPS >= 30) ? '1' : '0';
    
    // LOD Change from Previous: 1 if changed, 0 if same
    let lodChangeFromPrevious = '0';
    if (this.previousLOD !== null && decisionLOD !== -1) {
      if (this.previousLOD !== decisionLOD) {
        lodChangeFromPrevious = '1';
        this.lodChangeCount++;
      }
    }
    if (decisionLOD !== -1) {
      this.previousLOD = decisionLOD;
    }
    
    // Track FPS for stability calculation (keep last 30)
    if (deviceFPS > 0) {
      this.fpsHistory.push(deviceFPS);
      if (this.fpsHistory.length > 30) {
        this.fpsHistory.shift();
      }
    }
    
    // Format features
    const row = [
      timestamp,
      entryObj.frustumCoverage !== undefined ? entryObj.frustumCoverage.toFixed(4) : '',
      entryObj.occlusionRatio !== undefined ? entryObj.occlusionRatio.toFixed(4) : '',
      entryObj.meanVisibleDistance !== undefined ? entryObj.meanVisibleDistance.toFixed(2) : '',
      entryObj.deviceFPS !== undefined ? entryObj.deviceFPS.toFixed(2) : '',
      entryObj.deviceCPULoad !== undefined ? entryObj.deviceCPULoad.toFixed(4) : '',
      entryObj.deviceGPULoad !== undefined ? entryObj.deviceGPULoad.toFixed(2) : '',
      lastTrajectory.velocity !== undefined ? lastTrajectory.velocity.toFixed(4) : '',
      lastTrajectory.acceleration !== undefined ? lastTrajectory.acceleration.toFixed(4) : '',
      lastTrajectory.angularVelocity !== undefined ? lastTrajectory.angularVelocity.toFixed(4) : '',
      decisionLOD,
      decisionTiles,
      entryObj.latencyMs !== undefined ? entryObj.latencyMs.toFixed(2) : '',
      entryObj.fpsAfterDecision !== undefined ? entryObj.fpsAfterDecision.toFixed(2) : '',
      currentMemoryMB.toFixed(2),                    // Current tile memory
      jsHeapMemoryMB.toFixed(2),                     // Current JS heap memory
      this.memoryPeakMB.toFixed(2),                  // Peak tile memory
      this.jsHeapPeakMB.toFixed(2),                  // Peak JS heap memory
      fpsImprovement,                                 // FPS improvement %
      qualityScore,                                   // Quality score (0-1)
      fpsPerGPUPercent,                              // FPS per GPU %
      fpsPerMemoryMB,                                // FPS per MB
      inAcceptableRange,                             // In acceptable FPS range
      lodChangeFromPrevious                          // LOD changed from previous
    ].join(',');
    
    this.logEntries.push(row);
    
    // If buffer is full, trigger download
    if (this.logEntries.length >= this.maxBufferSize) {
      const result = this.flush();
      // Notify user if callback available (will be set from index.js)
      if (this.onFlushCallback) {
        this.onFlushCallback(result);
      }
    }
  }

  /**
   * Flush logs to downloadable CSV file
   */
  flush() {
    if (this.logEntries.length === 0) return;
    
    const entryCount = this.logEntries.length - 1; // Subtract header
    console.log(`📥 CSV Download: Exporting ${entryCount} entries to ${this.logFilePath}`);
    
    const csvContent = this.logEntries.join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = this.logFilePath;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
    
    // Clear buffer (keep header)
    if (this.initialized && this.logEntries.length > 0) {
      const header = this.logEntries[0];
      this.logEntries = [header];
    }
    
    // Return notification text
    return `CSV downloaded: ${entryCount} entries`;
  }

  /**
   * Clear all logged entries
   */
  clear() {
    this.logEntries = [];
    this.initialized = false;
    this.resetMemoryPeaks();
  }

  /**
   * Get memory peak statistics
   * @returns {Object} Memory peak stats
   */
  getMemoryPeaks() {
    return {
      tileMemoryPeakMB: this.memoryPeakMB,
      jsHeapPeakMB: this.jsHeapPeakMB,
      totalPeakMB: this.memoryPeakMB + this.jsHeapPeakMB
    };
  }

  /**
   * Reset memory peaks (useful for new test sessions)
   */
  resetMemoryPeaks() {
    this.memoryPeakMB = 0;
    this.jsHeapPeakMB = 0;
    this.previousLOD = null;
    this.lodChangeCount = 0;
    this.fpsHistory = [];
  }
  
  /**
   * Calculate FPS stability (coefficient of variation)
   * Lower value = more stable FPS
   * @returns {number} Coefficient of variation, or 0 if insufficient data
   */
  getFPSStability() {
    if (this.fpsHistory.length < 2) return 0;
    
    const mean = this.fpsHistory.reduce((sum, val) => sum + val, 0) / this.fpsHistory.length;
    if (mean === 0) return 0;
    
    const variance = this.fpsHistory.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / this.fpsHistory.length;
    const stdDev = Math.sqrt(variance);
    
    // Coefficient of variation = stdDev / mean
    return stdDev / mean;
  }
  
  /**
   * Get adaptation statistics
   * @returns {Object} Adaptation stats
   */
  getAdaptationStats() {
    const totalEntries = this.logEntries.length - 1; // Subtract header
    return {
      lodChangeCount: this.lodChangeCount,
      adaptationFrequency: totalEntries > 0 ? (this.lodChangeCount / totalEntries) : 0,
      fpsStability: this.getFPSStability()
    };
  }

  /**
   * Get current log entries as CSV string
   * @returns {string} CSV content
   */
  getCSV() {
    this.initializeCSV();
    return this.logEntries.join('\n');
  }

  /**
   * Get log statistics
   * @returns {Object} Statistics object
   */
  getStats() {
    if (this.logEntries.length <= 1) {
      return { totalEntries: 0 };
    }
    
    // Parse entries (skip header)
    const entries = this.logEntries.slice(1);
    const lodCounts = {};
    
    entries.forEach(entry => {
      const parts = entry.split(',');
      if (parts.length > 9) {
        const lod = parseInt(parts[9]);
        lodCounts[lod] = (lodCounts[lod] || 0) + 1;
      }
    });
    
    return {
      totalEntries: entries.length,
      lodDistribution: lodCounts
    };
  }
}

