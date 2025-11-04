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
      'viewportVelocity',
      'viewportAcceleration',
      'viewportAngularVelocity',
      'decisionLOD',
      'decisionTiles',
      'latencyMs',
      'fpsAfterDecision',
      'memoryMB'
    ].join(',');
    
    this.logEntries.push(header);
    this.initialized = true;
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
    
    // Format features
    const row = [
      timestamp,
      entryObj.frustumCoverage !== undefined ? entryObj.frustumCoverage.toFixed(4) : '',
      entryObj.occlusionRatio !== undefined ? entryObj.occlusionRatio.toFixed(4) : '',
      entryObj.meanVisibleDistance !== undefined ? entryObj.meanVisibleDistance.toFixed(2) : '',
      entryObj.deviceFPS !== undefined ? entryObj.deviceFPS.toFixed(2) : '',
      entryObj.deviceCPULoad !== undefined ? entryObj.deviceCPULoad.toFixed(4) : '',
      lastTrajectory.velocity !== undefined ? lastTrajectory.velocity.toFixed(4) : '',
      lastTrajectory.acceleration !== undefined ? lastTrajectory.acceleration.toFixed(4) : '',
      lastTrajectory.angularVelocity !== undefined ? lastTrajectory.angularVelocity.toFixed(4) : '',
      decisionLOD,
      decisionTiles,
      entryObj.latencyMs !== undefined ? entryObj.latencyMs.toFixed(2) : '',
      entryObj.fpsAfterDecision !== undefined ? entryObj.fpsAfterDecision.toFixed(2) : '',
      entryObj.memoryMB !== undefined ? entryObj.memoryMB.toFixed(2) : ''
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

