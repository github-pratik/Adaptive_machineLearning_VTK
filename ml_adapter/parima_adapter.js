/**
 * PARIMA Adapter
 * Handles communication with PARIMA backend API
 */

export class PARIMAAdapter {
  constructor(apiUrl = 'http://localhost:5000/api/parima/predict') {
    this.apiUrl = apiUrl;
    this.modelLoaded = false;
    this.lastError = null;
    this.requestTimeout = 5000; // 5 seconds
  }

  /**
   * Initialize adapter and check if model is available
   * @returns {Promise<boolean>} True if model is loaded
   */
  async init() {
    try {
      // Check health endpoint
      const healthUrl = this.apiUrl.replace('/api/parima/predict', '/health');
      const response = await fetch(healthUrl, {
        method: 'GET',
        timeout: this.requestTimeout
      });
      
      if (response.ok) {
        const data = await response.json();
        this.modelLoaded = data.model_loaded || false;
        return this.modelLoaded;
      }
    } catch (error) {
      console.warn('PARIMA adapter: Health check failed', error);
      this.modelLoaded = false;
      this.lastError = error.message;
      return false;
    }
    
    return false;
  }

  /**
   * Predict decision from features
   * @param {Object} features - Feature dictionary
   * @returns {Promise<Object>} Decision object { lod: number, tiles?: string[] }
   */
  async predictDecision(features) {
    if (!this.modelLoaded) {
      // Try to initialize once more
      await this.init();
      if (!this.modelLoaded) {
        // Return fallback decision
        console.warn('PARIMA model not available, using fallback LOD');
        return { lod: 1 }; // Default to medium LOD
      }
    }

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.requestTimeout);

      const response = await fetch(this.apiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(features),
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();

      if (data.success) {
        this.lastError = null;
        return data.decision || { lod: 1 };
      } else {
        // Model returned error, use fallback
        console.warn('PARIMA prediction failed:', data.error);
        this.lastError = data.error;
        return data.decision || { lod: 1 }; // Use fallback from server if provided
      }

    } catch (error) {
      if (error.name === 'AbortError') {
        console.warn('PARIMA prediction timeout, using fallback');
        this.lastError = 'Request timeout';
      } else {
        console.error('PARIMA prediction error:', error);
        this.lastError = error.message;
      }
      
      // Fallback decision
      return { lod: 1 };
    }
  }

  /**
   * Check if model is currently available
   * @returns {boolean}
   */
  isModelAvailable() {
    return this.modelLoaded;
  }

  /**
   * Get last error message
   * @returns {string|null}
   */
  getLastError() {
    return this.lastError;
  }
}

