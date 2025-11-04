/**
 * Tile Manager for PARIMA Tile-Based Streaming
 * Manages loading, unloading, and merging of VTP tiles
 */

import vtkXMLPolyDataReader from '@kitware/vtk.js/IO/XML/XMLPolyDataReader';
import vtkAppendPolyData from '@kitware/vtk.js/Filters/General/AppendPolyData';
import vtkPolyData from '@kitware/vtk.js/Common/DataModel/PolyData';

export class TileManager {
  constructor(config = {}) {
    this.basePath = config.basePath || './tiles';
    this.lodLevels = config.lodLevels || ['lod0', 'lod1', 'lod2'];
    this.currentModel = null;
    this.loadedTiles = new Map(); // Map<lod, Map<tileId, polyData>>
    this.tileActors = new Map(); // Map<tileId, actor>
    this.renderer = null;
    this.currentLOD = 1; // Default to medium LOD
    this.pendingLoads = new Set();
  }

  /**
   * Initialize tile manager with renderer
   * @param {Object} renderer - VTK.js renderer
   */
  initialize(renderer) {
    this.renderer = renderer;
  }

  /**
   * Set current model name (determines tile directory)
   * @param {string} modelName - Name of the model/dataset
   */
  setModel(modelName) {
    if (this.currentModel === modelName) return;
    
    // Clear existing tiles
    this.clearAllTiles();
    this.currentModel = modelName;
  }

  /**
   * Get tile file path
   * @param {number} lod - Level of detail (0, 1, 2, ...)
   * @param {number} x - Tile X coordinate
   * @param {number} y - Tile Y coordinate
   * @returns {string} Tile file path
   */
  getTilePath(lod, x, y) {
    const lodName = this.lodLevels[lod] || `lod${lod}`;
    return `${this.basePath}/${this.currentModel}/${lodName}/tile_${x}_${y}.vtp`;
  }

  /**
   * Load a single tile
   * @param {number} lod - Level of detail
   * @param {number} x - Tile X coordinate
   * @param {number} y - Tile Y coordinate
   * @returns {Promise<vtkPolyData>} Loaded poly data
   */
  async loadTile(lod, x, y) {
    const tileId = `${lod}_${x}_${y}`;
    
    // Check if already loaded
    if (this.loadedTiles.has(lod)) {
      const lodTiles = this.loadedTiles.get(lod);
      if (lodTiles.has(tileId)) {
        return lodTiles.get(tileId);
      }
    }

    // Check if already loading
    if (this.pendingLoads.has(tileId)) {
      // Wait for existing load
      return new Promise((resolve) => {
        const checkInterval = setInterval(() => {
          const lodTiles = this.loadedTiles.get(lod);
          if (lodTiles && lodTiles.has(tileId)) {
            clearInterval(checkInterval);
            resolve(lodTiles.get(tileId));
          }
        }, 50);
        
        setTimeout(() => {
          clearInterval(checkInterval);
          resolve(null);
        }, 5000); // Timeout after 5 seconds
      });
    }

    this.pendingLoads.add(tileId);

    try {
      const tilePath = this.getTilePath(lod, x, y);
      
      // Skip fetch if model name is null (tiles not set up)
      if (!this.currentModel || this.currentModel === 'null') {
        this.pendingLoads.delete(tileId);
        return null;
      }
      
      const response = await fetch(tilePath, { 
        // Suppress error logging for 404s
        signal: AbortSignal.timeout(5000) // 5 second timeout
      }).catch(() => null); // Silently catch fetch errors
      
      if (!response || !response.ok) {
        // Silently handle missing tiles
        this.pendingLoads.delete(tileId);
        return null;
      }

      const arrayBuffer = await response.arrayBuffer();
      const reader = vtkXMLPolyDataReader.newInstance();
      reader.parseAsArrayBuffer(arrayBuffer);
      const polyData = reader.getOutputData(0);

      // Store in cache
      if (!this.loadedTiles.has(lod)) {
        this.loadedTiles.set(lod, new Map());
      }
      this.loadedTiles.get(lod).set(tileId, polyData);

      this.pendingLoads.delete(tileId);
      return polyData;

    } catch (error) {
      // Silently handle errors - tiles not implemented yet
      // Only log if it's not a network/404 error
      if (error && error.name !== 'AbortError' && !error.message?.includes('404')) {
        // Only log unexpected errors, not 404s
      }
      this.pendingLoads.delete(tileId);
      return null;
    }
  }

  /**
   * Load tiles based on PARIMA decision
   * @param {Object} decision - Decision object { lod: number, tiles?: string[] }
   * @returns {Promise<vtkPolyData>} Merged poly data from loaded tiles
   */
  async loadTiles(decision) {
    // Early return if tiles are not set up (model name is null)
    // This prevents 404 fetch errors in console
    if (!this.currentModel || this.currentModel === 'null') {
      return null; // Silently skip tile loading
    }
    
    const targetLOD = decision.lod !== undefined ? decision.lod : this.currentLOD;
    
    // If decision specifies exact tiles
    if (decision.tiles && Array.isArray(decision.tiles) && decision.tiles.length > 0) {
      return await this.loadSpecificTiles(targetLOD, decision.tiles);
    }
    
    // Otherwise, load based on LOD (simplified: load all tiles for target LOD)
    // In a full implementation, this would compute visible tiles based on frustum
    return await this.loadLODTiles(targetLOD);
  }

  /**
   * Load specific tiles by ID
   * @param {number} lod - Level of detail
   * @param {string[]} tileIds - Array of tile IDs (e.g., ["0_0", "0_1"])
   * @returns {Promise<vtkPolyData>} Merged poly data
   */
  async loadSpecificTiles(lod, tileIds) {
    const tilePromises = tileIds.map(tileId => {
      const [x, y] = tileId.split('_').map(Number);
      return this.loadTile(lod, x, y);
    });

    const loadedTiles = await Promise.all(tilePromises);
    const validTiles = loadedTiles.filter(tile => tile !== null);

    if (validTiles.length === 0) {
      return null;
    }

    return this.mergeTiles(validTiles);
  }

  /**
   * Load all tiles for a given LOD
   * For now, this is a simplified implementation
   * In production, this would compute visible tiles based on camera frustum
   * @param {number} lod - Level of detail
   * @returns {Promise<vtkPolyData>} Merged poly data
   */
  async loadLODTiles(lod) {
    // Simplified: try to load a few default tiles (e.g., center tiles)
    // In production, compute visible tiles from frustum
    const defaultTiles = [
      [0, 0], [0, 1], [1, 0], [1, 1] // Try common center tiles
    ];

    const tilePromises = defaultTiles.map(([x, y]) => this.loadTile(lod, x, y));
    const loadedTiles = await Promise.all(tilePromises);
    const validTiles = loadedTiles.filter(tile => tile !== null);

    if (validTiles.length === 0) {
      // console.warn(`No tiles found for LOD ${lod}`); // Suppressed: tiles not implemented yet
      return null;
    }

    return this.mergeTiles(validTiles);
  }

  /**
   * Merge multiple poly data tiles into single poly data
   * @param {vtkPolyData[]} tiles - Array of poly data tiles
   * @returns {vtkPolyData} Merged poly data
   */
  mergeTiles(tiles) {
    if (tiles.length === 0) return null;
    if (tiles.length === 1) return tiles[0];

    try {
      const appendFilter = vtkAppendPolyData.newInstance();
      
      tiles.forEach(tile => {
        appendFilter.addInputData(tile);
      });

      appendFilter.update();
      return appendFilter.getOutputData(0);
    } catch (error) {
      // console.error('Failed to merge tiles:', error); // Suppressed: tiles not implemented yet
      // Fallback: return first tile
      return tiles[0];
    }
  }

  /**
   * Apply loaded tiles to renderer
   * @param {vtkPolyData} mergedData - Merged poly data from tiles
   * @param {Object} mapper - VTK.js mapper
   * @param {Object} actor - VTK.js actor
   */
  applyTiles(mergedData, mapper, actor) {
    if (!mergedData) return;

    try {
      mapper.setInputData(mergedData);
      
      // Add actor to renderer if not already added
      if (this.renderer && actor && !actor.getMapper()) {
        this.renderer.addActor(actor);
      }

      return true;
    } catch (error) {
      // console.error('Failed to apply tiles:', error); // Suppressed: tiles not implemented yet
      return false;
    }
  }

  /**
   * Unload tiles for a specific LOD
   * @param {number} lod - Level of detail to unload
   */
  unloadLOD(lod) {
    if (this.loadedTiles.has(lod)) {
      this.loadedTiles.get(lod).clear();
      this.loadedTiles.delete(lod);
    }
  }

  /**
   * Clear all loaded tiles
   */
  clearAllTiles() {
    this.loadedTiles.clear();
    this.tileActors.clear();
    this.pendingLoads.clear();
  }

  /**
   * Get current LOD
   * @returns {number}
   */
  getCurrentLOD() {
    return this.currentLOD;
  }

  /**
   * Set current LOD
   * @param {number} lod
   */
  setCurrentLOD(lod) {
    this.currentLOD = lod;
  }

  /**
   * Get memory usage estimate
   * @returns {number} Estimated memory in MB
   */
  getMemoryUsage() {
    let totalPoints = 0;
    
    this.loadedTiles.forEach((lodTiles) => {
      lodTiles.forEach((polyData) => {
        const points = polyData.getPoints();
        if (points) {
          totalPoints += points.getNumberOfPoints();
        }
      });
    });

    // Rough estimate: 3 floats per point * 4 bytes * loaded tiles
    const estimatedMB = (totalPoints * 3 * 4) / (1024 * 1024);
    return estimatedMB;
  }
}

