"""
GPU Metrics Module for macOS
Queries system GPU usage using macOS-specific tools
"""

import subprocess
import re
import platform
import logging

logger = logging.getLogger(__name__)

def get_macos_gpu_usage():
    """
    Get GPU usage on macOS using system commands
    Returns GPU usage percentage (0-100) or None if unavailable
    """
    if platform.system() != 'Darwin':  # Darwin = macOS
        return None
    
    # Method 1: Try powermetrics (most accurate, but requires sudo)
    gpu_usage = try_powermetrics()
    if gpu_usage is not None:
        return gpu_usage
    
    # Method 2: Try iostat (fallback)
    gpu_usage = try_iostat()
    if gpu_usage is not None:
        return gpu_usage
    
    # Method 3: Try Activity Monitor data via system_profiler
    gpu_usage = try_system_profiler()
    if gpu_usage is not None:
        return gpu_usage
    
    return None

def try_powermetrics():
    """Try to get GPU usage from powermetrics"""
    try:
        # Note: powermetrics requires sudo, but we can try without it first
        # If it fails, we'll fall back to other methods
        result = subprocess.run(
            ['powermetrics', '--samplers', 'gpu_power', '-i', '100', '-n', '1'],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3
        )
        
        if result.returncode == 0:
            # Parse GPU utilization from output
            # Format: "GPU Power: X.XX W" or "GPU Utilization: XX%"
            lines = result.stdout.split('\n')
            for line in lines:
                # Look for GPU utilization percentage
                if 'GPU' in line and ('utilization' in line.lower() or '%' in line):
                    match = re.search(r'(\d+\.?\d*)%', line)
                    if match:
                        usage = float(match.group(1))
                        return min(100, max(0, usage))
                
                # Look for GPU power (can infer usage)
                if 'GPU' in line and 'Power' in line:
                    match = re.search(r'(\d+\.?\d*)\s*W', line)
                    if match:
                        power = float(match.group(1))
                        # Estimate usage from power (rough approximation)
                        # Typical Mac GPU: 0-15W idle, 15-50W under load
                        usage = min(100, (power / 50.0) * 100)
                        return usage
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError, PermissionError):
        pass
    
    return None

def try_iostat():
    """Try to get system load from iostat (indirect GPU indicator)"""
    try:
        result = subprocess.run(
            ['iostat', '-w', '1', '-c', '2'],
            capture_output=True,
            text=True,
            timeout=3
        )
        
        if result.returncode == 0:
            # iostat doesn't directly show GPU, but high system load
            # can indicate GPU activity. This is a very rough estimate.
            # We'll use this as a last resort
            lines = result.stdout.split('\n')
            # Parse CPU idle time - lower idle = higher system load
            # This is not accurate for GPU, but better than nothing
            for line in lines:
                if 'id' in line.lower() and '%' in line:
                    match = re.search(r'(\d+\.?\d*)', line)
                    if match:
                        idle = float(match.group(1))
                        # Invert: lower idle = higher load
                        load = 100 - idle
                        # Scale down since this is system-wide, not GPU-specific
                        return load * 0.3  # Rough estimate
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    
    return None

def try_system_profiler():
    """Try to get GPU info from system_profiler"""
    try:
        result = subprocess.run(
            ['system_profiler', 'SPDisplaysDataType'],
            capture_output=True,
            text=True,
            timeout=2
        )
        
        if result.returncode == 0:
            # system_profiler shows GPU info but not real-time usage
            # This is mainly for GPU identification, not usage tracking
            # We'll return None to use other methods
            pass
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    
    return None

def get_gpu_info():
    """Get GPU information (name, model)"""
    if platform.system() != 'Darwin':
        return None
    
    try:
        result = subprocess.run(
            ['system_profiler', 'SPDisplaysDataType'],
            capture_output=True,
            text=True,
            timeout=2
        )
        
        if result.returncode == 0:
            # Parse GPU name
            lines = result.stdout.split('\n')
            for i, line in enumerate(lines):
                if 'Chipset Model' in line or 'Device ID' in line:
                    # Get next line or parse current line
                    match = re.search(r':\s*(.+)', line)
                    if match:
                        return match.group(1).strip()
    except:
        pass
    
    return None

