#!/usr/bin/env python
import argparse
import subprocess
import sys
import os
import time
import socket
import requests
import shutil
from pathlib import Path

from config import (
    MODEL_NAME, VLLM_MAX_MODEL_LEN, VLLM_TENSOR_PARALLEL_SIZE,
    VLLM_ENABLE_EXPERT_PARALLEL, VLLM_KV_CACHE_DTYPE
)

def is_port_in_use(port=8000):
    """Check if the specified port is in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def wait_for_server(port=8000, timeout=180, check_interval=20):
    """Wait for the server to be available on the specified port"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if is_port_in_use(port):
            # Try to connect to the API endpoint
            try:
                response = requests.get(f"http://0.0.0.0:{port}/v1/models")
                if response.status_code == 200:
                    print(f"✅ Server is ready and responding on port {port}!")
                    return True
            except requests.RequestException:
                pass  # Keep waiting
        
        # Still waiting
        print(f"Waiting for server to be ready on port {port}... ({int(time.time() - start_time)}s)")
        time.sleep(check_interval)
    
    print(f"❌ Server didn't respond within {timeout} seconds.")
    return False

def get_cuda_device_count():
    """
    Detect the number of available CUDA devices.
    Returns 0 if CUDA is not available.
    """
    # Try to import torch if available
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except ImportError:
        pass
    
    # Fallback: use nvidia-smi to check
    try:
        # Check if nvidia-smi is available
        if shutil.which("nvidia-smi") is not None:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True
            )
            return int(result.stdout.strip())
    except (subprocess.SubprocessError, ValueError):
        pass
    
    # Default to 0 if no CUDA devices could be detected
    return 0

def serve_vllm(
    model_name=None, 
    max_model_len=None, 
    gpu_memory_util=0.95, 
    tensor_parallel_size=None,
    enable_expert_parallel=None,
    kv_cache_dtype=None
):
    """
    Starts a vLLM server with the specified configuration.
    Server always runs in the background.
    
    Args:
        model_name (str): The name or path of the model to serve
        max_model_len (int): Maximum model context length
        gpu_memory_util (float): GPU memory utilization (0.0 to 1.0)
        tensor_parallel_size (int): Number of GPUs to use for tensor parallelism
        enable_expert_parallel (bool): Enable expert parallelism for MoE models
        kv_cache_dtype (str): Data type for KV cache ("auto", "fp8", "fp16", etc.)
    """
    # Count available CUDA devices
    cuda_device_count = get_cuda_device_count()
    
    # Use values from config if not provided
    model_name = model_name or MODEL_NAME
    max_model_len = max_model_len or VLLM_MAX_MODEL_LEN
    
    # If tensor_parallel_size is not provided, use the value from config, 
    # or all available CUDA devices if the config value is higher than available
    if tensor_parallel_size is None:
        if VLLM_TENSOR_PARALLEL_SIZE > cuda_device_count and cuda_device_count > 0:
            tensor_parallel_size = cuda_device_count
            print(f"Warning: Configured tensor_parallel_size ({VLLM_TENSOR_PARALLEL_SIZE}) exceeds available CUDA devices ({cuda_device_count}). Using {cuda_device_count} devices.")
        else:
            tensor_parallel_size = VLLM_TENSOR_PARALLEL_SIZE
    else:
        # If explicitly provided, warn if it exceeds available devices
        if tensor_parallel_size > cuda_device_count and cuda_device_count > 0:
            print(f"Warning: Requested tensor_parallel_size ({tensor_parallel_size}) exceeds available CUDA devices ({cuda_device_count}). This might cause errors.")
    
    # Ensure tensor_parallel_size is at least 1
    if tensor_parallel_size <= 0:
        tensor_parallel_size = 1
    
    # Use config values for other parameters if not provided
    enable_expert_parallel = enable_expert_parallel if enable_expert_parallel is not None else VLLM_ENABLE_EXPERT_PARALLEL
    kv_cache_dtype = kv_cache_dtype or VLLM_KV_CACHE_DTYPE
    
    cmd = [
        "vllm", "serve",
        model_name,
        "--max_model_len", str(max_model_len),
        "--disable-mm-preprocessor-cache",
        "--dtype", "auto",
        "--trust-remote-code",
        "--gpu-memory-utilization", str(gpu_memory_util),
        "--enforce-eager",  # Enforce eager execution mode for better token handling
        "--tensor-parallel-size", str(tensor_parallel_size),  # Set tensor parallel size
        "--kv-cache-dtype", kv_cache_dtype,  # Set KV cache data type
    ]
    
    # Add expert parallelism if enabled
    if enable_expert_parallel:
        cmd.append("--enable-expert-parallel")
    
    print(f"Starting vLLM server with command: {' '.join(cmd)}")
    print(f"Model: {model_name}")
    print(f"Context length: {max_model_len}")
    print(f"GPU memory utilization: {gpu_memory_util * 100:.0f}%")
    print(f"Available CUDA devices: {cuda_device_count}")
    print(f"Tensor parallelism: {tensor_parallel_size} GPU(s)")
    print(f"Expert parallelism: {'Enabled' if enable_expert_parallel else 'Disabled'}")
    print(f"KV cache data type: {kv_cache_dtype}")
    print(f"Using eager execution mode for better token handling")
    
    try:
        # Check if a server is already running
        if is_port_in_use(8000):
            print(f"⚠️ Port 8000 is already in use. A server might already be running.")
            return
            
        # Always run in background
        process = subprocess.Popen(cmd, 
                                stdout=subprocess.DEVNULL, 
                                stderr=subprocess.DEVNULL, 
                                start_new_session=True)
        print(f"Server is starting in the background with PID: {process.pid}")
        print(f"To stop the server: kill {process.pid}")
        
        # Wait for the server to be ready
        wait_for_server()
        
        return process.pid
    except Exception as e:
        print(f"Error starting vLLM server: {e}")
        sys.exit(1)

def main():
    # Get the number of CUDA devices
    cuda_device_count = get_cuda_device_count()
    
    parser = argparse.ArgumentParser(description="Start a vLLM server with specified configuration")
    parser.add_argument("--model", type=str, default=None, 
                        help=f"Model name or path (default: {MODEL_NAME})")
    parser.add_argument("--max-model-len", type=int, default=None, 
                        help=f"Maximum model context length (default: {VLLM_MAX_MODEL_LEN})")
    parser.add_argument("--gpu-util", type=float, default=0.95,
                        help="GPU memory utilization, from 0.0 to 1.0 (default: 0.95)")
    
    # Add tensor parallelism help text based on CUDA availability
    if cuda_device_count > 0:
        tp_help = f"Number of GPUs to use for tensor parallelism (default: {VLLM_TENSOR_PARALLEL_SIZE}, max available: {cuda_device_count})"
    else:
        tp_help = f"Number of GPUs to use for tensor parallelism (default: {VLLM_TENSOR_PARALLEL_SIZE}, no CUDA devices detected)"
    
    parser.add_argument("--tensor-parallel-size", type=int, default=None, help=tp_help)
    
    parser.add_argument("--enable-expert-parallel", action="store_true",
                        help=f"Enable expert parallelism for MoE models (default: {'Enabled' if VLLM_ENABLE_EXPERT_PARALLEL else 'Disabled'})")
    parser.add_argument("--kv-cache-dtype", type=str, default=None,
                        help=f"Data type for KV cache: auto, fp8, fp16, bf16, etc. (default: {VLLM_KV_CACHE_DTYPE})")
    parser.add_argument("--use-all-gpus", action="store_true",
                        help=f"Use all available GPUs for tensor parallelism (detected: {cuda_device_count})")
    
    args = parser.parse_args()
    
    # Override tensor_parallel_size if --use-all-gpus is specified
    if args.use_all_gpus and cuda_device_count > 0:
        args.tensor_parallel_size = cuda_device_count
        print(f"Using all {cuda_device_count} available CUDA devices for tensor parallelism")
    
    serve_vllm(
        model_name=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_util=args.gpu_util,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_expert_parallel=args.enable_expert_parallel,
        kv_cache_dtype=args.kv_cache_dtype
    )

if __name__ == "__main__":
    main() 