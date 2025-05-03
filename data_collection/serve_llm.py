#!/usr/bin/env python
import argparse
import subprocess
import sys
import os
import time
import socket
import requests
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
    # Use values from config if not provided
    model_name = model_name or MODEL_NAME
    max_model_len = max_model_len or VLLM_MAX_MODEL_LEN
    tensor_parallel_size = tensor_parallel_size or VLLM_TENSOR_PARALLEL_SIZE
    enable_expert_parallel = enable_expert_parallel if enable_expert_parallel is not None else VLLM_ENABLE_EXPERT_PARALLEL
    kv_cache_dtype = kv_cache_dtype or VLLM_KV_CACHE_DTYPE
    
    # Check tensor parallel size
    if tensor_parallel_size <= 0:
        print(f"Warning: Invalid tensor parallel size {tensor_parallel_size}. Using default value of 1.")
        tensor_parallel_size = 1
    
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
    parser = argparse.ArgumentParser(description="Start a vLLM server with specified configuration")
    parser.add_argument("--model", type=str, default=None, 
                        help=f"Model name or path (default: {MODEL_NAME})")
    parser.add_argument("--max-model-len", type=int, default=None, 
                        help=f"Maximum model context length (default: {VLLM_MAX_MODEL_LEN})")
    parser.add_argument("--gpu-util", type=float, default=0.95,
                        help="GPU memory utilization, from 0.0 to 1.0 (default: 0.95)")
    parser.add_argument("--tensor-parallel-size", type=int, default=None,
                        help=f"Number of GPUs to use for tensor parallelism (default: {VLLM_TENSOR_PARALLEL_SIZE})")
    parser.add_argument("--enable-expert-parallel", action="store_true",
                        help=f"Enable expert parallelism for MoE models (default: {'Enabled' if VLLM_ENABLE_EXPERT_PARALLEL else 'Disabled'})")
    parser.add_argument("--kv-cache-dtype", type=str, default=None,
                        help=f"Data type for KV cache: auto, fp8, fp16, bf16, etc. (default: {VLLM_KV_CACHE_DTYPE})")
    
    args = parser.parse_args()
    
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