#!/usr/bin/env python
import argparse
import subprocess
import sys
import os
import time
import socket
import requests
from pathlib import Path

from config import MODEL_NAME, VLLM_MAX_MODEL_LEN

def is_port_in_use(port=8000):
    """Check if the specified port is in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def wait_for_server(port=8000, timeout=60, check_interval=2):
    """Wait for the server to be available on the specified port"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if is_port_in_use(port):
            # Try to connect to the API endpoint
            try:
                response = requests.get(f"http://localhost:{port}/v1/models")
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

def serve_vllm(model_name=None, max_model_len=None):
    """
    Starts a vLLM server with the specified configuration.
    Server always runs in the background.
    
    Args:
        model_name (str): The name or path of the model to serve
        max_model_len (int): Maximum model context length
    """
    # Use values from config if not provided
    model_name = model_name or MODEL_NAME
    max_model_len = max_model_len or VLLM_MAX_MODEL_LEN
    
    cmd = [
        "vllm", "serve",
        model_name,
        "--max_model_len", str(max_model_len),
        "--disable-mm-preprocessor-cache",
        "--dtype", "auto",
        "--trust-remote-code"
    ]
    
    print(f"Starting vLLM server with command: {' '.join(cmd)}")
    print(f"Model: {model_name}")
    print(f"Context length: {max_model_len}")
    
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
    
    args = parser.parse_args()
    
    serve_vllm(
        model_name=args.model,
        max_model_len=args.max_model_len,
    )

if __name__ == "__main__":
    main() 