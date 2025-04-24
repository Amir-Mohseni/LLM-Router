#!/usr/bin/env python
import argparse
import subprocess
import sys
import os

from data_collection.config import (
    MODEL_NAME,
    VLLM_MAX_MODEL_LEN
)

def serve_vllm(model_name=None, 
               max_model_len=None,
               trust_remote_code=True):
    """
    Starts a vLLM server with the specified configuration.
    Server always runs in the background.
    
    Args:
        model_name (str): The name or path of the model to serve
        max_model_len (int): Maximum model context length
        host (str): Host to bind the server to
        port (int): Port to run the server on
        trust_remote_code (bool): Whether to trust remote code from the model repository
    """
    # Use values from config if not provided
    model_name = model_name or MODEL_NAME
    max_model_len = max_model_len or VLLM_MAX_MODEL_LEN
    
    cmd = [
        "vllm", "serve",
        model_name,
        "--max_model_len", str(max_model_len),
        "--disable-mm-preprocessor-cache",
        "--dtype", "auto"
    ]
    
    # Add trust remote code flag if specified
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    
    print(f"Starting vLLM server with command: {' '.join(cmd)}")
    print(f"Model: {model_name}")
    print(f"Context length: {max_model_len}")
    
    try:
        # Always run in background
        process = subprocess.Popen(cmd, 
                                stdout=subprocess.DEVNULL, 
                                stderr=subprocess.DEVNULL, 
                                start_new_session=True)
        print(f"Server is running in the background with PID: {process.pid}")
        print(f"To stop the server: kill {process.pid}")
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
        trust_remote_code=args.trust_remote_code
    )

if __name__ == "__main__":
    main() 