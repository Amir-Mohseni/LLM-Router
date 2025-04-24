#!/usr/bin/env python
import argparse
import subprocess
import sys
import os

from data_collection.config import (
    MODEL_NAME,
    VLLM_HOST,
    VLLM_PORT,
    VLLM_MODEL_IMPL,
    VLLM_MAX_MODEL_LEN
)

def serve_vllm(model_name=None, 
               model_impl=None, 
               max_model_len=None,
               host=None,
               port=None):
    """
    Starts a vLLM server with the specified configuration.
    
    Args:
        model_name (str): The name or path of the model to serve
        model_impl (str): The model implementation (transformers or vllm)
        max_model_len (int): Maximum model context length
        host (str): Host to bind the server to
        port (int): Port to run the server on
    """
    # Use values from config if not provided
    model_name = model_name or MODEL_NAME
    model_impl = model_impl or VLLM_MODEL_IMPL
    max_model_len = max_model_len or VLLM_MAX_MODEL_LEN
    host = host or VLLM_HOST
    port = port or VLLM_PORT
    
    cmd = [
        "vllm", "serve",
        model_name,
        "--model-impl", model_impl,
        "--max_model_len", str(max_model_len),
        "--host", host,
        "--port", str(port)
    ]
    
    print(f"Starting vLLM server with command: {' '.join(cmd)}")
    print(f"Model: {model_name}")
    print(f"Context length: {max_model_len}")
    print(f"Server will be available at: http://{host if host != '0.0.0.0' else 'localhost'}:{port}/v1")
    
    try:
        # Run the process without capturing output so it's shown directly in the terminal
        process = subprocess.Popen(cmd)
        process.wait()
    except KeyboardInterrupt:
        print("\nShutting down vLLM server...")
        process.terminate()
        sys.exit(0)
    except Exception as e:
        print(f"Error starting vLLM server: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Start a vLLM server with specified configuration")
    parser.add_argument("--model", type=str, default=None, 
                        help=f"Model name or path (default: {MODEL_NAME})")
    parser.add_argument("--model-impl", type=str, default=None, 
                        help=f"Model implementation (transformers or vllm) (default: {VLLM_MODEL_IMPL})")
    parser.add_argument("--max-model-len", type=int, default=None, 
                        help=f"Maximum model context length (default: {VLLM_MAX_MODEL_LEN})")
    parser.add_argument("--host", type=str, default=None, 
                        help=f"Host to bind the server to (default: {VLLM_HOST})")
    parser.add_argument("--port", type=int, default=None, 
                        help=f"Port to run the server on (default: {VLLM_PORT})")
    
    args = parser.parse_args()
    
    serve_vllm(
        model_name=args.model,
        model_impl=args.model_impl,
        max_model_len=args.max_model_len,
        host=args.host,
        port=args.port
    )

if __name__ == "__main__":
    main() 