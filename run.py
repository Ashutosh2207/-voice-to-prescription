#!/usr/bin/env python
"""Entry point to run the Voice-to-Prescription system."""
import subprocess
import sys
import os
import time
import threading
import webbrowser
from pathlib import Path


def run_api():
    """Run FastAPI server."""
    os.chdir(Path(__file__).parent)
    subprocess.run([
        sys.executable, "-m", "uvicorn", 
        "app.api:app", 
        "--host", "0.0.0.0", 
        "--port", "8000",
        "--reload"
    ])


def run_streamlit():
    """Run Streamlit app."""
    os.chdir(Path(__file__).parent)
    time.sleep(3)  # Wait for API to start
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        "streamlit_app.py",
        "--server.port", "8501",
        "--server.headless", "true"
    ])


def install_deps():
    """Install dependencies."""
    print("Installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
    
    # Download spaCy model
    print("Downloading spaCy model...")
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
    
    print("Dependencies installed!")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Voice-to-Prescription System")
    parser.add_argument("command", choices=["install", "api", "streamlit", "all"], 
                       help="Command to run")
    args = parser.parse_args()
    
    if args.command == "install":
        install_deps()
    
    elif args.command == "api":
        run_api()
    
    elif args.command == "streamlit":
        run_streamlit()
    
    elif args.command == "all":
        print("Starting Voice-to-Prescription System...")
        print("API will be at: http://localhost:8000")
        print("Streamlit UI will be at: http://localhost:8501")
        
        # Start API in background thread
        api_thread = threading.Thread(target=run_api, daemon=True)
        api_thread.start()
        
        # Start Streamlit (blocks)
        run_streamlit()


if __name__ == "__main__":
    main()