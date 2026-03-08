#!/usr/bin/env python3
"""Quick start script for CAV demo."""

import subprocess
import sys
from pathlib import Path


def main():
    """Run the CAV demo."""
    print("🧠 Concept Activation Vectors (CAV) Demo")
    print("=" * 50)
    
    # Check if streamlit is available
    try:
        import streamlit
        print("✅ Streamlit is available")
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
    
    # Run the demo
    demo_path = Path(__file__).parent / "demo" / "streamlit_app.py"
    
    if not demo_path.exists():
        print(f"❌ Demo file not found: {demo_path}")
        return
    
    print("🚀 Launching Streamlit demo...")
    print("The demo will open in your browser.")
    print("Press Ctrl+C to stop the demo.")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(demo_path),
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Demo stopped. Thank you for trying CAV!")


if __name__ == "__main__":
    main()
