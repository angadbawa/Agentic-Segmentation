#!/usr/bin/env python3
"""
Startup script for Gradio application
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.web.gradio_app import create_gradio_app

if __name__ == "__main__":
    config_name = os.environ.get("CONFIG_PRESET", "balanced")
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 7860))
    share = os.environ.get("SHARE", "false").lower() == "true"
    debug = os.environ.get("DEBUG", "false").lower() == "true"
    
    demo = create_gradio_app(config_name)
    
    print(f"Starting Gradio app with {config_name} configuration...")
    print(f"Server running on http://{host}:{port}")
    
    demo.launch(
        server_name=host,
        server_port=port,
        share=share,
        debug=debug,
        show_error=True
    )
