import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.web.flask_app import create_app

if __name__ == "__main__":
    config_name = os.environ.get("CONFIG_PRESET", "balanced")
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8000))
    debug = os.environ.get("DEBUG", "false").lower() == "true"
    
    app = create_app(config_name)
    
    print(f"Starting Flask app with {config_name} configuration...")
    print(f"Server running on http://{host}:{port}")
    
    app.run(host=host, port=port, debug=debug)
