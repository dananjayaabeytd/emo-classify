"""Script to run the FastAPI server."""

import argparse
import uvicorn
from pathlib import Path

from src.api.app import create_app


def main():
    """Main function to run the API server."""
    parser = argparse.ArgumentParser(description="Run emotion classification API server")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--threshold", type=float, default=0.35, help="Classification threshold")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    # Create app
    app = create_app(
        model_checkpoint_path=Path(args.checkpoint),
        device=args.device,
        threshold=args.threshold,
    )
    
    # Run server
    print(f"Starting API server on {args.host}:{args.port}")
    print(f"Docs available at http://{args.host}:{args.port}/docs")
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
