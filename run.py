"""SIG-ID Master Runner.

Usage:
    python run.py             # Start the Web Dashboard at http://127.0.0.1:8000
    python run.py --gui       # Launch the PyQt6 Desktop GUI
    python run.py --test      # Run the test suite
"""
import sys
from pathlib import Path

# Add src/ to sys.path so sigid is always importable
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if __name__ == "__main__":
    args = sys.argv[1:]

    if "--gui" in args or "-g" in args:
        print("[SIG-ID] Launching Desktop GUI (PyQt6)...")
        from sigid.gui.main_window import main
        main()
    elif "--test" in args or "-t" in args:
        print("[SIG-ID] Running test suite with pytest...")
        import pytest
        sys.exit(pytest.main(args[1:]))
    else:
        print("=" * 65)
        print("  SIG-ID Cyberdeck — Signal Intelligence & Demodulation Platform")
        print("=" * 65)
        print("  Web Server:  http://127.0.0.1:8000")
        print("  Swagger API: http://127.0.0.1:8000/docs")
        print("  (Press CTRL+C to stop the server)")
        print("=" * 65)

        import uvicorn
        uvicorn.run(
            "sigid.web.app:app",
            host="127.0.0.1",
            port=8000,
            reload=True,
            app_dir=str(SRC_DIR),
        )
