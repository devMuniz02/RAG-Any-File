import webview
import threading
import time
import socket
import sys
import os

# Add the src directory to the path so we can import the Flask app
sys.path.insert(0, os.path.dirname(__file__))

from src.app import app

def is_port_in_use(port):
    """Check if a port is in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def start_server():
    """Start the Flask server in a separate thread."""
    try:
        app.run(host='127.0.0.1', port=8000, debug=False, use_reloader=False)
    except Exception as e:
        print(f"Server error: {e}")

def on_window_closed():
    """Callback when the webview window is closed."""
    print("Window closed, shutting down server...")
    # Note: Flask doesn't have a built-in way to stop the server gracefully from another thread
    # In a production app, you might want to use a more advanced server like gunicorn or uvicorn
    # For now, we'll just exit the process
    os._exit(0)

def main():
    # Check if port 8000 is already in use
    if is_port_in_use(8000):
        print("Error: Port 8000 is already in use. Please close any other server running on this port.")
        return

    # Start the server in a background thread
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

    # Wait a bit for the server to start
    time.sleep(2)

    # Verify server is running
    if not is_port_in_use(8000):
        print("Error: Failed to start server on port 8000")
        return

    # Set user data directory for persistent storage
    user_data_dir = os.path.join(os.path.expanduser("~"), "RAG-Any-File", "webview_data")

    # Create the webview window
    window = webview.create_window(
        'RAG-Any-File',
        'http://127.0.0.1:8000',
        width=1200,
        height=800,
        resizable=True,
        frameless=False
    )

    # Set the close callback
    window.events.closed += on_window_closed

    # Start the webview application
    webview.start()

if __name__ == '__main__':
    main()