"""
SpectraAML // Application Launcher
Starts the Flask server and opens the browser interface.
"""

import sys
import webbrowser
import threading
import time
from app import app

def open_browser():
    time.sleep(1.2)
    url = "http://127.0.0.1:5000"
    print(f"\n=======================================================")
    print(f" SPECTRA // AML Graph Neural Network Platform")
    print(f" Running at: {url}")
    print(f"=======================================================\n")
    webbrowser.open(url)

if __name__ == '__main__':
    threading.Thread(target=open_browser, daemon=True).start()
    app.run(host='127.0.0.1', port=5000, debug=False)
