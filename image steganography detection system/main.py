"""
Image Steganography Detector - Main Entry Point
"""


import sys
import os
from gui.app import StegoDetectorApp
# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    app = StegoDetectorApp()
    app.mainloop()
