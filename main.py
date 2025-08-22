"""
Song Finder System - Main Application
A system that finds songs based on speech/text input using n-grams matching
"""

import sys
import os
from PyQt5.QtWidgets import QApplication

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.ui.main_window import MainWindow

def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("Song Finder")
    app.setApplicationVersion("1.0")
    
    # Set application icon if available
    icon_path = os.path.join(os.path.dirname(__file__), 'assets', 'icons', 'logo.png')
    if os.path.exists(icon_path):
        from PyQt5.QtGui import QIcon
        app.setWindowIcon(QIcon(icon_path))
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Start event loop
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()