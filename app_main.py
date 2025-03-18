# main.py

from app_utils_db import initialize_db
from app_ui import create_ui

if __name__ == "__main__":
    # Create and launch the interface
    initialize_db()
    app = create_ui()
    app.launch()
