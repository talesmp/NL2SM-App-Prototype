# main.py

# from ui import create_ui
from ui import create_new_ui
from db_utils import initialize_db

if __name__ == "__main__":
    # Create and launch the interface
    # app = create_ui()
    initialize_db()
    app = create_new_ui()
    app.launch()
