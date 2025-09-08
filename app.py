from flask import Flask, redirect, session
from config import SECRET_KEY
import os

# Import blueprints
from auth.routes import auth_bp
from data.routes import data_bp
from templates.html_templates import INDEX_HTML

# Create Flask app
app = Flask(__name__)
app.secret_key = SECRET_KEY

# Register blueprints
app.register_blueprint(auth_bp)
app.register_blueprint(data_bp)

@app.route('/')
def index():
    if 'user' not in session:
        return redirect('/login')
    return INDEX_HTML

# Check if Excel file exists
if __name__ == '__main__':
    from config import EXCEL_FILE_PATH
    
    if not os.path.exists(EXCEL_FILE_PATH):
        print(f"Warning: Excel file not found at {EXCEL_FILE_PATH}")
        print("Please update the EXCEL_FILE_PATH variable in config.py")
        
    app.run(debug=True, port=5000)
