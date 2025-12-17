from flask import Flask, send_from_directory
from flask_cors import CORS
from api import api_bp, init_data
import os

app = Flask(__name__, static_folder='static')
CORS(app)

app.register_blueprint(api_bp)


@app.route('/')
def serve_frontend():
    return send_from_directory('static', 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)


if __name__ == '__main__':
    init_data()
    print('Starting server at http://127.0.0.1:5000')
    app.run(debug=True)
