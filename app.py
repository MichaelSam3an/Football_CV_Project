import os
import json
import time
from flask import Flask, request, jsonify, Response, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(BASE_DIR, 'input')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'output')

app.config['INPUT_FOLDER'] = INPUT_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Global state tracker for the pipeline progress
processing_status = {
    "progress": 0, 
    "status": "idle",
    "metrics_ready": False
}

@app.route('/analyze', methods=['POST'])
def analyze_video():
    if 'file' not in request.files:
        return jsonify({"error": "No file payload found under the key 'file'"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
        
    filename = secure_filename(file.filename)
    input_path = os.path.join(app.config['INPUT_FOLDER'], filename)
    file.save(input_path)
    
    # Reset tracking state for this run
    global processing_status
    processing_status = {
        "progress": 0, 
        "status": "processing",
        "metrics_ready": False
    }
    
    # -------------------------------------------------------------------------
    # INTEGRATION TIP:
    # Inside your main.py/analytics.py execution, your loop updates the progress:
    #   processing_status["progress"] = int((current_frame / total_frames) * 100)
    #
    # When everything is completely done, compile your analytics into a file:
    #   with open('output/analytics_data.json', 'w') as f:
    #       json.dump(your_final_metrics_dictionary, f)
    #   processing_status["status"] = "completed"
    #   processing_status["metrics_ready"] = True
    # -------------------------------------------------------------------------
    
    return jsonify({
        "status": "started",
        "message": "Processing initialized. Monitor stream via /progress-stream"
    }), 202


@app.route('/progress-stream', methods=['GET'])
def progress_stream():
    """
    Continuous stream endpoint for the developer's progress bars.
    """
    def generate_progress_events():
        global processing_status
        while True:
            yield f"data: {json.dumps(processing_status)}\n\n"
            
            if processing_status['status'] in ['completed', 'failed']:
                break
                
            time.sleep(0.5)
            
    return Response(generate_progress_events(), mimetype='text/event-stream')


@app.route('/analytics-data', methods=['GET'])
def get_analytics_data():
    """
    Endpoint for the web developer to fetch the team stats, player speeds, 
    and tracking metrics to build their UI charts/dashboards.
    """
    analytics_path = os.path.join(app.config['OUTPUT_FOLDER'], 'analytics_data.json')
    
    if not os.path.exists(analytics_path):
        return jsonify({"error": "Analytics data not generated yet or file missing"}), 404
        
    try:
        with open(analytics_path, 'r') as f:
            data = json.load(f)
        return jsonify(data), 200
    except Exception as e:
        return jsonify({"error": "Failed to read analytics file", "details": str(e)}), 500


@app.route('/output/<filename>', methods=['GET'])
def serve_output(filename):
    """
    Endpoint to retrieve the final processed video file.
    """
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)


if __name__ == '__main__':
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    app.run(debug=True, port=5000, threaded=True)
