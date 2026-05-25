import os
import json
import time
import sys
import subprocess
import threading

from flask import Flask, request, jsonify, Response, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(BASE_DIR, "input")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "output")

app.config["INPUT_FOLDER"] = INPUT_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER

processing_status = {
    "progress": 0,
    "status": "idle",
    "metrics_ready": False
}

def run_pipeline_job(input_path: str, output_path: str) -> None:
    global processing_status

    try:
        processing_status["progress"] = 5
        processing_status["status"] = "processing"
        processing_status["metrics_ready"] = False

        # Run the existing CV pipeline in main.py
        result = subprocess.run(
            [
                sys.executable,
                os.path.join(BASE_DIR, "main.py"),
                "--input_video", input_path,
                "--output_video", output_path,
                "--chunk_size", "75",
            ],
            cwd=BASE_DIR,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print("Pipeline failed.")
            print(result.stdout)
            print(result.stderr)
            processing_status["status"] = "failed"
            processing_status["progress"] = 0
            return

        processing_status["progress"] = 95

        analytics_path = os.path.join(app.config["OUTPUT_FOLDER"], "analytics_data.json")
        processing_status["metrics_ready"] = os.path.exists(analytics_path)
        processing_status["progress"] = 100
        processing_status["status"] = "completed"

    except Exception as e:
        print(f"Pipeline exception: {e}")
        processing_status["status"] = "failed"
        processing_status["metrics_ready"] = False

@app.route("/analyze", methods=["POST"])
def analyze_video():
    if "file" not in request.files:
        return jsonify({"error": "No file payload found under the key 'file'"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    os.makedirs(INPUT_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    filename = secure_filename(file.filename)
    input_path = os.path.join(app.config["INPUT_FOLDER"], filename)
    output_name = f"{os.path.splitext(filename)[0]}_output.mp4"
    output_path = os.path.join(app.config["OUTPUT_FOLDER"], output_name)

    file.save(input_path)

    global processing_status
    processing_status = {
        "progress": 0,
        "status": "processing",
        "metrics_ready": False
    }

    worker = threading.Thread(
        target=run_pipeline_job,
        args=(input_path, output_path),
        daemon=True
    )
    worker.start()

    return jsonify({
        "status": "started",
        "message": "Processing initialized. Monitor stream via /progress-stream",
        "output_video": f"/output/{output_name}"
    }), 202

@app.route("/progress-stream", methods=["GET"])
def progress_stream():
    def generate_progress_events():
        while True:
            yield f"data: {json.dumps(processing_status)}\n\n"
            if processing_status["status"] in ["completed", "failed"]:
                break
            time.sleep(0.5)

    return Response(generate_progress_events(), mimetype="text/event-stream")

@app.route("/analytics-data", methods=["GET"])
def get_analytics_data():
    analytics_path = os.path.join(app.config["OUTPUT_FOLDER"], "analytics_data.json")

    if not os.path.exists(analytics_path):
        return jsonify({"error": "Analytics data not generated yet or file missing"}), 404

    try:
        with open(analytics_path, "r") as f:
            data = json.load(f)
        return jsonify(data), 200
    except Exception as e:
        return jsonify({"error": "Failed to read analytics file", "details": str(e)}), 500

@app.route("/output/<filename>", methods=["GET"])
def serve_output(filename):
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename)

if __name__ == "__main__":
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    app.run(debug=True, port=5000, threaded=True)
