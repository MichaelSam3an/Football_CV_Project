import os
import json
import time
import threading

from flask import Flask, request, jsonify, Response, send_from_directory
from werkzeug.utils import secure_filename

from main import process_video

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(BASE_DIR, "input")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "output")

app.config["INPUT_FOLDER"] = INPUT_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER

processing_status = {
    "progress": 0,
    "status": "idle",
    "metrics_ready": False,
    "message": "",
    "output_video": "",
    "analytics_file": ""
}

status_lock = threading.Lock()


def update_processing_status(progress=None, status=None, metrics_ready=None, message=None,
                             output_video=None, analytics_file=None):
    with status_lock:
        if progress is not None:
            processing_status["progress"] = int(progress)
        if status is not None:
            processing_status["status"] = status
        if metrics_ready is not None:
            processing_status["metrics_ready"] = metrics_ready
        if message is not None:
            processing_status["message"] = message
        if output_video is not None:
            processing_status["output_video"] = output_video
        if analytics_file is not None:
            processing_status["analytics_file"] = analytics_file


def progress_callback(progress):
    update_processing_status(
        progress=progress,
        status="processing",
        message=f"Processing... {progress}%"
    )


def run_pipeline_job(input_path: str, output_path: str) -> None:
    try:
        update_processing_status(
            progress=0,
            status="processing",
            metrics_ready=False,
            message="Starting processing",
            output_video=f"/output/{os.path.basename(output_path)}",
            analytics_file=""
        )

        result = process_video(
            input_video_path=input_path,
            output_video_path=output_path,
            progress_callback=progress_callback,
            chunk_size=75
        )

        if not isinstance(result, dict) or result.get("status") != "completed":
            update_processing_status(
                status="failed",
                metrics_ready=False,
                message=result.get("message", "Pipeline failed") if isinstance(result, dict) else "Pipeline failed"
            )
            return

        analytics_path = result.get("analytics_file", "")
        output_video_path = result.get("output_video", output_path)

        update_processing_status(
            progress=100,
            status="completed",
            metrics_ready=os.path.exists(analytics_path),
            message="Processing completed",
            output_video=f"/output/{os.path.basename(output_video_path)}",
            analytics_file=analytics_path
        )

    except Exception as e:
        update_processing_status(
            status="failed",
            metrics_ready=False,
            message=str(e)
        )


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

    update_processing_status(
        progress=0,
        status="processing",
        metrics_ready=False,
        message="Queued",
        output_video=f"/output/{output_name}",
        analytics_file=""
    )

    worker = threading.Thread(
        target=run_pipeline_job,
        args=(input_path, output_path),
        daemon=True
    )
    worker.start()

    return jsonify({
        "status": "started",
        "message": "Processing initialized. Monitor progress via /progress-stream or /status",
        "output_video": f"/output/{output_name}"
    }), 202


@app.route("/status", methods=["GET"])
def status():
    with status_lock:
        return jsonify(processing_status), 200


@app.route("/progress-stream", methods=["GET"])
def progress_stream():
    def generate_progress_events():
        last_payload = None
        while True:
            with status_lock:
                payload = dict(processing_status)

            if payload != last_payload:
                yield f"data: {json.dumps(payload)}\n\n"
                last_payload = payload

            if payload["status"] in ["completed", "failed"]:
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
