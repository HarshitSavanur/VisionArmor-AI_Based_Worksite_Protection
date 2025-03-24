from flask import Flask, render_template, request, redirect, url_for
import os
import shutil
from werkzeug.utils import secure_filename
from ultralytics import YOLO

app = Flask(__name__)

# Configure upload and output directories
UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/output"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER

# Load YOLO model
model = YOLO("models/best.pt")

def get_latest_prediction():
    """Find the latest prediction folder created by YOLOv8."""
    predict_dirs = [d for d in os.listdir("runs/detect/") if "predict" in d]
    if not predict_dirs:
        return None
    latest_folder = max(predict_dirs, key=lambda x: os.path.getctime(os.path.join("runs/detect/", x)))
    return os.path.join("runs/detect/", latest_folder)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            # Run YOLO prediction
            model.predict(file_path, save=True)

            # Get latest prediction folder
            latest_pred_folder = get_latest_prediction()
            if not latest_pred_folder:
                return "Error: No prediction results found."

            # Find processed image
            detected_images = [f for f in os.listdir(latest_pred_folder) if f.endswith((".jpg", ".png"))]
            if not detected_images:
                return "Error: No output image found."

            detected_image_path = os.path.join(latest_pred_folder, detected_images[0])
            output_filename = f"output_{filename}"
            output_path = os.path.join(app.config["OUTPUT_FOLDER"], output_filename)

            # Move detected image to static/output/
            shutil.move(detected_image_path, output_path)

            return render_template("index.html", uploaded_file=filename, output_file=output_filename)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=False)
