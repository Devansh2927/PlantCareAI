from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os, traceback, time, json, csv

# local imports
from utils.preprocess import preprocess_image
from models.predict_real import predict_real

BASE_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config['UPLOAD_FOLDER'] = UPLOAD_DIR
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

# load class mapping once
with open(os.path.join(BASE_DIR, "class_indices.json")) as f:
    class_indices = json.load(f)
inv_map = {v: k for k, v in class_indices.items()}

# Simple advice map (extend with your class names)
ADVICE_MAP = {
    "Apple___Apple_scab": {
        "advice": "Apple scab: remove infected leaves, rake fallen debris, improve airflow; avoid overhead watering; consider a fungicide labeled for apple scab if severe.",
        "advice_html": "<ul><li>Remove infected leaves</li><li>Rake fallen debris</li><li>Improve airflow around tree</li><li>Avoid overhead watering</li><li>Use a labeled fungicide if needed</li></ul>"
    },
    "Healthy": {
        "advice": "Plant appears healthy. Continue regular watering and monitor for pests or spots.",
        "advice_html": "<p>Plant appears healthy. Continue regular watering and monitor for pests or spots.</p>"
    }
}

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/predict", methods=["POST"])
def predict():
    import datetime, numpy as np
    print("=== /predict called ===", time.ctime())
    try:
        if 'file' not in request.files:
            return jsonify({"error":"no file part"}), 400
        f = request.files['file']
        if f.filename == "":
            return jsonify({"error":"no selected file"}), 400

        filename = secure_filename(f.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(save_path)
        print("Saved to:", save_path)

        # preprocess and predict
        img_arr = preprocess_image(save_path, target_size=(224,224))
        pred_probs, gradcam_filename = predict_real(img_arr)  # predict_real should return (probs, gradcam_filename) or (probs, None)

        # normalize probs to 1D numpy array
        probs = pred_probs[0] if getattr(pred_probs, "ndim", 1) == 2 else pred_probs
        probs = np.asarray(probs).astype(float)
        top_idx = np.argsort(probs)[-3:][::-1]
        top_k = [{"label": inv_map.get(int(i), "unknown"), "prob": float(probs[int(i)])} for i in top_idx]

        top1_idx = int(np.argmax(probs))
        top1_label = inv_map.get(top1_idx, "unknown")
        top1_conf = float(probs[top1_idx])

        preprocess_info = {"shape": list(img_arr.shape), "min": float(img_arr.min()), "max": float(img_arr.max())}
        meta = {"model": "MobileNetV2_transfer_v1", "timestamp": datetime.datetime.utcnow().isoformat() + "Z"}

        # Logging: append a CSV line
        os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
        log_path = os.path.join(BASE_DIR, "logs", "predictions.csv")
        header = ["timestamp","filename","prediction","confidence","top1","top2","top3"]
        if not os.path.exists(log_path):
            with open(log_path, "w", newline="", encoding="utf8") as lf:
                writer = csv.writer(lf)
                writer.writerow(header)
        with open(log_path, "a", newline="", encoding="utf8") as lf:
            writer = csv.writer(lf)
            writer.writerow([meta["timestamp"], filename, top1_label, top1_conf,
                             top_k[0]["label"], top_k[1]["label"], top_k[2]["label"]])

        # Save low-confidence images for review
        if top1_conf < 0.6:
            os.makedirs(os.path.join(BASE_DIR, "retrain"), exist_ok=True)
            low_path = os.path.join(BASE_DIR, "retrain", f"{datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%S')}_{filename}")
            try:
                from shutil import copyfile
                copyfile(save_path, low_path)
            except Exception:
                pass

        response = {
            "prediction": top1_label,
            "confidence": top1_conf,
            "top_k": top_k,
            "preprocess": preprocess_info,
            "explain": {"gradcam": gradcam_filename} if gradcam_filename else {},
            "meta": meta,
            "filename": filename
        }
        return jsonify(response)
    except Exception as e:
        tb = traceback.format_exc()
        print("Error in /predict:\n", tb)
        return jsonify({"error":"server_error", "detail": str(e), "trace": tb}), 500

@app.route("/advice", methods=["GET"])
def advice():
    label = request.args.get("label", "")
    # strip trailing confidence suffix if present (e.g., "Label:1")
    if ":" in label:
        label = label.split(":", 1)[0]
    info = ADVICE_MAP.get(label)
    if info:
        return jsonify({"label": label, "advice": info["advice"], "advice_html": info.get("advice_html")})
    return jsonify({"label": label, "advice": "No specific advice available for this label."})

@app.route("/report", methods=["POST"])
def report():
    try:
        data = request.get_json(force=True)
        os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
        report_path = os.path.join(BASE_DIR, "logs", "reports.csv")
        header = ["timestamp","filename","predicted","confidence","top1","top2","top3","comment"]
        if not os.path.exists(report_path):
            with open(report_path, "w", newline="", encoding="utf8") as rf:
                writer = csv.writer(rf)
                writer.writerow(header)
        top_k = data.get("top_k", [])
        top1 = top_k[0].get("label","") if len(top_k)>0 else ""
        top2 = top_k[1].get("label","") if len(top_k)>1 else ""
        top3 = top_k[2].get("label","") if len(top_k)>2 else ""
        with open(report_path, "a", newline="", encoding="utf8") as rf:
            writer = csv.writer(rf)
            writer.writerow([data.get("timestamp"), data.get("filename"), data.get("predicted"),
                             data.get("confidence"), top1, top2, top3, data.get("comment","")])
        return jsonify({"status":"ok"})
    except Exception as e:
        tb = traceback.format_exc()
        print("Error in /report:\n", tb)
        return jsonify({"status":"error", "detail": str(e)}), 500

if __name__ == '__main__':
    # bind to localhost for local testing
    app.run(host='127.0.0.1', port=5000, debug=True)
