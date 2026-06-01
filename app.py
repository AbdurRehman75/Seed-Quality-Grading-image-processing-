import os
import uuid
import sqlite3
import cv2
import numpy as np
import json
from datetime import datetime
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_file
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from services.ml_predictor import predict_grade


# ── Paths ─────────────────────────────────────────────────────────────────────
APP_DIR       = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR    = os.path.join(APP_DIR, "static", "uploads")
PROCESSED_DIR = os.path.join(APP_DIR, "static", "processed")
DB_PATH       = os.path.join(APP_DIR, "database.db")
ALLOWED_EXT   = {"png", "jpg", "jpeg", "webp", "tiff"}

os.makedirs(UPLOAD_DIR,    exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
app.secret_key = os.environ.get("SECRET_KEY", "seedsense-uaf-2026-secure-key")


# ── Database ──────────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    cur  = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            fullname      TEXT    NOT NULL,
            email         TEXT    NOT NULL UNIQUE,
            password_hash TEXT    NOT NULL,
            created_at    DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS grading_results (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id        INTEGER NOT NULL,
            batch_id       TEXT    NOT NULL,
            seed_type      TEXT    NOT NULL DEFAULT 'wheat',
            upload_path    TEXT    NOT NULL,
            processed_path TEXT    NOT NULL,
            grade          TEXT    NOT NULL,
            confidence     REAL    NOT NULL,
            seed_count     INTEGER NOT NULL,
            defect_ratio   REAL    NOT NULL,
            purity         REAL    NOT NULL DEFAULT 0,
            broken_pct     REAL    NOT NULL DEFAULT 0,
            foreign_pct    REAL    NOT NULL DEFAULT 0,
            color_score    REAL    NOT NULL DEFAULT 0,
            texture_score  REAL    NOT NULL DEFAULT 0,
            shape_score    REAL    NOT NULL DEFAULT 0,
            notes          TEXT,
            created_at     DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    conn.commit()
    conn.close()


init_db()


# ── Auth ──────────────────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            flash("Please login to continue.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def generate_batch_id():
    now = datetime.now()
    return f"UAF-{now.strftime('%Y%m')}-{uuid.uuid4().hex[:6].upper()}"


# ══════════════════════════════════════════════════════════════════════════════
#  SEEDSENSE GRADING ENGINE v2.0
#  Multi-channel analysis: Shape + Color + Texture + Fragment Detection
# ══════════════════════════════════════════════════════════════════════════════

SEED_PROFILES = {
    "wheat": {
        "aspect_ideal": (2.2, 4.8),
        "color_hsv_low":  np.array([8,  25,  60]),
        "color_hsv_high": np.array([38, 220, 240]),
        "min_area_frac": 0.00003,
        "name": "Wheat",
        "emoji": "🌾",
        "texture_thresh": 1200.0,
        "description": "Hard red winter wheat variety"
    },
    "rice": {
        "aspect_ideal": (2.8, 6.5),
        "color_hsv_low":  np.array([12,  8,  130]),
        "color_hsv_high": np.array([42, 130, 255]),
        "min_area_frac": 0.00002,
        "name": "Rice",
        "emoji": "🍚",
        "texture_thresh": 800.0,
        "description": "Long-grain paddy rice variety"
    },
    "corn": {
        "aspect_ideal": (0.9, 2.4),
        "color_hsv_low":  np.array([12, 70,  90]),
        "color_hsv_high": np.array([42, 255, 255]),
        "min_area_frac": 0.00006,
        "name": "Corn",
        "emoji": "🌽",
        "texture_thresh": 1800.0,
        "description": "Yellow dent corn variety"
    },
}


def preprocess_image(img):
    """Enhanced preprocessing with adaptive thresholding."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Normalize brightness
    gray = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # Try Otsu first
    _, thr_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Adaptive as fallback
    thr_adapt = cv2.adaptiveThreshold(blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3)

    # Combine for better detection
    thr = cv2.bitwise_or(thr_otsu, thr_adapt)

    kernel = np.ones((3, 3), np.uint8)
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN,  kernel, iterations=2)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=3)
    thr = cv2.dilate(thr, kernel, iterations=1)
    return gray, thr


def analyze_color(roi_bgr, profile):
    if roi_bgr is None or roi_bgr.size == 0:
        return 0.5
    hsv  = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, profile["color_hsv_low"], profile["color_hsv_high"])
    in_range = np.count_nonzero(mask) / max(1, mask.size)
    # Perfect = high ratio of pixels in ideal range
    return float(np.clip(1.0 - in_range, 0, 1))


def analyze_texture(roi_gray, profile):
    if roi_gray is None or roi_gray.size == 0:
        return 0.5
    lap      = cv2.Laplacian(roi_gray, cv2.CV_64F)
    variance = lap.var()
    score    = float(np.clip(variance / profile["texture_thresh"], 0, 1))
    return score


def analyze_shape(contour, profile):
    area      = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0 or area == 0:
        return 0.5

    circularity = (4 * np.pi * area) / (perimeter ** 2 + 1e-9)
    x, y, bw, bh = cv2.boundingRect(contour)
    aspect = max(bw, bh) / (min(bw, bh) + 1e-9)

    lo, hi = profile["aspect_ideal"]
    if lo <= aspect <= hi:
        aspect_score = 0.0
    elif aspect < lo:
        aspect_score = min(1.0, (lo - aspect) * 0.45)
    else:
        aspect_score = min(1.0, (aspect - hi) * 0.35)

    # Solidity (fill ratio)
    hull      = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity  = area / (hull_area + 1e-9)
    solid_score = max(0.0, 0.75 - solidity)

    circ_score = max(0.0, 0.55 - circularity) * 1.1

    return float(np.clip(aspect_score * 0.5 + circ_score * 0.3 + solid_score * 0.2, 0, 1))


def grade_seeds(image_bgr, seed_type="wheat"):
    profile  = SEED_PROFILES.get(seed_type, SEED_PROFILES["wheat"])
    original = image_bgr.copy()
    h, w     = image_bgr.shape[:2]

    gray, thr = preprocess_image(image_bgr)
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(250, int(h * w * profile["min_area_frac"]))
    kernels  = [c for c in contours if cv2.contourArea(c) >= min_area]

    if len(kernels) == 0:
        return {
            "grade": "C", "confidence": 0.25, "seed_count": 0, "defect_ratio": 1.0,
            "purity": 0.0, "broken_pct": 0.0, "foreign_pct": 0.0,
            "color_score": 1.0, "texture_score": 1.0, "shape_score": 1.0,
            "notes": "No seeds detected. Use a plain light-colored background and ensure proper lighting.",
            "processed": original,
        }

    areas        = [cv2.contourArea(c) for c in kernels]
    median_area  = float(np.median(areas))
    shape_scores, color_scores, texture_scores = [], [], []
    broken = 0
    foreign = 0

    for c in kernels:
        x, y, bw, bh = cv2.boundingRect(c)
        roi_bgr  = image_bgr[max(0,y):y+bh, max(0,x):x+bw]
        roi_gray = gray[max(0,y):y+bh, max(0,x):x+bw]

        s = analyze_shape(c, profile)
        c_score = analyze_color(roi_bgr, profile)
        t = analyze_texture(roi_gray, profile)

        shape_scores.append(s)
        color_scores.append(c_score)
        texture_scores.append(t)

        area = cv2.contourArea(c)
        if area < median_area * 0.45:
            broken += 1
        if c_score > 0.7 and s > 0.6:
            foreign += 1

        # Color-coded bounding boxes
        overall = s * 0.35 + c_score * 0.35 + t * 0.30
        if overall < 0.22:
            box_col = (39, 174, 96)    # green
            lbl = "A"
        elif overall < 0.48:
            box_col = (52, 152, 219)   # blue/yellow
            lbl = "B"
        else:
            box_col = (231, 76, 60)    # red
            lbl = "C"

        # Draw rounded-ish rectangle
        cv2.rectangle(original, (x, y), (x+bw, y+bh), box_col, 2)
        cv2.putText(original, lbl, (x+2, max(12, y-4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, box_col, 1, cv2.LINE_AA)

    n = len(kernels)
    avg_shape   = float(np.mean(shape_scores))
    avg_color   = float(np.mean(color_scores))
    avg_texture = float(np.mean(texture_scores))

    broken_ratio  = broken  / n if n >= 5 else 0.0
    foreign_ratio = foreign / n

    defect_ratio = float(np.clip(
        avg_shape   * 0.35 +
        avg_color   * 0.35 +
        avg_texture * 0.20 +
        broken_ratio  * 0.07 +
        foreign_ratio * 0.03,
        0, 1
    ))

    # Grade thresholds
    if defect_ratio < 0.20:
        grade = "A"
    elif defect_ratio < 0.42:
        grade = "B"
    else:
        grade = "C"

    confidence = float(np.clip(1.0 - defect_ratio * 0.75, 0.40, 0.98))

    # Derived stats
    purity      = round((1 - foreign_ratio) * 100, 1)
    broken_pct  = round(broken_ratio * 100, 1)
    foreign_pct = round(foreign_ratio * 100, 1)

    # Notes
    notes_list = []
    if avg_color > 0.42:
        notes_list.append(f"Color deviation detected — possible discoloration or foreign material in {profile['name']} batch.")
    if avg_texture > 0.38:
        notes_list.append("Texture irregularities found — surface cracks or damage present.")
    if avg_shape > 0.33:
        notes_list.append("Shape anomalies detected — shriveled or malformed kernels observed.")
    if n >= 5 and broken_ratio > 0.12:
        notes_list.append(f"High broken kernel proportion ({broken_pct}%) — check harvest handling.")
    if foreign_ratio > 0.05:
        notes_list.append(f"Foreign material detected ({foreign_pct}%) — cleaning recommended.")
    if not notes_list:
        notes_list.append(f"{profile['name']} batch shows excellent uniformity. Suitable for premium storage or direct planting.")

    # Overlay
    grade_col = {"A": (39,174,96), "B": (52,152,219), "C": (231,76,60)}[grade]
    overlay = original.copy()
    cv2.rectangle(overlay, (6, 6), (230, 72), (15, 15, 25), -1)
    cv2.addWeighted(overlay, 0.75, original, 0.25, 0, original)
    cv2.rectangle(original, (6, 6), (230, 72), grade_col, 1)
    cv2.putText(original, f"SeedSense  Grade: {grade}", (14, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, grade_col, 2, cv2.LINE_AA)
    cv2.putText(original, f"{n} seeds | {confidence*100:.0f}% conf | {defect_ratio*100:.1f}% defects",
                (14, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (190,190,190), 1, cv2.LINE_AA)

    return {
        "grade":         grade,
        "confidence":    confidence,
        "seed_count":    n,
        "defect_ratio":  defect_ratio,
        "purity":        purity,
        "broken_pct":    broken_pct,
        "foreign_pct":   foreign_pct,
        "color_score":   avg_color,
        "texture_score": avg_texture,
        "shape_score":   avg_shape,
        "notes":         " ".join(notes_list),
        "processed":     original,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    stats = {"total": 0, "a_count": 0, "b_count": 0, "c_count": 0}
    if "user_id" in session:
        conn = get_db()
        row  = conn.execute(
            "SELECT COUNT(*) as total, "
            "SUM(CASE WHEN grade='A' THEN 1 ELSE 0 END) as a_count, "
            "SUM(CASE WHEN grade='B' THEN 1 ELSE 0 END) as b_count, "
            "SUM(CASE WHEN grade='C' THEN 1 ELSE 0 END) as c_count "
            "FROM grading_results WHERE user_id=?",
            (session["user_id"],)
        ).fetchone()
        conn.close()
        if row and row["total"]:
            stats = dict(row)
    return render_template("index.html", stats=stats)


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        fullname = request.form.get("fullname", "").strip()
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm  = request.form.get("confirm_password", "")

        if not fullname or not email or not password:
            flash("All fields are required.", "danger")
            return redirect(url_for("register"))
        if password != confirm:
            flash("Passwords do not match.", "danger")
            return redirect(url_for("register"))
        if len(password) < 6:
            flash("Password must be at least 6 characters.", "danger")
            return redirect(url_for("register"))

        pw_hash = generate_password_hash(password)
        try:
            conn = get_db()
            conn.execute(
                "INSERT INTO users (fullname, email, password_hash) VALUES (?, ?, ?)",
                (fullname, email, pw_hash),
            )
            conn.commit()
            conn.close()
            flash("Account created! Please login.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Email already registered.", "warning")
            return redirect(url_for("register"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        conn.close()

        if not user or not check_password_hash(user["password_hash"], password):
            flash("Invalid email or password.", "danger")
            return redirect(url_for("login"))

        session["user_id"]   = user["id"]
        session["user_name"] = user["fullname"]
        session["user_email"]= user["email"]
        flash(f"Welcome back, {user['fullname']}!", "success")
        return redirect(url_for("index"))

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("index"))


@app.route("/analyze")
@login_required
def analyze():
    return render_template("analyze.html")


def draw_final_cnn_badge(image, grade, confidence):
    """
    Draws a secondary badge below the OpenCV badge on the processed image overlay
    showing the CNN grade classification or fallback status.
    """
    overlay = image.copy()
    cv2.rectangle(overlay, (6, 78), (230, 144), (15, 15, 25), -1)
    cv2.addWeighted(overlay, 0.75, image, 0.25, 0, image)
    
    if grade is not None and confidence is not None:
        grade_col = {"A": (39, 174, 96), "B": (52, 152, 219), "C": (231, 76, 60)}.get(grade, (120, 120, 120))
        cv2.rectangle(image, (6, 78), (230, 144), grade_col, 1)
        cv2.putText(image, f"CNN Grade: {grade}", (14, 102),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, grade_col, 2, cv2.LINE_AA)
        cv2.putText(image, f"Confidence: {confidence * 100:.1f}%", (14, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (190, 190, 190), 1, cv2.LINE_AA)
    else:
        cv2.rectangle(image, (6, 78), (230, 144), (120, 120, 120), 1)
        cv2.putText(image, "CNN: Not Trained", (14, 102),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (120, 120, 120), 1, cv2.LINE_AA)
        cv2.putText(image, "OpenCV Fallback Active", (14, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA)


@app.route("/training")
@login_required
def training_status():
    status_data = {}
    for stype in ["wheat", "rice", "corn"]:
        model_name = f"{stype}_grade_model.keras"
        model_path = os.path.join(APP_DIR, "models", model_name)
        metrics_path = os.path.join(APP_DIR, "training_reports", f"{stype}_metrics.json")
        
        has_model = os.path.exists(model_path)
        metrics = None
        has_curve = os.path.exists(os.path.join(APP_DIR, "training_reports", f"{stype}_accuracy_curve.png"))
        has_cm = os.path.exists(os.path.join(APP_DIR, "training_reports", f"{stype}_confusion_matrix.png"))
        
        if has_model and os.path.exists(metrics_path):
            try:
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)
            except Exception:
                pass
                
        last_updated = None
        if has_model:
            mtime = os.path.getmtime(model_path)
            last_updated = datetime.fromtimestamp(mtime).strftime("%d %b %Y, %I:%M %p")
            
        status_data[stype] = {
            "has_model": has_model,
            "last_updated": last_updated,
            "metrics": metrics,
            "has_curve": has_curve,
            "has_cm": has_cm,
            "curve_url": url_for("serve_training_report", filename=f"{stype}_accuracy_curve.png") if has_curve else None,
            "cm_url": url_for("serve_training_report", filename=f"{stype}_confusion_matrix.png") if has_cm else None
        }
        
    return render_template("training.html", status_data=status_data)


@app.route("/training_reports/<path:filename>")
@login_required
def serve_training_report(filename):
    reports_dir = os.path.join(APP_DIR, "training_reports")
    return send_file(os.path.join(reports_dir, filename))


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    seed_type = request.form.get("seed_type", "wheat").lower()
    if seed_type not in SEED_PROFILES:
        seed_type = "wheat"

    if "seed_image" not in request.files:
        flash("No file uploaded.", "danger")
        return redirect(url_for("analyze"))

    file = request.files["seed_image"]
    if file.filename == "":
        flash("No file selected.", "danger")
        return redirect(url_for("analyze"))

    if not allowed_file(file.filename):
        flash("Invalid file type. Use PNG / JPG / JPEG / WEBP / TIFF.", "danger")
        return redirect(url_for("analyze"))

    ext          = secure_filename(file.filename).rsplit(".", 1)[1].lower()
    uid          = uuid.uuid4().hex
    upload_name  = f"{uid}.{ext}"
    upload_path  = os.path.join(UPLOAD_DIR, upload_name)
    file.save(upload_path)

    img = cv2.imread(upload_path)
    if img is None:
        flash("Could not read image. Please try another file.", "danger")
        return redirect(url_for("analyze"))

    # Resize large images for speed
    max_dim = 1600
    mh, mw  = img.shape[:2]
    if max(mh, mw) > max_dim:
        scale = max_dim / max(mh, mw)
        img   = cv2.resize(img, (int(mw*scale), int(mh*scale)))

    result = grade_seeds(img, seed_type)

    # Run CNN grading if model is trained
    cnn_result = predict_grade(img, seed_type)
    
    if cnn_result is not None:
        final_grade = cnn_result["grade"]
        final_confidence = cnn_result["confidence"]
        notes = (
            f"Grading finalized using SeedSense CNN model (High-accuracy grading under controlled image conditions. "
            f"Accuracy depends on dataset quality, image clarity, lighting, and seed separation) with {final_confidence * 100:.1f}% confidence. "
            f"OpenCV analysis details: {result['notes']}"
        )
        draw_final_cnn_badge(result["processed"], final_grade, final_confidence)
    else:
        final_grade = result["grade"]
        final_confidence = result["confidence"]
        notes = (
            f"CNN model not trained yet. Safely fell back to OpenCV feature-based grading. "
            f"OpenCV analysis details: {result['notes']}"
        )
        draw_final_cnn_badge(result["processed"], None, None)

    processed_name = f"{uid}_proc.jpg"
    processed_path = os.path.join(PROCESSED_DIR, processed_name)
    cv2.imwrite(processed_path, result["processed"], [cv2.IMWRITE_JPEG_QUALITY, 92])

    batch_id = generate_batch_id()

    conn = get_db()
    cur  = conn.execute(
        """INSERT INTO grading_results
           (user_id, batch_id, seed_type, upload_path, processed_path,
            grade, confidence, seed_count, defect_ratio,
            purity, broken_pct, foreign_pct,
            color_score, texture_score, shape_score, notes)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            session["user_id"], batch_id, seed_type,
            f"static/uploads/{upload_name}",
            f"static/processed/{processed_name}",
            final_grade,
            round(final_confidence,    4),
            result["seed_count"],
            round(result["defect_ratio"],  4),
            round(result["purity"],        2),
            round(result["broken_pct"],    2),
            round(result["foreign_pct"],   2),
            round(result["color_score"],   4),
            round(result["texture_score"], 4),
            round(result["shape_score"],   4),
            notes,
        )
    )
    result_id = cur.lastrowid
    conn.commit()
    conn.close()

    return render_template(
        "result.html",
        result_id     = result_id,
        batch_id      = batch_id,
        seed_type     = SEED_PROFILES[seed_type]["name"],
        seed_emoji    = SEED_PROFILES[seed_type]["emoji"],
        uploaded      = f"/static/uploads/{upload_name}",
        processed     = f"/static/processed/{processed_name}",
        grade         = final_grade,
        confidence    = f"{final_confidence*100:.1f}",
        seed_count    = result["seed_count"],
        defect_ratio  = f"{result['defect_ratio']*100:.1f}",
        purity        = result["purity"],
        broken_pct    = result["broken_pct"],
        foreign_pct   = result["foreign_pct"],
        color_score   = f"{(1-result['color_score'])*100:.1f}",
        texture_score = f"{(1-result['texture_score'])*100:.1f}",
        shape_score   = f"{(1-result['shape_score'])*100:.1f}",
        notes         = notes,
        timestamp     = datetime.now().strftime("%d %b %Y, %I:%M %p"),
    )


@app.route("/dashboard")
@login_required
def dashboard():
    conn = get_db()
    stats = conn.execute(
        """SELECT COUNT(*) as total,
           SUM(CASE WHEN grade='A' THEN 1 ELSE 0 END) as a_count,
           SUM(CASE WHEN grade='B' THEN 1 ELSE 0 END) as b_count,
           SUM(CASE WHEN grade='C' THEN 1 ELSE 0 END) as c_count,
           AVG(confidence)*100 as avg_conf,
           AVG(seed_count) as avg_seeds,
           AVG(purity) as avg_purity
           FROM grading_results WHERE user_id=?""",
        (session["user_id"],)
    ).fetchone()

    type_breakdown = conn.execute(
        "SELECT seed_type, COUNT(*) as cnt FROM grading_results WHERE user_id=? GROUP BY seed_type",
        (session["user_id"],)
    ).fetchall()

    recent = conn.execute(
        "SELECT * FROM grading_results WHERE user_id=? ORDER BY created_at DESC LIMIT 8",
        (session["user_id"],)
    ).fetchall()

    trend = conn.execute(
        """SELECT strftime('%Y-%m', created_at) as month,
           COUNT(*) as cnt,
           SUM(CASE WHEN grade='A' THEN 1 ELSE 0 END) as a_cnt,
           AVG(confidence)*100 as avg_conf
           FROM grading_results WHERE user_id=?
           GROUP BY month ORDER BY month DESC LIMIT 6""",
        (session["user_id"],)
    ).fetchall()

    conn.close()
    return render_template(
        "dashboard.html",
        stats=dict(stats) if stats else {},
        type_breakdown=[dict(r) for r in type_breakdown],
        recent=[dict(r) for r in recent],
        trend=[dict(r) for r in trend],
    )


@app.route("/history")
@login_required
def history():
    page        = int(request.args.get("page", 1))
    per_page    = 12
    offset      = (page - 1) * per_page
    seed_filter = request.args.get("seed_type", "all")
    grade_filter= request.args.get("grade", "all")

    conn  = get_db()
    query = "SELECT * FROM grading_results WHERE user_id=?"
    args  = [session["user_id"]]

    if seed_filter != "all":
        query += " AND seed_type=?"; args.append(seed_filter)
    if grade_filter != "all":
        query += " AND grade=?"; args.append(grade_filter.upper())

    total = conn.execute(query.replace("SELECT *", "SELECT COUNT(*)"), args).fetchone()[0]
    rows  = conn.execute(
        query + " ORDER BY created_at DESC LIMIT ? OFFSET ?",
        args + [per_page, offset]
    ).fetchall()
    conn.close()

    return render_template(
        "history.html",
        results=[dict(r) for r in rows],
        page=page,
        total_pages=max(1, (total + per_page - 1) // per_page),
        seed_filter=seed_filter,
        grade_filter=grade_filter,
        total=total,
    )


@app.route("/result/<int:result_id>")
@login_required
def view_result(result_id):
    conn = get_db()
    row  = conn.execute(
        "SELECT * FROM grading_results WHERE id=? AND user_id=?",
        (result_id, session["user_id"])
    ).fetchone()
    conn.close()
    if not row:
        flash("Result not found.", "danger")
        return redirect(url_for("history"))
    r = dict(row)
    p = SEED_PROFILES.get(r["seed_type"], {})
    return render_template(
        "result.html",
        result_id     = r["id"],
        batch_id      = r.get("batch_id", f"UAF-{r['id']:06d}"),
        seed_type     = p.get("name", r["seed_type"].title()),
        seed_emoji    = p.get("emoji", "🌱"),
        uploaded      = "/" + r["upload_path"],
        processed     = "/" + r["processed_path"],
        grade         = r["grade"],
        confidence    = f"{r['confidence']*100:.1f}",
        seed_count    = r["seed_count"],
        defect_ratio  = f"{r['defect_ratio']*100:.1f}",
        purity        = r.get("purity", round((1-r["defect_ratio"])*100, 1)),
        broken_pct    = r.get("broken_pct", 0),
        foreign_pct   = r.get("foreign_pct", 0),
        color_score   = f"{(1-r['color_score'])*100:.1f}",
        texture_score = f"{(1-r['texture_score'])*100:.1f}",
        shape_score   = f"{(1-r['shape_score'])*100:.1f}",
        notes         = r["notes"],
        timestamp     = r["created_at"],
    )


@app.route("/report/<int:result_id>")
@login_required
def generate_report(result_id):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, HRFlowable
        from reportlab.lib.units import cm
        import io

        conn = get_db()
        r    = conn.execute(
            "SELECT gr.*, u.fullname, u.email FROM grading_results gr "
            "JOIN users u ON u.id=gr.user_id "
            "WHERE gr.id=? AND gr.user_id=?",
            (result_id, session["user_id"])
        ).fetchone()
        conn.close()

        if not r:
            flash("Result not found.", "danger")
            return redirect(url_for("history"))

        r   = dict(r)
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
                                rightMargin=2*cm, leftMargin=2*cm,
                                topMargin=2*cm, bottomMargin=2*cm)

        GREEN  = colors.HexColor("#1a6b3c")
        DARK   = colors.HexColor("#0d2b1a")
        LIGHT  = colors.HexColor("#f0fdf4")
        GRAY   = colors.HexColor("#6b7280")

        styles = getSampleStyleSheet()
        story  = []

        h1 = ParagraphStyle("h1", fontSize=22, textColor=DARK, fontName="Helvetica-Bold", spaceAfter=4)
        h2 = ParagraphStyle("h2", fontSize=13, textColor=GREEN, fontName="Helvetica-Bold", spaceAfter=6, spaceBefore=12)
        sub = ParagraphStyle("sub", fontSize=9, textColor=GRAY, spaceAfter=10)
        body = ParagraphStyle("body", fontSize=9, leading=14, textColor=colors.HexColor("#374151"))

        story.append(Paragraph("SeedSense — Seed Quality Report", h1))
        story.append(Paragraph("University of Agriculture Faisalabad · Final Year Project 2026", sub))
        story.append(HRFlowable(width="100%", thickness=1.5, color=GREEN))
        story.append(Spacer(1, 0.3*cm))

        grade_color = {"A": colors.HexColor("#16a34a"), "B": colors.HexColor("#d97706"), "C": colors.HexColor("#dc2626")}.get(r["grade"], colors.black)

        info = [
            ["Field", "Value"],
            ["Batch ID",       r.get("batch_id", f"#{r['id']:04d}")],
            ["Analyst",        r["fullname"]],
            ["Email",          r["email"]],
            ["Seed Type",      r["seed_type"].title()],
            ["Analysis Date",  r["created_at"]],
            ["Final Grade",    f"Grade {r['grade']}"],
            ["Confidence",     f"{r['confidence']*100:.1f}%"],
            ["Seeds Detected", str(r["seed_count"])],
            ["Defect Ratio",   f"{r['defect_ratio']*100:.1f}%"],
            ["Purity",         f"{r.get('purity', 0):.1f}%"],
            ["Broken Kernels", f"{r.get('broken_pct', 0):.1f}%"],
        ]
        tbl = Table(info, colWidths=[5*cm, 12*cm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), DARK),
            ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
            ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",   (0,0), (-1,-1), 9),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [LIGHT, colors.white]),
            ("GRID",       (0,0), (-1,-1), 0.4, colors.HexColor("#d1d5db")),
            ("TEXTCOLOR",  (1,6), (1,6), grade_color),
            ("FONTNAME",   (1,6), (1,6), "Helvetica-Bold"),
            ("FONTSIZE",   (1,6), (1,6), 13),
            ("PADDING",    (0,0), (-1,-1), 7),
        ]))
        story.append(tbl)

        story.append(Paragraph("Feature Analysis Scores", h2))
        feat = [
            ["Feature", "Defect Score", "Weight", "Interpretation"],
            ["Shape Analysis",   f"{r['shape_score']*100:.1f}%",   "35%", "Contour geometry, aspect ratio, solidity"],
            ["Color Analysis",   f"{r['color_score']*100:.1f}%",   "35%", "HSV color deviation from ideal range"],
            ["Texture Analysis", f"{r['texture_score']*100:.1f}%", "20%", "Laplacian variance for surface damage"],
        ]
        ft = Table(feat, colWidths=[5*cm, 3*cm, 2*cm, 7*cm])
        ft.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), GREEN),
            ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
            ("FONTNAME",   (0,0), (-1,-1), "Helvetica"),
            ("FONTSIZE",   (0,0), (-1,-1), 9),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [LIGHT, colors.white]),
            ("GRID",       (0,0), (-1,-1), 0.4, colors.HexColor("#d1d5db")),
            ("PADDING",    (0,0), (-1,-1), 7),
        ]))
        story.append(ft)

        proc_abs = os.path.join(APP_DIR, r["processed_path"])
        if os.path.exists(proc_abs):
            story.append(Paragraph("Processed Image (Detection Overlay)", h2))
            story.append(RLImage(proc_abs, width=14*cm, height=8*cm))

        story.append(Paragraph("Analysis Notes", h2))
        story.append(Paragraph(r["notes"] or "No notes.", body))
        story.append(Spacer(1, 0.6*cm))
        story.append(HRFlowable(width="100%", thickness=0.5, color=GRAY))
        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph(
            "Generated by SeedSense v2.0 · University of Agriculture Faisalabad · FYP 2026 · "
            "Supervisor: Sir Imran Mumtaz · Student: Abdur Rehman (2022-ag-8038)",
            ParagraphStyle("foot", fontSize=7, textColor=GRAY)
        ))

        doc.build(story)
        buf.seek(0)
        return send_file(buf, mimetype="application/pdf", as_attachment=True,
                         download_name=f"seedsense_report_{r.get('batch_id','').replace('-','_')}.pdf")

    except ImportError:
        flash("ReportLab not installed. Run: pip install reportlab", "danger")
        return redirect(url_for("view_result", result_id=result_id))


@app.route("/delete/<int:result_id>", methods=["POST"])
@login_required
def delete_result(result_id):
    conn = get_db()
    conn.execute("DELETE FROM grading_results WHERE id=? AND user_id=?", (result_id, session["user_id"]))
    conn.commit()
    conn.close()
    flash("Analysis deleted.", "info")
    return redirect(url_for("history"))


@app.route("/api/stats")
@login_required
def api_stats():
    conn = get_db()
    rows = conn.execute(
        "SELECT grade, COUNT(*) as cnt FROM grading_results WHERE user_id=? GROUP BY grade",
        (session["user_id"],)
    ).fetchall()
    trend = conn.execute(
        """SELECT strftime('%b %Y', created_at) as month, COUNT(*) as cnt,
           AVG(confidence)*100 as avg_conf
           FROM grading_results WHERE user_id=?
           GROUP BY strftime('%Y-%m', created_at) ORDER BY created_at ASC LIMIT 6""",
        (session["user_id"],)
    ).fetchall()
    conn.close()
    return jsonify({
        "grades": {r["grade"]: r["cnt"] for r in rows},
        "trend": [dict(t) for t in trend],
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
