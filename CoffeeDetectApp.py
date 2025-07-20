import os
import streamlit    
import pages.about as about
import pages.instruction as instruction
import pages.history as history
import streamlit as st
import numpy as np
import cv2
import math
import csv
import shutil
from datetime import datetime
from collections import Counter
import sqlite3
from datetime import datetime
from PIL import Image
from sklearn.cluster import DBSCAN
from ultralytics import YOLO
import geojson
from geojson import Feature, Point, FeatureCollection
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
import rasterio
import pyproj
from db import init_db, insert_detection
import base64
import requests
import json
import uuid
import os
import sys
import tempfile
from cryptography.fernet import Fernet

FERNET_KEY = b'NbJ77pL1f_XMqiBFENAOka6dY1Ei8ilmZjEjx2boA7c='  # <-- HARDCODE THIS SECURELY
ENCRYPTED_MODEL_PATH = "best.enc"

def resource_path(relative_path):
    try:
        return os.path.join(sys._MEIPASS, relative_path)
    except AttributeError:
        return os.path.abspath(relative_path)

@st.cache_resource(show_spinner="Decrypting model...")
def load_encrypted_model():
    fernet = Fernet(FERNET_KEY)

    enc_path = resource_path("models/best.enc")

    with open(enc_path, "rb") as enc_file:
        encrypted_model = enc_file.read()
    decrypted_model = fernet.decrypt(encrypted_model)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt", dir=tempfile.gettempdir()) as tmp:
        tmp.write(decrypted_model)
        tmp_model_path = tmp.name

    model = YOLO(tmp_model_path).to("cpu")

    try:
        os.remove(tmp_model_path)
    except Exception as e:
        st.warning(f"Could not delete temp file: {e}")

    return model






# === SETTINGS ===
ADMIN_SERVER_URL = "https://coffee-license-server.onrender.com/validate"
LICENSE_FILE = "license.json"



# === HELPERS ===
def get_mac():
    mac = uuid.getnode()
    return ':'.join(("%012X" % mac)[i:i+2] for i in range(0, 12, 2))


def save_license(data):
    with open(LICENSE_FILE, "w") as f:
        json.dump(data, f)


def load_license():
    if os.path.exists(LICENSE_FILE):
        with open(LICENSE_FILE, "r") as f:
            return json.load(f)
    return None


def validate_license_with_server(mac, key):
    try:
        response = requests.post(ADMIN_SERVER_URL, json={"mac_address": mac, "license_key": key}, timeout=10)
        return response.status_code == 200 and response.json().get("status") == "active"
    except:
        return False


# === LICENSE CHECK FIRST ===
mac_address = get_mac()
license_data = load_license()

# Block app completely if not validated
if not license_data or not validate_license_with_server(license_data["mac_address"], license_data["license_key"]):
    # Show license prompt only
    st.set_page_config(page_title="🔒 License Required", layout="centered")
    st.title("🔐 License Activation Required")

    license_input = st.text_input("Paste your license key", type="password")

    if st.button("Activate"):
        if validate_license_with_server(mac_address, license_input):
            save_license({"mac_address": mac_address, "license_key": license_input})
            st.success("✅ License activated! Please restart the app.")
            st.stop()
        else:
            st.error("❌ Invalid or revoked license key. Please contact support.")
            st.stop()

    # Always stop app if not validated
    st.stop()
# CONFIG
st.set_page_config(
    page_title="CoffeeDetect",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"Get Help": None, "Report a bug": None, "About": None}
)

# Hide specific sidebar elements including extra container div, duplicate nav, and collapse icon
st.markdown("""
<style>
div.stAppToolbar.st-emotion-cache-14vh5up.e4x2yc32 {
    display: none;
}
ul[class^="st-emotion-cache-"][class$="e16b601d1"] {
    display: none !important;
}
div.st-emotion-cache-595tnf.e1quxfqw4 {
    display: none !important;
}
div.st-emotion-cache-kuzxw1.e16b601d9 {
    display: none !important;
}
span[data-testid="stIconMaterial"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

init_db()
Image.MAX_IMAGE_PIXELS = None
TILE_SIZE = 1600
OVERLAP = 200
MERGE_EPS_M = 1.0
MERGE_EPS_PX = 50
SPACING_DROP = 8
CONF_THRES = 0.45
IOU_THRES = 0.30
CLASS_NAMES = {0: 'Coffee', 1: 'DeadCoffee', 2: 'YoungCoffee'}

# Sidebar Navigation
# Encode local logo image to base64
logo_path = os.path.join("assets", "cacount_1.png")
with open(logo_path, "rb") as f:
    logo_base64 = base64.b64encode(f.read()).decode()

# Sidebar with logo + title
st.sidebar.markdown(f"""
    <div style='display: flex; align-items: left: 10px;'>
        <img src='data:image/png;base64,{logo_base64}' width='50'style'margin-bottom: 3px;'/>
        <h1 style='margin: 0; color: black;'>Coffee Count Lite v1.0</h1>
    </div>
""", unsafe_allow_html=True)
page = st.sidebar.radio("Navigate", ["About", "Instruction", "Detect", "History"])




st.markdown("""
<style>
/* Sidebar container */
[data-testid="stSidebar"] {
    background-color: #1c1c1c;
    color: white;
    padding-top: 30px;
}

/* Sidebar title */
[data-testid="stSidebar"] h1 {
    font-size: 20px;
    color: white !important;
    font-weight: bold;
    text-align: center;
    margin-bottom: 25px;
}

/* Radio group container */
[data-testid="stSidebar"] .stRadio > div {
    gap: 0.5rem;
    flex-direction: column;
}

/* Individual radio labels */
[data-testid="stSidebar"] label {
    font-size: 20px;
    padding: 8px 14px;
    border-radius: 8px;
    cursor: pointer;
    background-color: transparent;
    transition: background 0.2s ease;
}

/* Hover effect */
[data-testid="stSidebar"] label:hover {
    background-color: rgba(255, 255, 255, 0.1);
}

/* Selected radio item */
[data-testid="stSidebar"] input:checked + div > label {
    background-color: #FFD700;
    color: black;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


# Render Pages
if page == "About":
    about.render_about()
    st.stop()
elif page == "Instruction":
    instruction.render_instruction()
    st.stop()
elif page == "History":
    history.render_history()
    st.stop()

# Upload Section
st.title("📤 Upload File for Detection")
st.markdown("""
<div class="file-upload-box">
    <div class="upload-label">Select and upload the file of your choice (max. 1 file only).</div>
    <div class="upload-label">Supports geo-referenced TIFF format only. After detection the system will display:Total Count, Young Coffee, Mature Coffee, Dead Coffee, Average Spacing, Smallest Spacing, Largest Spacing.</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(label="Upload GeoTIFF", type=["tif"], label_visibility="collapsed")




if uploaded_file and not st.session_state.get("detection_complete"):
    st.write(f"📄 **Selected file:** {uploaded_file.name}")
    st.markdown('<div class="detect-btn-container">', unsafe_allow_html=True)
    run_detection = st.button("Run Detection")
    st.markdown('</div>', unsafe_allow_html=True)

    if run_detection:
        image_path = f"temp_{uploaded_file.name}"
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            with rasterio.open(image_path) as src:
                LON_LEFT, LAT_BOTTOM, LON_RIGHT, LAT_TOP = src.bounds
                src_crs = src.crs
                if not src_crs:
                    raise ValueError("Missing coordinate system.")
        except:
            st.error("⚠️ Invalid or unsupported file. Please upload a valid GeoTIFF with proper coordinate information.")
            os.remove(image_path)
            st.stop()

        to_m = pyproj.Transformer.from_crs(src_crs, "EPSG:3857", always_xy=True)
        to_ll = pyproj.Transformer.from_crs("EPSG:3857", src_crs, always_xy=True)

        basename = os.path.splitext(os.path.basename(image_path))[0]
        out_dir = os.path.join("results", f"Coffee-Count-{basename}")
        tile_dir = os.path.join(out_dir, "tiles")
        os.makedirs(tile_dir, exist_ok=True)

        csv_path = os.path.join(out_dir, "detections.csv")
        geojson_path = os.path.join(out_dir, "detections.geojson")
        annot_path = os.path.join(out_dir, f"{basename}_annotated.jpg")
        pdf_path = os.path.join(out_dir, "report.pdf")

        model = load_encrypted_model()


        image = Image.open(image_path)
        w_img, h_img = image.size

        def slice_image(img):
            tiles, pos, size = [], [], []
            for y in range(0, h_img, TILE_SIZE - OVERLAP):
                for x in range(0, w_img, TILE_SIZE - OVERLAP):
                    w = min(TILE_SIZE, w_img - x)
                    h = min(TILE_SIZE, h_img - y)
                    tiles.append(img.crop((x, y, x + w, y + h)))
                    pos.append((x, y))
                    size.append((w, h))
            return tiles, pos, size

        progress_bar = st.progress(0)
        progress_text = st.empty()
        with st.spinner("☕ Brew some coffee while you wait..."):
             tiles, positions, sizes = slice_image(image)
             annotated = np.zeros((h_img, w_img, 3), dtype=np.uint8)
             detections = []
             uid = 0
             with open(csv_path, "w", newline="") as f_csv:
                writer = csv.writer(f_csv)
                writer.writerow(["Tile", "TreeID", "Class", "X", "Y", "Lat", "Lon", "Confidence"])

                for idx, (tile, (xo, yo), (tw, th)) in enumerate(zip(tiles, positions, sizes), 1):
                    tp = os.path.join(tile_dir, f"tile_{idx}.png")
                    tile.save(tp)

                    res = model(tp, conf=CONF_THRES, iou=IOU_THRES)[0]
                    res.names = CLASS_NAMES

                    for box in res.boxes:
                        cls_id = int(box.cls)
                        conf = float(box.conf)
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        cx = xo + (x1 + x2) / 2
                        cy = yo + (y1 + y2) / 2

                        lon = LON_LEFT + (cx / w_img) * (LON_RIGHT - LON_LEFT)
                        lat = LAT_TOP - (cy / h_img) * (LAT_TOP - LAT_BOTTOM)
                        mx, my = to_m.transform(lon, lat)

                        detections.append(dict(x=mx, y=my, cls=cls_id, conf=conf, src_idx=idx))
                        uid += 1
                        writer.writerow([idx, uid, CLASS_NAMES[cls_id], round(mx, 3), round(my, 3), lat, lon, round(conf, 3)])

                    at = res.plot()
                    at = cv2.resize(at, (tw, th))
                    annotated[yo:yo + th, xo:xo + tw] = at

                    progress_bar.progress(idx / len(tiles))
                    progress_text.text(f"Processing tile {idx}/{len(tiles)}...")

        progress_bar.empty()
        progress_text.empty()

        cv2.imwrite(annot_path, annotated)
        shutil.rmtree(tile_dir)

        X = np.array([[d["x"], d["y"]] for d in detections])
        labels = DBSCAN(eps=MERGE_EPS_M, min_samples=1).fit(X).labels_

        merged, class_counter = [], Counter()
        for lbl in set(labels):
            idxs = np.where(labels == lbl)[0]
            best = max(idxs, key=lambda i: detections[i]["conf"])
            cls = detections[best]["cls"]
            cx, cy = X[idxs].mean(axis=0)
            merged.append((cx, cy, cls))
            class_counter[cls] += 1

        nd = [min(math.hypot(x1 - x2, y1 - y2) for j, (x2, y2, _) in enumerate(merged) if j != i) for i, (x1, y1, _) in enumerate(merged)]
        nd = [d for d in nd if d <= SPACING_DROP]

        avg_d = sum(nd) / len(nd) if nd else 0
        smallest = min(nd) if nd else 0
        largest = max(nd) if nd else 0

        features = [Feature(geometry=Point(to_ll.transform(mx, my)), properties=dict(id=i + 1, class_=CLASS_NAMES[cls])) for i, (mx, my, cls) in enumerate(merged)]

        with open(geojson_path, "w") as gjs:
            geojson.dump(FeatureCollection(features), gjs)

        c = canvas.Canvas(pdf_path, pagesize=letter)
        pw, ph = letter
        px = 50
        py = ph - 50

        c.setFont("Helvetica-Bold", 16)
        c.drawString(px, py, "Coffee Tree Detection Report")
        py -= 25
        c.setFont("Helvetica", 12)
        c.drawString(px, py, f"Input image: {basename}")
        py -= 20
        c.drawString(px, py, f"Date: {datetime.now():%Y-%m-%d %H:%M}")
        py -= 20
        c.drawString(px, py, f"Total coffee trees: {len(merged)}")
        py -= 20
        for cls_id in sorted(CLASS_NAMES):
            c.drawString(px + 20, py, f"{CLASS_NAMES[cls_id]}: {class_counter.get(cls_id, 0)}")
            py -= 15

        c.drawString(px, py, f"Average spacing: {avg_d:.2f} m")
        c.drawString(px, py - 15, f"Smallest spacing: {smallest:.2f} m")
        c.drawString(px, py - 30, f"Largest spacing: {largest:.2f} m")

        try:
            img = ImageReader(annot_path)
            iw, ih = img.getSize()
            max_w, max_h = 400, 300
            scale = min(max_w / iw, max_h / ih)
            dw, dh = iw * scale, ih * scale
            x_center = (pw - dw) / 2
            y_pos = py - dh - 50
            c.drawImage(img, x_center, y_pos, dw, dh)
        except:
            pass

        c.showPage()
        c.save()

        st.session_state.update({
            "detection_complete": True,
            "saved_to_db": False,
            "annot_path": annot_path,
            "pdf_path": pdf_path,
            "geojson_path": geojson_path,
            "csv_path": csv_path,
            "total_trees": len(merged),
            "class_counter": class_counter,
            "avg_d": avg_d,
            "smallest": smallest,
            "largest": largest,
            "basename": basename
        })

if st.session_state.get("detection_complete") and not st.session_state.get("saved_to_db"):
    insert_detection(
        filename=st.session_state["basename"],
        total=st.session_state["total_trees"],
        young=st.session_state["class_counter"].get(2, 0),
        mature=st.session_state["class_counter"].get(0, 0),
        dead=st.session_state["class_counter"].get(1, 0),
        avg=st.session_state["avg_d"],
        small=st.session_state["smallest"],
        large=st.session_state["largest"],
        annot_path=st.session_state["annot_path"],
        pdf_path=st.session_state["pdf_path"],
        geo_path=st.session_state["geojson_path"],  # ✅ comma added here
        csv_path=st.session_state["csv_path"] if "csv_path" in st.session_state else None
    )
    st.session_state["saved_to_db"] = True

if st.session_state.get("detection_complete"):
    st.success(f"✅ {st.session_state['total_trees']} unique coffee trees detected.")
    colL, colR = st.columns([2, 1])
    with colL:
        st.markdown(f"""
**File name:** {st.session_state['basename']}  
**Total Count:** {st.session_state['total_trees']}  
**Young Coffee:** {st.session_state['class_counter'].get(2, 0)}  
**Mature Coffee:** {st.session_state['class_counter'].get(0, 0)}  
**Dead Coffee:** {st.session_state['class_counter'].get(1, 0)}  
**Average Spacing:** {st.session_state['avg_d']:.2f} m  
**Smallest Spacing:** {st.session_state['smallest']:.2f} m  
**Largest Spacing:** {st.session_state['largest']:.2f} m  
""")
    with colR:
        with st.expander("🔍 View  Annotated Image", expanded=False):
            st.image(st.session_state["annot_path"], width=500)
        st.download_button("Download Annotated Image", open(st.session_state["annot_path"], "rb"), file_name="annotated.jpg")
        st.download_button("Download PDF", open(st.session_state["pdf_path"], "rb"), file_name="report.pdf")
        st.download_button("Download GeoJSON", open(st.session_state["geojson_path"], "rb"), file_name="detections.geojson")
        st.download_button("Download CSV", open(os.path.join("results", f"Coffee-Count-{st.session_state['basename']}", "detections.csv"), "rb"), file_name="detections.csv", key="current_csv")

    st.markdown("---")
    if st.button("🔄 Start New Detection"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
