"""
TimberGrade - Website (Streamlit)
Run:  python -m streamlit run app.py
"""
import base64
import glob
import io
import os
import random
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from grading import grade_plank

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "best.pt")
DB = os.path.join(HERE, "history.db")
DATASET_DIR = os.path.join(HERE, "dataset")

GRADE_STYLE = {"A": ("✅", st.success), "B": ("⚠️", st.warning), "C": ("❌", st.error)}
GRADE_COLOR = {"A": "#2e9d5b", "B": "#e0a526", "C": "#d9534f"}


# ---------- Model ----------
@st.cache_resource
def load_model():
    from ultralytics import YOLO
    return YOLO(MODEL_PATH)


def detect(img, conf):
    """Runs YOLO on the photo. Returns (detections, image_with_boxes)"""
    r = load_model()(img, imgsz=1024, conf=conf, verbose=False)[0]
    dets = [(r.names[int(c)], float(s), tuple(map(float, b)))
            for c, s, b in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy)]
    boxed = Image.fromarray(r.plot()[:, :, ::-1])   # BGR -> RGB
    return dets, boxed


def dataset_images():
    """All plank photos inside the dataset folder (searched in sub-folders too)."""
    pics = []
    for ext in ("jpg", "jpeg", "png", "bmp"):
        pics += glob.glob(os.path.join(DATASET_DIR, "**", f"*.{ext}"), recursive=True)
        pics += glob.glob(os.path.join(DATASET_DIR, "**", f"*.{ext.upper()}"), recursive=True)
    return sorted(set(pics))


# ---------- Database ----------
def db():
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS inspections
                   (time TEXT, plank_id TEXT, grade TEXT, score INTEGER, defects INTEGER, details TEXT)""")
    return con


def save(plank_id, res):
    details = ", ".join(f"{k}: {v}" for k, v in res["counts"].items()) or "No defects"
    with db() as con:
        con.execute("INSERT INTO inspections VALUES (?,?,?,?,?,?)",
                    (datetime.now().strftime("%d-%m-%Y %H:%M"), plank_id, res["grade"],
                     res["score"], len(res["defects"]), details))


def history():
    with db() as con:
        return pd.read_sql("SELECT * FROM inspections ORDER BY rowid DESC", con)


# ---------- Simple HTML table / charts (no pyarrow needed) ----------
def show_table(df):
    st.markdown(df.to_html(index=False, escape=True, border=0, classes="tg"), unsafe_allow_html=True)


def bar_chart(counts):
    top = max(1, max(counts.values()))
    html = ""
    for g, n in counts.items():
        w = int(100 * n / top)
        html += (f'<div style="display:flex;align-items:center;gap:8px;margin:6px 0">'
                 f'<b style="width:70px">Grade {g}</b>'
                 f'<div style="flex:1;background:#eee;border-radius:6px">'
                 f'<div style="width:{w}%;min-width:4px;background:{GRADE_COLOR[g]};height:26px;border-radius:6px"></div></div>'
                 f'<span style="width:30px">{n}</span></div>')
    st.markdown(html, unsafe_allow_html=True)


def defect_bar(counts):
    """Horizontal bar chart of defect-type counts (dict name -> count)."""
    if not counts:
        st.write("No defects recorded yet.")
        return
    top = max(counts.values())
    html = ""
    for name, n in sorted(counts.items(), key=lambda x: -x[1]):
        w = int(100 * n / top)
        html += (f'<div style="display:flex;align-items:center;gap:8px;margin:5px 0">'
                 f'<span style="width:140px;font-size:13px">{name}</span>'
                 f'<div style="flex:1;background:#eee;border-radius:6px">'
                 f'<div style="width:{w}%;min-width:4px;background:#8a4b14;height:22px;border-radius:6px"></div></div>'
                 f'<b style="width:34px">{n}</b></div>')
    st.markdown(html, unsafe_allow_html=True)


def line_chart(scores):
    W, H, pad = 500, 200, 25
    n = len(scores)
    xs = [pad + (W - 2 * pad) * (i / max(1, n - 1)) for i in range(n)]
    ys = [H - pad - (H - 2 * pad) * (s / 100) for s in scores]
    pts = " ".join(f"{x:.0f},{y:.0f}" for x, y in zip(xs, ys))
    dots = "".join(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="4" fill="#8a4b14"/>' for x, y in zip(xs, ys))
    st.markdown(
        f'<svg viewBox="0 0 {W} {H}" style="width:100%;max-width:{W}px">'
        f'<line x1="{pad}" y1="{H-pad}" x2="{W-pad}" y2="{H-pad}" stroke="#bbb"/>'
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{H-pad}" stroke="#bbb"/>'
        f'<text x="2" y="{pad+4}" font-size="11" fill="#777">100</text>'
        f'<text x="8" y="{H-pad+4}" font-size="11" fill="#777">0</text>'
        f'<polyline points="{pts}" fill="none" stroke="#8a4b14" stroke-width="2"/>{dots}</svg>',
        unsafe_allow_html=True)


# ---------- Quality certificate (HTML, opens and prints to PDF) ----------
def certificate_html(plank_id, res, boxed_img):
    buf = io.BytesIO()
    img = boxed_img.copy()
    img.thumbnail((900, 900))
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    g = res["grade"]
    rows = "".join(
        f"<tr><td>{d['Defect']}</td><td>{d['Confidence %']}%</td>"
        f"<td>{d['Area % of plank']}%</td><td>{d['Penalty']}</td></tr>"
        for d in res["defects"]) or "<tr><td colspan='4'>No defects found</td></tr>"
    now = datetime.now().strftime("%d-%m-%Y %H:%M")
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>TimberGrade Certificate {plank_id}</title>
<style>
 body{{font-family:Arial,sans-serif;color:#222;max-width:800px;margin:24px auto;padding:0 20px}}
 .box{{border:3px solid #5a3412;border-radius:10px;padding:26px}}
 h1{{color:#5a3412;text-align:center;margin:0}}
 .sub{{text-align:center;color:#555;margin:4px 0 18px}}
 .grade{{text-align:center;font-size:40px;font-weight:bold;color:#fff;background:{GRADE_COLOR[g]};
         border-radius:10px;padding:12px;margin:14px 0}}
 table{{border-collapse:collapse;width:100%;margin-top:10px;font-size:14px}}
 th,td{{border:1px solid #ccc;padding:6px 10px;text-align:left}}
 th{{background:#f3e7da}}
 .meta td{{border:none;padding:3px 0}}
 img{{width:100%;border-radius:8px;margin:10px 0}}
 .foot{{text-align:center;color:#888;font-size:12px;margin-top:18px}}
 @media print{{.noprint{{display:none}}}}
</style></head><body>
<p class="noprint" style="text-align:center;color:#555">
  Press <b>Ctrl + P</b> and choose <b>Save as PDF</b> to save this certificate.</p>
<div class="box">
  <h1>🪵 TimberGrade</h1>
  <p class="sub">Wooden Plank Quality Inspection Certificate</p>
  <table class="meta">
    <tr><td><b>Plank ID:</b> {plank_id}</td><td><b>Date:</b> {now}</td></tr>
  </table>
  <div class="grade">GRADE {g} &nbsp;—&nbsp; Quality score {res['score']}%</div>
  <p><b>Recommended use:</b> {res['use']}</p>
  <img src="data:image/png;base64,{b64}" alt="AI detection result">
  <table>
    <tr><th>Defect</th><th>Confidence</th><th>Area</th><th>Penalty</th></tr>
    {rows}
  </table>
  <p class="foot">Generated by TimberGrade - AI-Based Wooden Plank Defect Detection and Quality Grading System</p>
</div></body></html>"""


# ---------- UI ----------
st.set_page_config(page_title="TimberGrade", page_icon="🪵", layout="wide")
st.title("🪵 TimberGrade")
st.caption("AI-Based Wooden Plank Defect Detection and Quality Grading System")
st.markdown("<style>table.tg{border-collapse:collapse;width:100%;font-size:14px}"
            "table.tg th{background:#f3e7da;color:#3a2410;text-align:left}"
            "table.tg th,table.tg td{padding:6px 10px;border-bottom:1px solid #ddd}</style>",
            unsafe_allow_html=True)

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Put the `best.pt` downloaded from Colab inside the `models` folder, "
             "then refresh the page.")
    st.stop()

tab1, tab4, tab2, tab3 = st.tabs(
    ["🔍 Inspect plank", "📦 Batch inspection", "📊 Dashboard", "📁 Dataset & Model"])

# ================================================================= INSPECT
with tab1:
    left, right = st.columns([1, 1])
    with left:
        src = st.radio("Photo source", ["Upload photo", "Webcam / phone camera", "From dataset"], horizontal=True)
        file = None
        if src == "Upload photo":
            file = st.file_uploader("Plank photo", type=["jpg", "jpeg", "png", "bmp"])
        elif src == "Webcam / phone camera":
            file = st.camera_input("Take a photo of the plank")
        else:
            pics = dataset_images()
            if not pics:
                st.info("No dataset photos found. Put the dataset images in a folder named "
                        "**dataset** next to app.py, then refresh the page.")
            else:
                names = [os.path.relpath(p, DATASET_DIR) for p in pics]
                if "pick" not in st.session_state or st.session_state.pick not in names:
                    st.session_state.pick = names[0]
                if st.button(f"🎲 Random photo ({len(names)} in dataset)"):
                    st.session_state.pick = random.choice(names)
                choice = st.selectbox("Dataset photo", names, key="pick")
                file = os.path.join(DATASET_DIR, choice)
                st.image(file, caption=choice, use_container_width=True)
        plank_id = st.text_input("Plank ID (optional)", value=f"P-{datetime.now():%H%M%S}")
        conf = st.slider("AI confidence threshold", 0.10, 0.90, 0.25, 0.05,
                         help="Defects detected below this confidence are ignored")
        go = st.button("🔍 Check plank", type="primary", use_container_width=True)

    if go:
        if file is None:
            st.warning("Please upload, capture or choose a plank photo first")
        else:
            img = Image.open(file).convert("RGB")
            with st.spinner("AI is detecting defects..."):
                dets, boxed = detect(img, conf)
            res = grade_plank(dets, *img.size)
            save(plank_id, res)
            st.session_state.last = {"plank_id": plank_id, "res": res, "boxed": boxed}

    last = st.session_state.get("last")
    if last:
        res = last["res"]
        with right:
            st.image(last["boxed"], caption="AI detection result", use_container_width=True)
            icon, box = GRADE_STYLE[res["grade"]]
            box(f"{icon} GRADE {res['grade']}  —  Quality score {res['score']}%")
            st.progress(res["score"] / 100)
            st.write(f"**Recommended use:** {res['use']}")
            if res["defects"]:
                st.write(f"**{len(res['defects'])} defect(s) found:**")
                show_table(pd.DataFrame(res["defects"]))
            else:
                st.write("**No defects found** 🎉")
            st.download_button("📄 Download quality certificate",
                               certificate_html(last["plank_id"], res, last["boxed"]),
                               file_name=f"TimberGrade_{last['plank_id']}.html",
                               mime="text/html",
                               help="Opens in the browser; press Ctrl+P to save as PDF")

# ================================================================= BATCH
with tab4:
    st.write("### Batch inspection")
    st.caption("Check a whole shipment at once. Upload many plank photos, or pick random photos "
               "from the dataset, and grade them together.")
    bsrc = st.radio("Photos from", ["Upload photos", "Random from dataset"], horizontal=True, key="bsrc")
    bconf = st.slider("AI confidence threshold", 0.10, 0.90, 0.25, 0.05, key="bconf")

    batch_items = []   # list of (name, file_or_path)
    if bsrc == "Upload photos":
        files = st.file_uploader("Plank photos", type=["jpg", "jpeg", "png", "bmp"],
                                 accept_multiple_files=True, key="batch")
        batch_items = [(f.name, f) for f in (files or [])]
    else:
        pics = dataset_images()
        if not pics:
            st.info("No dataset photos found. Put the dataset images in a folder named "
                    "**dataset** next to app.py, then refresh the page.")
        else:
            n = st.number_input("How many random planks to check", 1, 50, 10, key="bn")
            if st.button(f"🎲 Pick {n} random planks from dataset"):
                st.session_state.batch_pick = random.sample(pics, min(int(n), len(pics)))
            picked = st.session_state.get("batch_pick", [])
            if picked:
                st.write(f"**{len(picked)} planks selected from dataset.**")
            batch_items = [(os.path.basename(p), p) for p in picked]

    if st.button("📦 Grade all planks", type="primary", disabled=not batch_items):
        rows = []
        prog = st.progress(0.0)
        for i, (name, src) in enumerate(batch_items):
            img = Image.open(src).convert("RGB")
            dets, _ = detect(img, bconf)
            r = grade_plank(dets, *img.size)
            pid = os.path.splitext(name)[0][:20]
            save(pid, r)
            rows.append({"Plank": name, "Grade": r["grade"], "Score %": r["score"],
                         "Defects": len(r["defects"]),
                         "Types": ", ".join(r["counts"].keys()) or "None"})
            prog.progress((i + 1) / len(batch_items))
        st.session_state.batch_rows = rows
        prog.empty()

    brows = st.session_state.get("batch_rows")
    if brows:
        bdf = pd.DataFrame(brows)
        c = {g: int((bdf["Grade"] == g).sum()) for g in ["A", "B", "C"]}
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Planks checked", len(bdf))
        m2.metric("Grade A", c["A"])
        m3.metric("Grade B", c["B"])
        m4.metric("Grade C", c["C"])
        st.write("#### Grade breakdown")
        bar_chart(c)
        st.write("#### Results")
        show_table(bdf)
        st.download_button("⬇️ Download batch report (CSV)", bdf.to_csv(index=False),
                           "timbergrade_batch.csv")

# ================================================================= DASHBOARD
with tab2:
    df = history()
    if df.empty:
        st.info("No planks inspected yet.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total planks", len(df))
        c2.metric("Grade A", int((df.grade == "A").sum()))
        c3.metric("Grade B", int((df.grade == "B").sum()))
        c4.metric("Grade C", int((df.grade == "C").sum()))
        a, b = st.columns(2)
        with a:
            st.write("#### Grade breakdown")
            bar_chart({g: int((df.grade == g).sum()) for g in ["A", "B", "C"]})
        with b:
            st.write("#### Quality score per plank")
            line_chart(df.iloc[::-1]["score"].tolist())

        # ----- defect-type breakdown across all inspections -----
        st.write("#### Most common defects")
        dcount = {}
        for det in df["details"]:
            if det and det != "No defects":
                for part in det.split(", "):
                    name = part.rsplit(":", 1)[0].strip()
                    try:
                        n = int(part.rsplit(":", 1)[1])
                    except (IndexError, ValueError):
                        n = 1
                    dcount[name] = dcount.get(name, 0) + n
        defect_bar(dcount)

        # ----- estimated shipment value -----
        st.write("#### Estimated shipment value")
        st.caption("Set a price per plank for each grade to estimate the value of all inspected planks.")
        p1, p2, p3 = st.columns(3)
        pa = p1.number_input("Grade A price (₹)", min_value=0, value=500, step=50)
        pb = p2.number_input("Grade B price (₹)", min_value=0, value=300, step=50)
        pc = p3.number_input("Grade C price (₹)", min_value=0, value=150, step=50)
        na = int((df.grade == "A").sum()); nb = int((df.grade == "B").sum()); nc = int((df.grade == "C").sum())
        total = na * pa + nb * pb + nc * pc
        st.success(f"**Estimated total value: ₹ {total:,}**  "
                   f"(A: {na}×{pa} + B: {nb}×{pb} + C: {nc}×{pc})")

        st.write("#### Inspection history")
        show_table(df)
        st.download_button("⬇️ Download report (CSV)", df.to_csv(index=False), "timbergrade_report.csv")
        if st.button("🗑️ Clear history"):
            with db() as con:
                con.execute("DELETE FROM inspections")
            st.session_state.pop("batch_rows", None)
            st.session_state.pop("last", None)
            st.rerun()

# ================================================================= DATASET & MODEL
with tab3:
    st.write("### Dataset")
    st.markdown(
        "**Name:** Large Scale Image Dataset of Wood Surface Defects  \n"
        "**Source:** [Kaggle](https://www.kaggle.com/datasets/nomihsa965/large-scale-image-dataset-of-wood-surface-defects) (open dataset)  \n"
        "**Images:** 4,000 photos of wood surfaces (2800 × 1024 px), with a box drawn around every defect  \n"
        "**Split:** 3,400 images for training (85%) and 600 images for validation (15%)  \n"
        "**Defects labelled in the validation set:** 1,391")

    st.write("#### 8 defect classes")
    from grading import DEFECT_INFO
    classes = [
        ("Live_Knot", "live_knot", "Healthy knot, firmly attached to the wood"),
        ("Dead_Knot", "dead_knot", "Dark, loose knot that can fall out"),
        ("Knot_with_crack", "knot_with_crack", "Knot that has a crack through it"),
        ("Knot_missing", "knot_missing", "Hole left where a knot fell out"),
        ("Crack", "crack", "Split in the wood - the most serious defect"),
        ("Resin", "resin", "Pocket or streak of sticky resin"),
        ("Marrow", "marrow", "Soft centre (pith) of the tree visible on the surface"),
        ("Quartzity", "quartzity", "Rough, irregular grain pattern"),
    ]
    show_table(pd.DataFrame([{"Class": c, "Description": d, "Penalty points": DEFECT_INFO[k]["penalty"],
                              "Serious": "Yes" if DEFECT_INFO[k]["serious"] else "No"} for c, k, d in classes]))

    st.write("### Model training")
    show_table(pd.DataFrame([
        {"Setting": "Model", "Value": "YOLOv8n (Ultralytics), 3.0 million parameters"},
        {"Setting": "Trained on", "Value": "Google Colab, Tesla T4 GPU"},
        {"Setting": "Image size", "Value": "1024 px"},
        {"Setting": "Epochs", "Value": "30 (batch size 8, AdamW optimizer)"},
        {"Setting": "Best mAP50", "Value": "63.8% (epoch 15)"},
        {"Setting": "Model file", "Value": "models/best.pt (about 6 MB)"},
    ]))

    # Training graphs and sample images from training_results.zip (if extracted next to app.py)
    found = {}
    for name in ["results.png", "confusion_matrix_normalized.png", "confusion_matrix.png",
                 "labels.jpg", "val_batch0_labels.jpg", "val_batch0_pred.jpg"]:
        hits = glob.glob(os.path.join(HERE, "training_results", "**", name), recursive=True)
        if hits:
            found[name] = hits[0]
    if found:
        st.write("### Training results")
        captions = {
            "results.png": "Loss and accuracy (precision, recall, mAP) after every epoch",
            "confusion_matrix_normalized.png": "Confusion matrix - how often each defect was identified correctly",
            "confusion_matrix.png": "Confusion matrix - how often each defect was identified correctly",
            "labels.jpg": "How many defects of each class are in the training data",
            "val_batch0_labels.jpg": "Dataset images with the real (human-labelled) defect boxes",
            "val_batch0_pred.jpg": "The same images with the boxes predicted by our model",
        }
        if "confusion_matrix_normalized.png" in found:
            found.pop("confusion_matrix.png", None)
        for name, path in found.items():
            st.image(path, caption=captions[name], use_container_width=True)
    else:
        st.info("To show training graphs and sample dataset images here, extract training_results.zip "
                "into a folder named training_results next to app.py, then refresh this page.")
        