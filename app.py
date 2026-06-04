import json
import time
import numpy as np
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import tensorflow as tf
import plotly.graph_objects as go  # type: ignore[import-not-found]
import plotly.express as px  # type: ignore[import-not-found]

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Animated nature background ── */
    .stApp {
        background-color: #020d06;
        position: relative;
        overflow: hidden;
    }

    /* Deep forest gradient base */
    .stApp::before {
        content: '';
        position: fixed;
        inset: 0;
        z-index: 0;
        background:
            radial-gradient(ellipse 80% 60% at 20% 10%,  #0a2e14 0%,  transparent 60%),
            radial-gradient(ellipse 60% 80% at 80% 90%,  #091f0c 0%,  transparent 55%),
            radial-gradient(ellipse 100% 50% at 50% 50%, #030f05 0%,  transparent 80%),
            linear-gradient(160deg, #020d06 0%, #041208 40%, #020d06 100%);
        pointer-events: none;
    }

    /* Floating glowing orbs */
    .stApp::after {
        content: '';
        position: fixed;
        inset: 0;
        z-index: 0;
        background:
            radial-gradient(circle 300px at 15% 25%,  rgba(34,197,94,0.07)  0%, transparent 70%),
            radial-gradient(circle 200px at 85% 15%,  rgba(74,222,128,0.05) 0%, transparent 70%),
            radial-gradient(circle 400px at 70% 75%,  rgba(21,128,61,0.08)  0%, transparent 70%),
            radial-gradient(circle 250px at 40% 85%,  rgba(34,197,94,0.06)  0%, transparent 70%);
        animation: floatOrbs 12s ease-in-out infinite alternate;
        pointer-events: none;
    }

    @keyframes floatOrbs {
        0%   { opacity: 0.7; transform: scale(1)    translateY(0px);  }
        50%  { opacity: 1.0; transform: scale(1.05) translateY(-10px);}
        100% { opacity: 0.8; transform: scale(0.97) translateY(5px);  }
    }

    /* Particle leaf layer via SVG data URI */
    .leaf-canvas {
        position: fixed;
        inset: 0;
        z-index: 0;
        pointer-events: none;
        overflow: hidden;
    }

    .leaf {
        position: absolute;
        width: 10px;
        height: 10px;
        background: radial-gradient(circle, rgba(74,222,128,0.35), transparent);
        border-radius: 50% 0 50% 0;
        animation: fall linear infinite;
        opacity: 0;
    }

    /* Generate 18 leaves with varied positions/timings */
    .leaf:nth-child(1)  { left:5%;   width:8px;  height:8px;  animation-duration:14s; animation-delay:0s;    }
    .leaf:nth-child(2)  { left:12%;  width:12px; height:12px; animation-duration:18s; animation-delay:2s;    }
    .leaf:nth-child(3)  { left:22%;  width:6px;  height:6px;  animation-duration:12s; animation-delay:4s;    }
    .leaf:nth-child(4)  { left:33%;  width:10px; height:10px; animation-duration:20s; animation-delay:1s;    }
    .leaf:nth-child(5)  { left:44%;  width:7px;  height:7px;  animation-duration:16s; animation-delay:6s;    }
    .leaf:nth-child(6)  { left:55%;  width:11px; height:11px; animation-duration:13s; animation-delay:3s;    }
    .leaf:nth-child(7)  { left:66%;  width:9px;  height:9px;  animation-duration:17s; animation-delay:5s;    }
    .leaf:nth-child(8)  { left:77%;  width:13px; height:13px; animation-duration:15s; animation-delay:0.5s;  }
    .leaf:nth-child(9)  { left:88%;  width:7px;  height:7px;  animation-duration:19s; animation-delay:7s;    }
    .leaf:nth-child(10) { left:95%;  width:9px;  height:9px;  animation-duration:11s; animation-delay:2.5s;  }
    .leaf:nth-child(11) { left:8%;   width:5px;  height:5px;  animation-duration:22s; animation-delay:8s;    }
    .leaf:nth-child(12) { left:18%;  width:14px; height:14px; animation-duration:10s; animation-delay:1.5s;  }
    .leaf:nth-child(13) { left:28%;  width:8px;  height:8px;  animation-duration:24s; animation-delay:9s;    }
    .leaf:nth-child(14) { left:48%;  width:6px;  height:6px;  animation-duration:14s; animation-delay:3.5s;  }
    .leaf:nth-child(15) { left:60%;  width:10px; height:10px; animation-duration:16s; animation-delay:6.5s;  }
    .leaf:nth-child(16) { left:72%;  width:8px;  height:8px;  animation-duration:18s; animation-delay:4.5s;  }
    .leaf:nth-child(17) { left:82%;  width:12px; height:12px; animation-duration:20s; animation-delay:0.8s;  }
    .leaf:nth-child(18) { left:92%;  width:7px;  height:7px;  animation-duration:13s; animation-delay:7.5s;  }

    @keyframes fall {
        0%   { top: -5%;   opacity: 0;   transform: rotate(0deg)   translateX(0px);  }
        10%  {              opacity: 0.7;                                              }
        50%  {                            transform: rotate(180deg) translateX(30px); }
        90%  {              opacity: 0.5;                                              }
        100% { top: 105%;  opacity: 0;   transform: rotate(360deg) translateX(-20px);}
    }

    /* Make all Streamlit content sit above background layers */
    .stApp > div { position: relative; z-index: 1; }
    section[data-testid="stSidebar"] { position: relative; z-index: 10; }


    /* Card style — frosted glass */
    .result-card {
        background: rgba(20, 40, 20, 0.55);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(74,222,128,0.25);
        border-radius: 16px;
        padding: 24px;
        margin: 10px 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    }

    /* Big result label */
    .big-label {
        font-size: 2rem;
        font-weight: 800;
        color: #4ade80;
        margin: 0;
    }

    .big-label-red {
        font-size: 2rem;
        font-weight: 800;
        color: #f87171;
        margin: 0;
    }

    .sub-label {
        font-size: 1rem;
        color: #9ca3af;
        margin-top: 4px;
    }

    /* Confidence pill */
    .confidence-pill {
        display: inline-block;
        background: #14532d;
        color: #4ade80;
        border-radius: 999px;
        padding: 4px 16px;
        font-size: 0.9rem;
        font-weight: 600;
        margin-top: 10px;
    }

    .confidence-pill-red {
        display: inline-block;
        background: #450a0a;
        color: #f87171;
        border-radius: 999px;
        padding: 4px 16px;
        font-size: 0.9rem;
        font-weight: 600;
        margin-top: 10px;
    }

    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #d1d5db;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 12px;
    }

    /* Tip box */
    .tip-box {
        background: rgba(20, 20, 50, 0.5);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border-left: 4px solid #6366f1;
        border-radius: 8px;
        padding: 14px 18px;
        margin-top: 16px;
        color: #c7d2fe;
        font-size: 0.92rem;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Floating leaf particles ───────────────────────────────────────────────────
st.markdown("""
<div class="leaf-canvas">
  <div class="leaf"></div><div class="leaf"></div><div class="leaf"></div>
  <div class="leaf"></div><div class="leaf"></div><div class="leaf"></div>
  <div class="leaf"></div><div class="leaf"></div><div class="leaf"></div>
  <div class="leaf"></div><div class="leaf"></div><div class="leaf"></div>
  <div class="leaf"></div><div class="leaf"></div><div class="leaf"></div>
  <div class="leaf"></div><div class="leaf"></div><div class="leaf"></div>
</div>
""", unsafe_allow_html=True)

# ── Disease info database ─────────────────────────────────────────────────────
DISEASE_INFO = {
    "healthy": {
        "tip": "Your plant looks healthy! Maintain regular watering and adequate sunlight.",
        "severity": "None",
        "treatment": "No treatment needed.",
    },
    "early blight": {
        "tip": "Early blight is a fungal disease. Remove infected leaves and apply fungicide.",
        "severity": "Moderate",
        "treatment": "Apply copper-based fungicide every 7–10 days.",
    },
    "late blight": {
        "tip": "Late blight spreads rapidly in cool, wet conditions. Act quickly.",
        "severity": "High",
        "treatment": "Apply chlorothalonil or mancozeb fungicide immediately.",
    },
    "leaf scorch": {
        "tip": "Leaf scorch is often caused by drought stress or excess fertilizer.",
        "severity": "Low",
        "treatment": "Improve irrigation and reduce fertilizer use.",
    },
    "black rot": {
        "tip": "Black rot is a bacterial disease. Remove and destroy infected plant parts.",
        "severity": "High",
        "treatment": "Apply copper-based bactericide. Avoid overhead watering.",
    },
    "powdery mildew": {
        "tip": "Powdery mildew thrives in dry conditions with high humidity.",
        "severity": "Moderate",
        "treatment": "Apply sulfur-based fungicide or neem oil spray.",
    },
    "rust": {
        "tip": "Rust fungi spread via wind. Remove infected leaves immediately.",
        "severity": "Moderate",
        "treatment": "Apply triazole-based fungicide every 10–14 days.",
    },
    "mosaic virus": {
        "tip": "Mosaic virus is spread by aphids. Control insect vectors.",
        "severity": "High",
        "treatment": "No cure — remove infected plants. Control aphids with insecticide.",
    },
}

def get_disease_info(disease: str) -> dict:
    disease_lower = disease.lower()
    for key, info in DISEASE_INFO.items():
        if key in disease_lower:
            return info
    return {
        "tip": "Consult a local agronomist for specific treatment advice.",
        "severity": "Unknown",
        "treatment": "Seek professional diagnosis.",
    }

SEVERITY_COLOR = {
    "None":     "#4ade80",
    "Low":      "#facc15",
    "Moderate": "#fb923c",
    "High":     "#f87171",
    "Unknown":  "#9ca3af",
}

# ── Load model & class indices ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("PlantDisease.h5")

@st.cache_resource
def load_class_indices():
    with open("class_indices.json", "r") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}

# ── Helpers ───────────────────────────────────────────────────────────────────
IMG_SIZE = (224, 224)

def preprocess(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)

def predict(image: Image.Image):
    arr = preprocess(image)
    preds = model.predict(arr)
    idx = int(np.argmax(preds, axis=1)[0])
    confidence = float(np.max(preds)) * 100
    label = class_indices[idx]
    return label, confidence, preds[0]

def format_label(raw: str):
    parts = raw.split("___")
    plant   = parts[0].replace("_", " ") if len(parts) > 0 else raw
    disease = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"
    return plant, disease

def apply_image_adjustment(image: Image.Image, brightness, contrast, sharpness) -> Image.Image:
    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    image = ImageEnhance.Sharpness(image).enhance(sharpness)
    return image

# ── Load resources ────────────────────────────────────────────────────────────
with st.spinner("🌿 Loading model..."):
    model       = load_model()
    class_indices = load_class_indices()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🌿 Plant Disease Detector")
st.markdown("Upload a leaf image to instantly detect plant diseases using deep learning.")
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    st.markdown("### 🖼️ Image Adjustments")
    brightness = st.slider("Brightness", 0.5, 2.0, 1.0, 0.1)
    contrast   = st.slider("Contrast",   0.5, 2.0, 1.0, 0.1)
    sharpness  = st.slider("Sharpness",  0.5, 2.0, 1.0, 0.1)

    st.divider()

    st.markdown("### 📊 Chart Type")
    chart_type = st.radio(
        "Top predictions as:",
        ["Bar Chart", "Pie Chart", "Gauge"],
        index=0,
    )

    st.divider()

    st.markdown("### 🔬 Top N Predictions")
    top_n = st.slider("Show top N", 3, 10, 5)

    st.divider()
    st.caption("Model: CNN · PlantVillage Dataset · 224×224px input")

# ── Main layout ───────────────────────────────────────────────────────────────
upload_col, result_col = st.columns([1, 1], gap="large")

with upload_col:
    st.markdown("### 📤 Upload Leaf Image")
    uploaded_file = st.file_uploader(
        "Supported formats: JPG, JPEG, PNG, WEBP",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        original_image = Image.open(uploaded_file)
        display_image  = apply_image_adjustment(original_image, brightness, contrast, sharpness)

        st.image(display_image, caption="📷 Preview (adjustments applied for display only)", use_container_width=True)

        # Image metadata
        with st.expander("🔍 Image Details"):
            w, h = original_image.size
            st.markdown(f"- **Filename:** {uploaded_file.name}")
            st.markdown(f"- **Resolution:** {w} × {h} px")
            st.markdown(f"- **Format:** {original_image.format or 'N/A'}")
            st.markdown(f"- **Mode:** {original_image.mode}")
            st.markdown(f"- **File size:** {uploaded_file.size / 1024:.1f} KB")
    else:
        st.info("⬆️ Upload a leaf image to get started.")

# ── Results column ────────────────────────────────────────────────────────────
with result_col:
    st.markdown("### 🔍 Analysis Results")

    if uploaded_file:
        # Progress bar animation
        progress_bar = st.progress(0, text="Preprocessing image…")
        for i in range(40):
            time.sleep(0.01)
            progress_bar.progress(i + 1, text="Preprocessing image…")

        label, confidence, all_probs = predict(original_image)

        for i in range(40, 100):
            time.sleep(0.005)
            progress_bar.progress(i + 1, text="Running inference…")

        progress_bar.empty()

        plant, disease = format_label(label)
        is_healthy     = "healthy" in disease.lower()
        info           = get_disease_info(disease)
        severity       = info["severity"]
        sev_color      = SEVERITY_COLOR.get(severity, "#9ca3af")

        # ── Result card ───────────────────────────────────────────────────────
        card_class  = "result-card"
        label_class = "big-label" if is_healthy else "big-label-red"
        pill_class  = "confidence-pill" if is_healthy else "confidence-pill-red"
        icon        = "✅" if is_healthy else "⚠️"

        st.markdown(f"""
        <div class="{card_class}">
            <div class="sub-label">PLANT</div>
            <div class="big-label">{plant}</div>
            <div class="sub-label" style="margin-top:14px">CONDITION</div>
            <div class="{label_class}">{icon} {disease}</div>
            <div class="{pill_class}">{confidence:.1f}% confidence</div>
        </div>
        """, unsafe_allow_html=True)

        # Severity + Treatment
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div style="background:#1a1a2e;border-radius:12px;padding:16px;text-align:center;">
                <div style="color:#9ca3af;font-size:0.8rem;text-transform:uppercase;">Severity</div>
                <div style="color:{sev_color};font-size:1.6rem;font-weight:800;">{severity}</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div style="background:#1a1a2e;border-radius:12px;padding:16px;text-align:center;">
                <div style="color:#9ca3af;font-size:0.8rem;text-transform:uppercase;">Status</div>
                <div style="color:{'#4ade80' if is_healthy else '#f87171'};font-size:1.6rem;font-weight:800;">{'Healthy' if is_healthy else 'Diseased'}</div>
            </div>
            """, unsafe_allow_html=True)

        # Treatment tip
        st.markdown(f"""
        <div class="tip-box">
            💡 <strong>Treatment:</strong> {info['treatment']}<br><br>
            🌱 <strong>Tip:</strong> {info['tip']}
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="background:#1a1a2e;border-radius:16px;padding:40px;text-align:center;color:#6b7280;">
            <div style="font-size:3rem;">🌿</div>
            <div style="font-size:1rem;margin-top:12px;">Results will appear here after upload</div>
        </div>
        """, unsafe_allow_html=True)

# ── Charts section ────────────────────────────────────────────────────────────
if uploaded_file:
    st.divider()
    st.markdown("### 📊 Prediction Breakdown")

    top_idx   = np.argsort(all_probs)[::-1][:top_n]
    top_names = ["{} — {}".format(*format_label(class_indices[i])) for i in top_idx]
    top_probs = [float(all_probs[i]) * 100 for i in top_idx]
    colors    = ["#4ade80" if "healthy" in n.lower() else "#f87171" for n in top_names]

    if chart_type == "Bar Chart":
        fig = go.Figure(go.Bar(
            x=top_probs[::-1],
            y=top_names[::-1],
            orientation="h",
            marker_color=colors[::-1],
            text=[f"{p:.1f}%" for p in top_probs[::-1]],
            textposition="outside",
        ))
        fig.update_layout(
            plot_bgcolor="#0f1117",
            paper_bgcolor="#0f1117",
            font_color="#d1d5db",
            xaxis=dict(title="Confidence (%)", gridcolor="#1f2937", range=[0, max(top_probs) * 1.2]),
            yaxis=dict(title=""),
            margin=dict(l=10, r=40, t=20, b=20),
            height=80 + top_n * 45,
        )
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Pie Chart":
        fig = px.pie(
            values=top_probs,
            names=top_names,
            color_discrete_sequence=px.colors.sequential.Greens_r,
            hole=0.4,
        )
        fig.update_layout(
            paper_bgcolor="#0f1117",
            font_color="#d1d5db",
            margin=dict(l=10, r=10, t=20, b=20),
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Gauge":
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=confidence,
            title={"text": f"{plant} — {disease}", "font": {"color": "#d1d5db"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#d1d5db"},
                "bar":  {"color": "#4ade80" if is_healthy else "#f87171"},
                "bgcolor": "#1f2937",
                "steps": [
                    {"range": [0,  50], "color": "#1f2937"},
                    {"range": [50, 75], "color": "#1c3028"},
                    {"range": [75, 100], "color": "#14532d"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 3},
                    "thickness": 0.75,
                    "value": confidence,
                },
            },
            number={"suffix": "%", "font": {"color": "#4ade80" if is_healthy else "#f87171"}},
        ))
        fig.update_layout(
            paper_bgcolor="#0f1117",
            font_color="#d1d5db",
            height=350,
            margin=dict(l=30, r=30, t=60, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── All predictions table ─────────────────────────────────────────────────
    with st.expander("📋 View All Class Probabilities"):
        all_idx   = np.argsort(all_probs)[::-1]
        all_names = ["{} — {}".format(*format_label(class_indices[i])) for i in all_idx]
        all_p     = [float(all_probs[i]) * 100 for i in all_idx]

        import pandas as pd
        df = pd.DataFrame({"Class": all_names, "Confidence (%)": [f"{p:.2f}%" for p in all_p]})
        st.dataframe(df, use_container_width=True, height=300)

    # ── Download result ───────────────────────────────────────────────────────
    st.divider()
    result_text = f"""Plant Disease Detection Report
==============================
Plant     : {plant}
Condition : {disease}
Confidence: {confidence:.1f}%
Severity  : {severity}
Treatment : {info['treatment']}
Tip       : {info['tip']}

Top {top_n} Predictions:
""" + "\n".join([f"  {n}: {p:.2f}%" for n, p in zip(top_names, top_probs)])

    st.download_button(
        label="⬇️ Download Report (.txt)",
        data=result_text,
        file_name=f"plant_disease_report_{plant.replace(' ', '_')}.txt",
        mime="text/plain",
    )