import streamlit as st
import os
import base64
import numpy as np
from io import BytesIO
from PIL import Image
import openai

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Design Style Classifier",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif; }

html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #e0e0e0;
}

[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(16px);
    border-right: 1px solid rgba(255,255,255,0.1);
}

[data-testid="stSidebar"] * { color: #e0e0e0 !important; }

.hero-banner {
    background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
    border-radius: 20px;
    padding: 48px 40px;
    margin-bottom: 32px;
    text-align: center;
    box-shadow: 0 20px 60px rgba(101,17,203,0.35);
}

.hero-banner h1 {
    font-size: 2.8rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0 0 12px 0;
    letter-spacing: -0.5px;
}

.hero-banner p {
    font-size: 1.1rem;
    color: rgba(255,255,255,0.85);
    margin: 0;
}

.glass-card {
    background: rgba(255,255,255,0.07);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 28px;
    margin-bottom: 20px;
}

.metric-card {
    background: linear-gradient(135deg, rgba(101,17,203,0.25), rgba(37,117,252,0.25));
    border: 1px solid rgba(101,17,203,0.4);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}

.metric-card .label {
    font-size: 0.8rem;
    color: rgba(255,255,255,0.6);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
}

.metric-card .value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #ffffff;
}

.prob-bar-wrap {
    margin: 10px 0;
}

.prob-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.88rem;
    color: rgba(255,255,255,0.8);
    margin-bottom: 4px;
}

.prob-bar-bg {
    background: rgba(255,255,255,0.1);
    border-radius: 8px;
    height: 10px;
    overflow: hidden;
}

.prob-bar-fill {
    height: 100%;
    border-radius: 8px;
    background: linear-gradient(90deg, #6a11cb, #2575fc);
    transition: width 0.6s ease;
}

.result-badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 20px;
    background: linear-gradient(90deg, #6a11cb, #2575fc);
    color: #fff;
    font-weight: 600;
    font-size: 0.9rem;
    margin-bottom: 10px;
}

.ai-analysis-box {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 12px;
    padding: 22px;
    font-size: 0.95rem;
    line-height: 1.75;
    color: #d0d0e0;
    white-space: pre-wrap;
}

.stButton button {
    background: linear-gradient(135deg, #6a11cb, #2575fc) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 12px 28px !important;
    font-size: 1rem !important;
    transition: all 0.3s ease !important;
    width: 100%;
}

.stButton button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(101,17,203,0.45) !important;
}

[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.05) !important;
    border: 2px dashed rgba(101,17,203,0.5) !important;
    border-radius: 12px !important;
}

div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea,
div.stSelectbox select {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    border-radius: 8px !important;
    color: #e0e0e0 !important;
}

.step-circle {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: linear-gradient(135deg, #6a11cb, #2575fc);
    color: white;
    font-weight: 700;
    font-size: 0.9rem;
    margin-right: 10px;
}

.step-row {
    display: flex;
    align-items: center;
    padding: 10px 0;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    font-size: 0.95rem;
    color: rgba(255,255,255,0.8);
}

.tip-box {
    background: rgba(37,117,252,0.12);
    border-left: 3px solid #2575fc;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    font-size: 0.88rem;
    color: rgba(255,255,255,0.75);
    margin-top: 12px;
}

h2, h3, h4 { color: #ffffff !important; }
p { color: rgba(255,255,255,0.8); }
hr { border-color: rgba(255,255,255,0.1); }
</style>
""", unsafe_allow_html=True)

# ─── Session State Init ────────────────────────────────────────────────────────
if "model" not in st.session_state:
    st.session_state.model = None
if "predict_results" not in st.session_state:
    st.session_state.predict_results = None
if "ai_analysis" not in st.session_state:
    st.session_state.ai_analysis = None

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    st.markdown("### 🤖 Model Settings")
    model_path = st.text_input(
        "YOLOv8 Model Path",
        value="best.pt",
        help="Path to your trained YOLOv8 classification model (.pt file). "
             "Train via the Training tab first, or upload a pre-trained model.",
    )

    st.markdown("---")
    st.markdown("### 🔑 OpenAI API Key")
    openai_key = st.text_input(
        "API Key",
        value="",
        type="password",
        placeholder="sk-proj-...",
        help="Required for GPT-4o AI analysis of the detected design.",
    )

    st.markdown("---")
    st.markdown("### 📋 How to Use")
    for i, step in enumerate([
        "Train the model (Training tab)",
        "Upload your design image",
        "Click **Classify**",
        "View probabilities & AI analysis",
    ], 1):
        st.markdown(
            f'<div class="step-row"><span class="step-circle">{i}</span>{step}</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="tip-box">💡 You can skip training if you already have a <b>best.pt</b> file from previous runs.</div>',
        unsafe_allow_html=True,
    )

# ─── Hero Banner ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <h1>🎨 Design Style Classifier</h1>
    <p>Classify design images into Modern, Minimalist, Flat, Vintage, or Retro styles<br>
       powered by YOLOv8 + GPT-4o Vision Analysis</p>
</div>
""", unsafe_allow_html=True)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_classify, tab_train, tab_about = st.tabs(["🔍 Classify", "🏋️ Train Model", "ℹ️ About"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CLASSIFY
# ══════════════════════════════════════════════════════════════════════════════
with tab_classify:
    col_upload, col_result = st.columns([1, 1], gap="large")

    # ── Upload ──────────────────────────────────────────────────────────────
    with col_upload:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a design image",
            type=["png", "jpg", "jpeg", "webp", "bmp"],
            label_visibility="collapsed",
        )

        if uploaded_file:
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption="Uploaded Design", use_container_width=True)

            run_ai = st.checkbox(
                "🤖 Run GPT-4o AI Analysis",
                value=True,
                help="Requires a valid OpenAI API Key in the sidebar.",
            )

            classify_btn = st.button("✨ Classify Design")
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Result ──────────────────────────────────────────────────────────────
    with col_result:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Classification Results")

        if uploaded_file and classify_btn:
            # ── Load model ──────────────────────────────────────────────────
            if not os.path.exists(model_path):
                st.error(
                    f"❌ Model not found at `{model_path}`. "
                    "Please train the model in the **Training** tab first, "
                    "or update the path in the sidebar."
                )
            else:
                with st.spinner("Loading model & classifying…"):
                    try:
                        from ultralytics import YOLO
                        model = YOLO(model_path)
                        st.session_state.model = model

                        results = model.predict(img, verbose=False)
                        st.session_state.predict_results = results

                    except Exception as e:
                        st.error(f"Error during classification: {e}")
                        results = None

                if st.session_state.predict_results:
                    res = st.session_state.predict_results[0]
                    names = res.names
                    probs = res.probs.data.tolist()

                    top_idx = probs.index(max(probs))
                    top_class = names[top_idx]
                    top_prob = max(probs)

                    # Badge + top metric
                    st.markdown(
                        f'<div class="result-badge">🏆 {top_class}</div>',
                        unsafe_allow_html=True,
                    )

                    m1, m2 = st.columns(2)
                    with m1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="label">Predicted Class</div>
                            <div class="value">{top_class}</div>
                        </div>""", unsafe_allow_html=True)
                    with m2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="label">Confidence</div>
                            <div class="value">{top_prob*100:.1f}%</div>
                        </div>""", unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("**All Class Probabilities**")

                    # Sort descending
                    sorted_pairs = sorted(zip(names.values(), probs), key=lambda x: x[1], reverse=True)
                    for cls_name, prob in sorted_pairs:
                        pct = prob * 100
                        st.markdown(f"""
                        <div class="prob-bar-wrap">
                            <div class="prob-label">
                                <span>{"✅ " if cls_name == top_class else ""}{cls_name}</span>
                                <span>{pct:.1f}%</span>
                            </div>
                            <div class="prob-bar-bg">
                                <div class="prob-bar-fill" style="width:{pct}%"></div>
                            </div>
                        </div>""", unsafe_allow_html=True)

                    # ── Annotated image ──────────────────────────────────
                    im_arr = res.plot()
                    im_rgb = im_arr[..., ::-1]
                    st.image(im_rgb, use_container_width=True)

                    # ── GPT-4o Analysis ───────────────────────────────────
                    if run_ai:
                        if not openai_key:
                            st.warning("⚠️ OpenAI API Key not set. Add it in the sidebar.")
                        else:
                            with st.spinner("Analyzing with GPT-4o…"):
                                try:
                                    # Use clean uploaded image for better AI vision evaluation
                                    buf = BytesIO()
                                    img.save(buf, format="PNG")
                                    encoded = base64.b64encode(buf.getvalue()).decode()

                                    client = openai.OpenAI(api_key=openai_key)
                                    response = client.chat.completions.create(
                                        model="gpt-4o",
                                        messages=[
                                            {
                                                "role": "system",
                                                "content": "Kamu adalah AI evaluator dan pakar desain grafis profesional."
                                            },
                                            {
                                                "role": "user",
                                                "content": [
                                                    {
                                                        "type": "text",
                                                        "text": (
                                                            f"Kamu adalah AI yang berperan sebagai evaluator desain grafis profesional.\n\n"
                                                            f"Tugas kamu adalah menganalisis kualitas sebuah desain (khususnya poster atau desain media sosial) "
                                                            f"berdasarkan prinsip dan teori desain grafis yang telah diakui secara akademis.\n\n"
                                                            f"Model sebelumnya mendeteksi desain ini sebagai '{top_class}' dengan tingkat kepercayaan {top_prob*100:.1f}%.\n\n"
                                                            f"Gunakan pendekatan berikut dalam analisis:\n"
                                                            f"1. Teori Gestalt (proximity, similarity, continuity, closure)\n"
                                                            f"2. Psikologi warna\n"
                                                            f"3. Aturan warna 60:30:10\n"
                                                            f"4. Golden ratio / komposisi visual\n"
                                                            f"5. Prinsip desain dasar:\n"
                                                            f"   - Balance (keseimbangan)\n"
                                                            f"   - Contrast (kontras)\n"
                                                            f"   - Visual hierarchy (hierarki visual)\n"
                                                            f"   - Alignment (perataan)\n"
                                                            f"   - Repetition (pengulangan)\n"
                                                            f"   - Proximity (kedekatan)\n"
                                                            f"   - Unity (kesatuan)\n"
                                                            f"6. Tipografi:\n"
                                                            f"   - Readability\n"
                                                            f"   - Legibility\n\n"
                                                            f"Berikan output dengan format berikut:\n\n"
                                                            f"1. Style desain (misalnya: Modern, Minimalist, Retro-Vintage, dll)\n"
                                                            f"2. Skor total (0–100)\n"
                                                            f"3. Kategori kualitas:\n"
                                                            f"   - Sangat Baik (85–100)\n"
                                                            f"   - Baik (70–84)\n"
                                                            f"   - Cukup (55–69)\n"
                                                            f"   - Kurang (<55)\n\n"
                                                            f"4. Analisis detail per aspek:\n"
                                                            f"   - Warna (berdasarkan psikologi warna & aturan 60:30:10)\n"
                                                            f"   - Layout (grid system & golden ratio)\n"
                                                            f"   - Tipografi (readability & hierarchy)\n"
                                                            f"   - Komposisi (Gestalt & balance)\n\n"
                                                            f"Untuk setiap aspek, berikan:\n"
                                                            f"- Skor\n"
                                                            f"- Penjelasan singkat berbasis teori\n"
                                                            f"- Kesimpulan (baik / cukup / kurang)\n\n"
                                                            f"5. Ringkasan dalam bentuk tabel (Aspek, Skor, Status)\n\n"
                                                            f"6. Rekomendasi perbaikan yang spesifik, praktis, dan dapat diterapkan\n\n"
                                                            f"Gunakan bahasa Indonesia yang formal namun tetap jelas dan mudah dipahami.\n"
                                                            f"Pastikan analisis bersifat objektif, tidak subjektif, dan selalu dikaitkan dengan teori desain."
                                                        ),
                                                    },
                                                    {
                                                        "type": "image_url",
                                                        "image_url": {
                                                            "url": f"data:image/png;base64,{encoded}"
                                                        },
                                                    },
                                                ],
                                            },
                                        ],
                                        max_tokens=1500,
                                    )
                                    ai_text = response.choices[0].message.content
                                    st.session_state.ai_analysis = ai_text
                                except Exception as e:
                                    st.error(f"GPT-4o error: {e}")

                    if st.session_state.ai_analysis:
                        st.markdown("---")
                        st.markdown("### 🤖 GPT-4o Design Analysis")
                        st.markdown(
                            f'<div class="ai-analysis-box">{st.session_state.ai_analysis}</div>',
                            unsafe_allow_html=True,
                        )

        elif not uploaded_file:
            st.info("⬅️ Upload a design image to get started.")

        st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — TRAIN
# ══════════════════════════════════════════════════════════════════════════════
with tab_train:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🏋️ Train YOLOv8 Classification Model")
    st.markdown(
        "This replicates the notebook training pipeline. "
        "The dataset is downloaded from Roboflow and a YOLOv8n-cls model is fine-tuned."
    )

    c1, c2 = st.columns(2)
    with c1:
        rf_api_key = st.text_input(
            "Roboflow API Key",
            value="",
            type="password",
            placeholder="Your Roboflow API key",
        )
        rf_workspace = st.text_input("Roboflow Workspace", value="hendaru")
        rf_project = st.text_input("Roboflow Project", value="classification-design")
        rf_version = st.number_input("Dataset Version", min_value=1, value=1)

    with c2:
        epochs = st.slider("Epochs", 5, 100, 20)
        imgsz = st.selectbox("Image Size", [320, 416, 512, 640], index=3)
        base_model = st.selectbox(
            "Base Model",
            ["yolov8n-cls.pt", "yolov8s-cls.pt", "yolov8m-cls.pt"],
            index=0,
        )
        output_dir = st.text_input("Output directory", value="runs/classify/train")

    train_btn = st.button("🚀 Start Training")
    st.markdown("</div>", unsafe_allow_html=True)

    if train_btn:
        if not rf_api_key:
            st.error("❌ Please enter your Roboflow API key.")
        else:
            log_area = st.empty()
            progress = st.progress(0, text="Initialising…")

            st.session_state.training_log = []

            def log(msg):
                st.session_state.training_log.append(msg)
                log_area.code("\n".join(st.session_state.training_log), language="bash")

            try:
                log("📦 Downloading dataset from Roboflow…")
                progress.progress(10, text="Downloading dataset…")
                from roboflow import Roboflow
                rf = Roboflow(api_key=rf_api_key)
                project_rf = rf.workspace(rf_workspace).project(rf_project)
                version_rf = project_rf.version(int(rf_version))
                dataset = version_rf.download("folder")
                log(f"✅ Dataset downloaded to: {dataset.location}")

                log(f"🤖 Loading base model: {base_model}")
                progress.progress(30, text="Loading model…")
                from ultralytics import YOLO
                model = YOLO(base_model)

                log(f"🏋️ Training for {epochs} epochs at {imgsz}px…")
                progress.progress(40, text="Training in progress…")

                results = model.train(
                    data=dataset.location,
                    epochs=epochs,
                    imgsz=imgsz,
                    project="runs/classify",
                    name="train",
                    exist_ok=True,
                )
                log("✅ Training complete!")
                progress.progress(90, text="Saving model…")

                best_path = os.path.join("runs", "classify", "train", "weights", "best.pt")
                if os.path.exists(best_path):
                    log(f"💾 Best model saved at: {best_path}")
                    st.success(
                        f"🎉 Training done! Best model: `{best_path}`\n\n"
                        "Update the **Model Path** in the sidebar to this path, then go to **Classify** tab."
                    )
                progress.progress(100, text="Done!")

            except Exception as e:
                st.error(f"Training failed: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### ℹ️ About This App")
    st.markdown("""
This app is a **Streamlit conversion** of the `Aduy_CNN.ipynb` notebook.
It classifies design images into one of 5 style categories using a YOLOv8
classification model, then provides detailed AI-powered feedback via GPT-4o.

---

#### 🏷️ Design Classes
| Class | Description |
|---|---|
| **Modern Design** | Clean lines, bold typography, minimalistic elements |
| **Minimalist Design** | Extreme simplicity, white space, functional elements |
| **Flat Design** | 2D imagery, bright colors, no gradients or shadows |
| **Vintage Design** | Retro colors, aged textures, classic typography |
| **Retro Design** | 80s/90s inspired, neon tones, geometric forms |

---

#### 🔧 Tech Stack
- **YOLOv8** (Ultralytics) — image classification
- **Roboflow** — dataset management & download
- **OpenAI GPT-4o** — vision-based design analysis
- **Streamlit** — interactive web interface
- **OpenCV / PIL** — image processing

---

#### 📂 Project Structure
```
Aduy/
├── app.py              ← This Streamlit app
├── Aduy_CNN.ipynb      ← Original notebook
├── extract.py          ← Extracted notebook code
├── sample.png          ← Example image
└── runs/               ← Training outputs (auto-created)
    └── classify/train/weights/best.pt
```
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:rgba(255,255,255,0.4); font-size:0.8rem;'>"
    "Design Style Classifier · Built with YOLOv8 + GPT-4o · Converted from Jupyter Notebook"
    "</p>",
    unsafe_allow_html=True,
)
