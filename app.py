import streamlit as st
import numpy as np
import json
from PIL import Image
import io
from datetime import datetime
import torch

# Import custom prediction functions
from utils_pytorch import load_model, predict_disease, get_disease_info, CLASS_NAMES

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION & CUSTOM STYLING
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AgriVision AI - Crop Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": None,
        "Report a bug": None,
        "About": None
    }
)
with st.sidebar:
    st.markdown(
        "<p style='color:#1A1A1A;font-weight:600;'>🌐 Language</p>",
        unsafe_allow_html=True
    )

    language = st.selectbox(
        "",
        ["English", "తెలుగు", "हिन्दी"]
    )
translations = {
    "English": {
        "upload": "Choose a leaf image",
        "analyze": "🔍 Analyze Image",
        "reset": "🔄 Reset",
        "preview": "📤 Upload & Preview",
        "analysis": "🔬 Inference Analysis",
        "complete": "✅ Analysis Complete!",
        "profile": "📋 Profile",
        "treatment": "💊 Treatment",
        "prevention": "🛡️ Prevention",
        "confidence_score": "Confidence Score",
        "top_predictions": "Top 5 Predictions",
        "confidence_text": "Detected with",
        "confidence_end": "% confidence",
        "confidence_analysis": "Confidence Analysis",
        "upload_desc": "Upload a clear image of a crop leaf",
        "supported_crops": "(Corn, Potato, or Tomato)",
        "begin_analysis": "⬅️ Upload an image to begin analysis",
        "preview_placeholder": "Image preview will appear here",
        "sample_guide": "Sample Image Guide",
        "project_info": "PROJECT INFORMATION",
        "internship_details": "INTERNSHIP DETAILS",
        "model_architecture": "MODEL ARCHITECTURE",
        "dataset_metrics": "DATASET METRICS",
        "configuration": "CONFIGURATION",
        "confidence_threshold": "Confidence Threshold (%)"
    },

    "తెలుగు": {
        "upload": "ఆకు చిత్రాన్ని ఎంచుకోండి",
        "analyze": "🔍 విశ్లేషించండి",
        "reset": "🔄 రీసెట్",
        "preview": "📤 అప్లోడ్ & ప్రివ్యూ",
        "analysis": "🔬 విశ్లేషణ",
        "complete": "✅ విశ్లేషణ పూర్తైంది!",
        "profile": "📋 వివరాలు",
        "treatment": "💊 చికిత్స",
        "prevention": "🛡️ నివారణ",
        "confidence_score": "నమ్మక స్థాయి",
        "top_predictions": "టాప్ 5 అంచనాలు",
        "confidence_text": "",
        "confidence_end": "% నమ్మక స్థాయితో గుర్తించబడింది",
        "confidence_analysis": "నమ్మక విశ్లేషణ",
        "upload_desc": "పంట ఆకు యొక్క స్పష్టమైన చిత్రాన్ని అప్లోడ్ చేయండి",
        "supported_crops": "(మొక్కజొన్న, బంగాళాదుంప లేదా టమోటా)",

        "begin_analysis": "⬅️ విశ్లేషణ ప్రారంభించడానికి చిత్రాన్ని అప్లోడ్ చేయండి",

        "preview_placeholder": "చిత్ర ప్రివ్యూ ఇక్కడ కనిపిస్తుంది",

        "sample_guide": "నమూనా చిత్ర మార్గదర్శిని",

        "project_info": "ప్రాజెక్ట్ సమాచారం",
        "internship_details": "ఇంటర్న్‌షిప్ వివరాలు",
        "model_architecture": "మోడల్ నిర్మాణం",
        "dataset_metrics": "డేటాసెట్ వివరాలు" ,
        "framework": "ఫ్రేమ్‌వర్క్",
        "base_model": "బేస్ మోడల్",
        "input_size": "ఇన్‌పుట్ పరిమాణం",
        "output_classes": "అవుట్‌పుట్ తరగతులు",
        "source": "మూలం",
        "crops_covered": "మద్దతు ఉన్న పంటలు",
        "training_images": "శిక్షణ చిత్రాలు",
        "disease_categories": "వ్యాధి వర్గాలు",
        "configuration": "కాన్ఫిగరేషన్",
        "confidence_threshold": "నమ్మక పరిమితి (%)"
        },

    "हिन्दी": {
        "upload": "पत्ती की छवि चुनें",
        "analyze": "🔍 विश्लेषण करें",
        "reset": "🔄 रीसेट",
        "preview": "📤 अपलोड और पूर्वावलोकन",
        "analysis": "🔬 विश्लेषण",
        "complete": "✅ विश्लेषण पूर्ण!",
        "profile": "📋 विवरण",
        "treatment": "💊 उपचार",
        "prevention": "🛡️ रोकथाम",
        "confidence_score": "विश्वास स्तर",
        "top_predictions": "शीर्ष 5 भविष्यवाणियाँ",
        "confidence_text": "पहचाना गया",
        "confidence_end": "% विश्वास के साथ",
        "confidence_analysis": "विश्वास विश्लेषण",
        "upload_desc": "फसल की पत्ती की स्पष्ट छवि अपलोड करें",
        "supported_crops": "(मक्का, आलू या टमाटर)",

        "begin_analysis": "⬅️ विश्लेषण शुरू करने के लिए छवि अपलोड करें",

        "preview_placeholder": "छवि पूर्वावलोकन यहाँ दिखाई देगा",
 
        "sample_guide": "नमूना छवि मार्गदर्शिका",

        "project_info": "परियोजना जानकारी",
        "internship_details": "इंटर्नशिप विवरण",
        "model_architecture": "मॉडल संरचना",
        "dataset_metrics": "डेटासेट विवरण",
        "framework": "फ्रेमवर्क",
        "base_model": "बेस मॉडल",
        "input_size": "इनपुट आकार",
        "output_classes": "आउटपुट वर्ग",
        "source": "स्रोत",
        "crops_covered": "समर्थित फसलें",
        "training_images": "प्रशिक्षण चित्र",
        "disease_categories": "रोग श्रेणियाँ",
        "configuration": "कॉन्फ़िगरेशन",
        "confidence_threshold": "विश्वास सीमा (%)"
        }
}

text = translations[language]
# Premium custom CSS styling - simplified and robust for both themes
custom_css = """
<style>
    /* Universal base styles */
    * {
        --primary-green: #2E7D32;
        --light-bg: #F8F9FA;
        --dark-text: #1A1A1A;
    }

    /* Ensure readable text everywhere */
    body, .main, .stApp {
        background-color: #F8F9FA;
    }

    /* Force text readability */
    h1, h2, h3, h4, h5, h6 {
        color: #1A1A1A !important;
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    p, span, div, li, a, label {
        color: #333333 !important;
    }

    .stMarkdown {
        color: #1A1A1A !important;
    }

    .stMarkdown p {
        color: #333333 !important;
    }

    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #1A1A1A !important;
    }

    /* Header styling */
    h1 {
        border-bottom: 3px solid #2E7D32;
        padding-bottom: 12px;
        margin-bottom: 24px;
    }

    /* Language Dropdown */

    .stSelectbox > div > div {
        background-color: #FFFFFF !important;
        color: #1A1A1A !important;
        border: 1px solid #2E7D32 !important;
        border-radius: 8px !important;
    }

    .stSelectbox label {
        color: #1A1A1A !important;
        font-weight: 600;
    }

    /* Container styling */
    .stContainer {
        background-color: transparent;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 16px;
    }

    /* Premium card */
    .premium-card {
        background: rgba(46, 125, 50, 0.08);
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #2E7D32;
        border-top: 1px solid rgba(46, 125, 50, 0.1);
        margin-bottom: 16px;
        color: #1A1A1A;
    }

    .premium-card h2, .premium-card p {
        color: #1A1A1A !important;
    }

    /* Sidebar sections */
    .sidebar-section {
        background-color: #FFFFFF;
        border-left: 4px solid #2E7D32;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        padding: 16px;
        border-radius: 10px;
        margin-bottom: 16px;
        border-left: 4px solid #2E7D32;
        border-top: 1px solid rgba(46, 125, 50, 0.1);
    }

    .sidebar-section p, .sidebar-section strong, .sidebar-section span {
        color: #1A1A1A !important;
        font-weight: 500;
    }

    .sidebar-title {
        color: #2E7D32 !important;
        font-weight: 700;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 10px;
    }

    /* Info box */
    .info-box {
        background-color: rgba(46, 125, 50, 0.08);
        padding: 16px;
        border-radius: 8px;
        border-left: 3px solid #2E7D32;
        color: #1A1A1A !important;
    }

    .info-box p, .info-box strong {
        color: #1A1A1A !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background-color: transparent;
        border-bottom: 2px solid #E0E0E0;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 0;
        padding: 12px 20px;
        color: #666666;
        font-weight: 600;
        border-bottom: 3px solid transparent;
        margin-right: 0;
        border-top: none;
    }

    .stTabs [aria-selected="true"] {
        background-color: transparent;
        color: #2E7D32 !important;
        border-bottom-color: #2E7D32;
    }

    .stTabs [data-baseweb="tab-panel"] {
        background-color: rgba(46, 125, 50, 0.04);
        padding: 24px;
        border-radius: 8px;
        margin-top: 8px;
    }

    .stTabs [data-baseweb="tab-panel"] p,
    .stTabs [data-baseweb="tab-panel"] h3,
    .stTabs [data-baseweb="tab-panel"] li,
    .stTabs [data-baseweb="tab-panel"] strong,
    .stTabs [data-baseweb="tab-panel"] span {
        color: #1A1A1A !important;
    }

    /* Buttons */
    .stButton > button {
        background-color: #2E7D32;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 12px 32px;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-size: 14px;
        box-shadow: 0 2px 8px rgba(46, 125, 50, 0.15);
    }

    .stButton > button:hover {
        background-color: #1b5e20;
        box-shadow: 0 4px 16px rgba(46, 125, 50, 0.25);
    }

    /* File uploader */
    .stFileUploader {
    background-color: #FFFFFF !important;
    border: 2px dashed #2E7D32 !important;
    border-radius: 10px;
    padding: 28px;
    }

    .stFileUploader label {
        color: #1A1A1A !important;
    }

    .stFileUploader section {
         background-color: #FFFFFF !important;
    }

    .stFileUploader button {
        background-color: #2E7D32 !important;
        color: white !important;
        border-radius: 8px !important;
    }

    /* Alerts */
    .stAlert {
        border-radius: 10px;
        border-left: 4px solid;
    }

    .stAlert p, .stAlert strong, .stAlert span {
        color: #1A1A1A !important;
    }

    /* Success/Warning/Error/Info boxes */
    .stSuccess, .stWarning, .stError, .stInfo {
        color: #1A1A1A !important;
    }

    .stSuccess p, .stWarning p, .stError p, .stInfo p {
        color: #1A1A1A !important;
    }

    /* Metric box */
    .metric-box {
        background: linear-gradient(135deg, #2E7D32 0%, #43A047 100%);
        color: white !important;
        padding: 24px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 12px;
        box-shadow: 0 4px 12px rgba(46, 125, 50, 0.2);
    }

    .metric-label, .metric-value {
        color: white !important;
    }

    /* Progress bar */
    .stProgress > div > div > div {
        background-color: #2E7D32;
    }

    /* Prediction ranks */
    .prediction-rank {
        color: #1A1A1A !important;
    }

    /* Code blocks */
    code {
        color: #1A1A1A !important;
        background-color: rgba(46, 125, 50, 0.05);
        padding: 2px 6px;
        border-radius: 3px;
    }

    /* Lists */
    ul, ol {
        margin: 10px 0;
    }

    li {
        color: #1A1A1A !important;
        margin: 5px 0;
        line-height: 1.6;
    }

    /* Divider */
    hr, .stDivider {
        border-color: #E0E0E0;
    }

    /* Placeholder/muted text */
    .placeholder-text {
        color: #999999;
        opacity: 0.7;
    }

    /* Selectbox main box */

    div[data-baseweb="select"] {
        background: white !important;
    }

    div[data-baseweb="select"] > div {
        background: white !important;
        color: black !important;
        border: 1px solid #2E7D32 !important;
    }

    /* Dropdown popup */

    ul[role="listbox"] {
        background: white !important;
        color: black !important;
    }

    ul[role="listbox"] li {
        background: white !important;
        color: black !important;
    }

    /* Selected option */

    [data-baseweb="popover"] {
        background: white !important;
    }

    [data-baseweb="menu"] {
        background: white !important;
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    if language == "తెలుగు":
        project_name = "అగ్రివిజన్ AI"
        project_desc = "అధునాతన పంట వ్యాధి గుర్తింపు"

    elif language == "हिन्दी":
        project_name = "एग्रीविज़न AI"
        project_desc = "उन्नत फसल रोग पहचान"

    else:
        project_name = "AgriVision AI"
        project_desc = "Advanced Crop Disease Detection
    st.markdown(f"""
    <div class="sidebar-section">
        <div class="sidebar-title">📍 {text["project_info"]}</div>
        <p style="font-size: 13px; line-height: 1.5; margin: 0;">
            <strong>{project_name}</strong><br>
            {project_desc}
        </p>
    </div>
    """, unsafe_allow_html=True)
    if language == "తెలుగు":
        internship_title = "APSSDC సమ్మర్ ఇంటర్న్‌షిప్"
        internship_desc = "వ్యవసాయ ML అప్లికేషన్"

    elif language == "हिन्दी":
        internship_title = "APSSDC समर इंटर्नशिप"
        internship_desc = "कृषि ML एप्लिकेशन"

    else:
        internship_title = "APSSDC Summer Internship"
        internship_desc = "Agricultural ML Application"
    st.markdown(f"""
    <div class="sidebar-section">
        <div class="sidebar-title">🎓 {text["internship_details"]}</div>
        <p style="font-size: 13px; line-height: 1.5; margin: 0;">
            <strong>{internship_title}</strong><br>
            {internship_desc}
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="sidebar-section">
        <div class="sidebar-title">🔬 {text["model_architecture"]}</div>
        <p style="font-size: 13px; line-height: 1.5; margin: 0;">
            <strong>{text["framework"]}:</strong> PyTorch<br>
            <strong>{text["base_model"]}:</strong> EfficientNet-B0<br>
            <strong>{text["input_size"]}:</strong> 224 × 224px<br>
            <strong>{text["output_classes"]}:</strong> 9 diseases
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="sidebar-section">
        <div class="sidebar-title">📊 {text["dataset_metrics"]}</div>
        <p style="font-size: 13px; line-height: 1.5; margin: 0;">
            <strong>{text["source"]}:</strong> PlantVillage Dataset<br>
            <strong>{text["crops_covered"]}:</strong> Corn, Potato, Tomato<br>
            <strong>{text["training_images"]}:</strong> 9,991<br>
            <strong>{text["disease_categories"]}:</strong> 9
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="sidebar-section">
        <div class="sidebar-title">⚙️ {text["configuration"]}</div>
    """, unsafe_allow_html=True)

    confidence_threshold = st.slider(
        text["confidence_threshold"],
        min_value=50,
        max_value=100,
        value=75,
        step=5,
        help="Minimum confidence required to display results"
    )

    gap_threshold = st.slider(
        "Top-2 Gap Threshold",
        min_value=0.05,
        max_value=0.30,
        value=0.15,
        step=0.01,
        help="Minimum gap between top-2 predictions"
    )

    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    st.markdown("""
    <div style="font-size: 11px; color: #999; text-align: center; margin-top: 20px;">
        <p style="margin: 4px 0;"><strong>Version 1.0</strong></p>
        <p style="margin: 4px 0;">Built with PyTorch & Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN PAGE HEADER
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# MAIN PAGE HERO BANNER WITH IMAGE
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────────────────────────────────────

st.image(
    "assets/Facebook Cover - AgriVision AI.png",
    use_container_width=True
)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DISEASE NAME TRANSLATIONS
# ─────────────────────────────────────────────────────────────────────────────

disease_names = {
    "Potato___healthy": {
        "English": "Potato - Healthy",
        "తెలుగు": "బంగాళాదుంప - ఆరోగ్యకరం",
        "हिन्दी": "आलू - स्वस्थ"
    },

    "Potato___Early_blight": {
        "English": "Potato - Early Blight",
        "తెలుగు": "బంగాళాదుంప - ఎర్లీ బ్లైట్",
        "हिन्दी": "आलू - अर्ली ब्लाइट"
    },

    "Potato___Late_blight": {
        "English": "Potato - Late Blight",
        "తెలుగు": "బంగాళాదుంప - లేట్ బ్లైట్",
        "हिन्दी": "आलू - लेट ब्लाइट"
    },

    "Tomato___healthy": {
        "English": "Tomato - Healthy",
        "తెలుగు": "టమోటా - ఆరోగ్యకరం",
        "हिन्दी": "टमाटर - स्वस्थ"
    },

    "Tomato___Early_blight": {
        "English": "Tomato - Early Blight",
        "తెలుగు": "టమోటా - ఎర్లీ బ్లైట్",
        "हिन्दी": "टमाटर - अर्ली ब्लाइट"
    },

    "Tomato___Late_blight": {
        "English": "Tomato - Late Blight",
        "తెలుగు": "టమోటా - లేట్ బ్లైట్",
        "हिन्दी": "टमाटर - लेट ब्लाइट"
    },

    "Corn_(maize)___healthy": {
        "English": "Corn - Healthy",
        "తెలుగు": "మొక్కజొన్న - ఆరోగ్యకరం",
        "हिन्दी": "मक्का - स्वस्थ"
    },

    "Corn_(maize)___Common_rust": {
        "English": "Corn - Common Rust",
        "తెలుగు": "మొక్కజొన్న - రస్ట్ వ్యాధి",
        "हिन्दी": "मक्का - कॉमन रस्ट"
    },

    "Corn_(maize)___Northern_Leaf_Blight": {
        "English": "Corn - Northern Leaf Blight",
        "తెలుగు": "మొక్కజొన్న - నార్తర్న్ లీఫ్ బ్లైట్",
        "हिन्दी": "मक्का - नॉर्दर्न लीफ ब्लाइट"
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# MODEL INITIALIZATION (CACHED)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model_cached():
    """Load and cache the PyTorch model."""
    try:
        # Load the trained model from the specified path
        model = load_model("model/agrivision_model.pth")
        return model
    except Exception as e:
        st.error(f"❌ Model Loading Error: {str(e)}")
        return None

# ─────────────────────────────────────────────────────────────────────────────
# INITIALIZE MODEL
# ─────────────────────────────────────────────────────────────────────────────

try:
    model = load_model_cached()
    if model is None:
        st.stop()
    model_loaded = True
except Exception as e:
    st.error(f"❌ Critical Error: Unable to load model - {str(e)}")
    st.info("💡 Ensure 'model/agrivision_model.pth' exists and is valid.")
    model_loaded = False
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# TWO-COLUMN LAYOUT: UPLOAD (LEFT) | ANALYSIS (RIGHT)
# ─────────────────────────────────────────────────────────────────────────────

col_left, col_right = st.columns([1, 1.2], gap="large")

# ─────────────────────────────────────────────────────────────────────────────
# LEFT COLUMN: FILE UPLOAD & PREVIEW
# ─────────────────────────────────────────────────────────────────────────────

with col_left:
    st.markdown(f"### {text['preview']}")

    st.markdown(f"""
    <div style="text-align: center; padding: 24px; font-size: 14px; line-height: 1.6; color: #555;">
        {text["upload_desc"]}<br>
        {text["supported_crops"]}
    </div>
    """, unsafe_allow_html=True)
    
       
    
    

    uploaded_file = st.file_uploader(
        text["upload"],
        type=["jpg","jpeg","png"],
        help=(
            "Recommended: Clear, well-lit images at least 200x200 pixels"
            if language == "English"
            else "కనీసం 200x200 పిక్సెల్స్ గల స్పష్టమైన చిత్రాన్ని ఉపయోగించండి"
            if language == "తెలుగు"
            else "कम से कम 200x200 पिक्सेल की स्पष्ट छवि का उपयोग करें"
        )
    )

    # Image preview
    if uploaded_file is not None:
        st.markdown("#### 🖼️ Image Preview")
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True, caption="Uploaded Image")

        # Image metadata
        img_size = uploaded_file.size / (1024 * 1024)  # Convert to MB
        st.markdown(f"""
        <div class="info-box">
            <p style="font-size: 13px; margin: 0; line-height: 1.6;">
                <strong>📄 File:</strong> {uploaded_file.name}<br>
                <strong>💾 Size:</strong> {img_size:.2f} MB<br>
                <strong>🏷️ Type:</strong> {uploaded_file.type}
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="text-align: center; padding: 48px 20px; font-size: 13px; color: #999;">
            📸 {text["preview_placeholder"]}
        </div>
        """, unsafe_allow_html=True)

    # Sample images guide
    with st.expander(f"📚 {text['sample_guide']}", expanded=False):
        if language == "తెలుగు":
            st.markdown("""
            **📸 నాణ్యత సూచనలు:**
            
            - మంచి వెలుతురు ఉండేలా చూడండి
            - నేపథ్యంపై కాకుండా ఆకుపై దృష్టి పెట్టండి
            - మసకబారిన లేదా నీడ ఉన్న చిత్రాలను నివారించండి
            - వ్యాధి ప్రభావిత ప్రాంతం స్పష్టంగా కనిపించేలా చూడండి
            
            **🌱 మద్దతు ఉన్న పంటలు:**
            
            - 🌽 మొక్కజొన్న (రస్ట్ వ్యాధి, నార్తర్న్ లీఫ్ బ్లైట్, ఆరోగ్యకరం)
            - 🥔 బంగాళాదుంప (ఎర్లీ బ్లైట్, లేట్ బ్లైట్, ఆరోగ్యకరం)
            - 🍅 టమోటా (ఎర్లీ బ్లైట్, లేట్ బ్లైట్, ఆరోగ్యకరం)
            """)
        elif language == "हिन्दी":
            st.markdown("""
            **📸 गुणवत्ता सुझाव:**
            
            - अच्छी रोशनी सुनिश्चित करें
            - पत्ती पर ध्यान केंद्रित करें, पृष्ठभूमि पर नहीं
            - धुंधली या छायादार तस्वीरों से बचें
            - प्रभावित क्षेत्र को स्पष्ट रूप से दिखाएँ

            **🌱 समर्थित फसलें:**

            - 🌽 मक्का (कॉमन रस्ट, नॉर्दर्न लीफ ब्लाइट, स्वस्थ)
            - 🥔 आलू (अर्ली ब्लाइट, लेट ब्लाइट, स्वस्थ)
            - 🍅 टमाटर (अर्ली ब्लाइट, लेट ब्लाइट, स्वस्थ)
            """)

        else:
            st.markdown("""
            **Quality Tips:**
            
            - Ensure good lighting
            - Focus on the leaf, not the background
            - Avoid blurry or shadowed images
            - Include the affected area in the frame

            **Supported Crops:**
            
            - 🌽 Corn (Common Rust, Northern Leaf Blight, Healthy)
            - 🥔 Potato (Early Blight, Late Blight, Healthy)
            - 🍅 Tomato (Early Blight, Late Blight, Healthy)
            """)

# ─────────────────────────────────────────────────────────────────────────────
# RIGHT COLUMN: INFERENCE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

with col_right:
    st.markdown(f"### {text['analysis']}")

    if uploaded_file is None:
        st.markdown("""
        <div style="text-align: center; padding: 48px 20px; font-size: 13px; color: #999;">
            ⬅️  Upload an image to begin analysis
        </div>
        """, unsafe_allow_html=True)
    else:
        # Analyze button
        analyze_col1, analyze_col2 = st.columns([1, 1])

        with analyze_col1:
            analyze_button = st.button(
                text["analyze"],
                use_container_width=True,
                key="analyze_btn"
            )

        with analyze_col2:
            reset_button = st.button(
                text["reset"],
                use_container_width=True,
                key="reset_btn"
            )

        if reset_button:
            st.rerun()

        if analyze_button:
            with st.spinner("🔄 Analyzing leaf image..."):
                try:
                    # Get predictions
                    predicted_class, confidence, all_probs = predict_disease(
                        model,
                        uploaded_file,
                        return_all=True
                    )

                    # Calculate top-2 gap for unknown rejection
                    top2_indices = np.argsort(all_probs)[-2:]
                    gap = all_probs[top2_indices[1]] - all_probs[top2_indices[0]]

                    # Validation checks
                    threshold_conf = confidence_threshold / 100.0

                    if confidence < threshold_conf or gap < gap_threshold:
                        st.markdown("""
                        <div style="background: rgba(229, 57, 53, 0.08); padding: 16px; border-radius: 10px; border-left: 4px solid #E53935; margin-bottom: 16px;">
                            <p style="color: #C62828; font-weight: 600; margin: 0 0 8px 0;">
                                ⚠️ Low Confidence Detection
                            </p>
                            <p style="font-size: 13px; color: #666; margin: 0;">
                                The model is uncertain about this image. This may indicate:
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                        reasons = []
                        if confidence < threshold_conf:
                            reasons.append(f"Confidence ({confidence*100:.1f}%) below threshold ({threshold_conf*100:.0f}%)")
                        if gap < gap_threshold:
                            reasons.append(f"Top-2 predictions too close (gap: {gap:.3f})")

                        for i, reason in enumerate(reasons, 1):
                            st.markdown(f"- {reason}")

                        st.warning("""
                        **Possible causes:**
                        - Not a Corn, Potato, or Tomato leaf
                        - Image is too blurry or poorly lit
                        - Leaf shows multiple diseases
                        - Unknown disease variant

                        **Action:** Upload a clearer image or consult an agricultural expert.
                        """)
                    else:
                        # SUCCESS: Confidence threshold met
                        display_name = disease_names.get(
                            predicted_class,
                            {}
                        ).get(
                            language,
                            predicted_class.replace('___', ' - ')
                        )

                        st.success(text["complete"])

                        # Confidence metric box
                        st.markdown(f"""
                        <div class="metric-box">
                            <div class="metric-label">{text["confidence_score"]}</div>
                            <div class="metric-value">{confidence*100:.2f}%</div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Confidence progress bar
                        st.progress(min(confidence, 1.0))

                        st.divider()

                        # Disease detected header
                        st.markdown(f"""
                        <div class="premium-card">
                            <h2 style="color: #2E7D32; margin: 0;">🧬 {display_name}</h2>
                            <p style="margin: 8px 0 0 0; font-size: 13px;">
                                {text["confidence_text"]} {confidence*100:.2f}{text["confidence_end"]}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                        # Tabbed interface for disease details
                        tab1, tab2, tab3, tab4 = st.tabs([
                            text["profile"],
                            text["treatment"],
                            text["prevention"],
                            text["analysis"]
                        ])

                        # Get disease information
                        disease_info = get_disease_info(
                            predicted_class,
                            language
                        )

                        with tab1:
                            st.markdown(
                                f"### {text['profile']}"
                            )
                            st.markdown(disease_info.get("description", "No description available."))

                            # Additional metadata
                            if "symptoms" in disease_info:
                                st.markdown("**Common Symptoms:**")
                                st.markdown(disease_info["symptoms"])

                            if "affected_crops" in disease_info:
                                st.markdown("**Affected Crops:**")
                                st.write(disease_info["affected_crops"])

                        with tab2:
                            st.markdown("### Treatment Recommendation")
                            st.markdown(disease_info.get("treatment", "No treatment information available."))

                            if "fungicides" in disease_info:
                                st.markdown("**Recommended Fungicides:**")
                                for fungicide in disease_info["fungicides"]:
                                    st.markdown(f"- {fungicide}")

                            if "dosage" in disease_info:
                                st.markdown("**Application Dosage:**")
                                st.code(disease_info["dosage"], language="text")

                        with tab3:
                            st.markdown("### Prevention Rules")
                            if "prevention" in disease_info:
                                st.markdown(disease_info["prevention"])
                            else:
                                if language == "తెలుగు":
                                    st.markdown("""
                                **సాధారణ నివారణ చర్యలు:**

                                - మొక్కల మధ్య సరైన దూరం ఉంచండి
                                - వ్యాధిగ్రస్త ఆకులను వెంటనే తొలగించండి
                                - వ్యాధి నిరోధక రకాలను ఉపయోగించండి
                                - పంటల మార్పిడి పద్ధతి పాటించండి
                                - ముందస్తుగా ఫంగిసైడ్లు వాడండి
                                - వ్యవసాయ పరికరాలను శుభ్రంగా ఉంచండి
                                - వాతావరణ పరిస్థితులను గమనించండి
                                """)

                                
                                elif language == "हिन्दी":
                                    st.markdown("""
                                **सामान्य रोकथाम उपाय:**

                                - पौधों के बीच उचित दूरी रखें
                                - संक्रमित पत्तियों को तुरंत हटाएँ
                                - रोग प्रतिरोधी किस्मों का उपयोग करें
                                - फसल चक्र अपनाएँ
                                - समय पर फफूंदनाशक का उपयोग करें
                                - कृषि उपकरणों को साफ रखें
                                - मौसम की स्थिति पर निगरानी रखें
                                """)
                                else:
                                    st.markdown("""
                                **General Prevention Practices:**
                                - Maintain proper crop spacing for air circulation
                                - Remove infected leaves immediately
                                - Use disease-resistant crop varieties
                                - Practice crop rotation (3-4 year cycle)
                                - Apply preventive fungicides before disease onset
                                - Keep tools and equipment sanitized
                                - Monitor weather conditions for disease favorable environment
                                """)

                        with tab4:
                            st.markdown("### Confidence Analysis")

                            # Top predictions breakdown
                            top_n = min(5, len(CLASS_NAMES))
                            top_indices = np.argsort(all_probs)[-top_n:][::-1]

                            st.markdown(
                                f"**{text['top_predictions']}:**"
                            )

                            for rank, idx in enumerate(top_indices, 1):
                                prob = all_probs[idx]
                                class_name = disease_names.get(
                                    CLASS_NAMES[idx],
                                    {}
                                ).get(
                                    language,
                                    CLASS_NAMES[idx].replace('___', ' - ')
                                )
                                bar_width = min(prob * 100, 100)

                                st.markdown(f"""
                                <div style="margin-bottom: 16px;">
                                    <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                                        <span class="prediction-rank" style="font-weight: 600; font-size: 13px;">{rank}. {class_name}</span>
                                        <span style="color: #2E7D32; font-weight: 600; font-size: 13px;">{prob*100:.2f}%</span>
                                    </div>
                                    <div style="background-color: rgba(46, 125, 50, 0.15); border-radius: 4px; height: 6px; overflow: hidden;">
                                        <div style="background: linear-gradient(90deg, #2E7D32 0%, #43A047 100%); height: 100%; width: {bar_width}%;"></div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)

                            st.markdown("---")
                            st.markdown(f"""
                            <div class="info-box">
                                <p style="font-size: 13px; color: #555; margin: 0;">
                                    <strong>Top-2 Gap:</strong> {gap:.3f}
                                    (Threshold: {gap_threshold:.3f})
                                </p>
                            </div>
                            """, unsafe_allow_html=True)

                        # Disclaimer
                        if language == "తెలుగు":
                            st.info("""
                        ⚠️ ముఖ్యమైన గమనిక

                       ఈ AI ఆధారిత వ్యవస్థ రైతులకు సహాయం చేయడానికి రూపొందించబడింది.
                       ఖచ్చితమైన నిర్ధారణ కోసం వ్యవసాయ నిపుణుడిని సంప్రదించండి.
                       """)

                        elif language == "हिन्दी":
                              st.info("""
                        ⚠️ महत्वपूर्ण सूचना

                        यह AI आधारित प्रणाली किसानों की सहायता के लिए बनाई गई है।
                        सटीक निदान के लिए कृषि विशेषज्ञ से सलाह लें।
                        """)
                        else:
                           st.info("""
                        ⚠️ Important Disclaimer

                        This is an AI-based classification system designed to assist agricultural professionals.
                        For accurate diagnosis and treatment recommendations, always consult a qualified agricultural
                        expert or extension officer. This system should be used as a supplementary tool only.
                        """)

                        # Timestamp
                        st.markdown(f"""
                        <p style="font-size: 12px; text-align: right; margin-top: 24px; color: #999;">
                            ✓ Analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                        </p>
                        """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"❌ Analysis Error: {str(e)}")
                    st.info("💡 Please ensure the image is a valid crop leaf and try again.")

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────

st.divider()

st.markdown("""
<div style="text-align: center; padding: 20px; font-size: 12px; color: #999;">
    <p style="margin: 0;">
        <strong>AgriVision AI</strong> | APSSDC Summer Internship Project<br>
        Powered by PyTorch & Streamlit | PlantVillage Dataset
    </p>
</div>
""", unsafe_allow_html=True)
