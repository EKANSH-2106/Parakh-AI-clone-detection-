import streamlit as st
import tempfile
import os
from detector import analyze_audio

st.set_page_config(page_title="परख", page_icon="🎙️", layout="wide")

st.markdown("""
<style>
    .stApp {
        background-color: #0A1929;
        color: #E6F1FF;
    }
    .badge-tag {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        border: 1px solid #38BDF844;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 12px;
        letter-spacing: 1px;
        color: #7DD3FC;
        margin-bottom: 18px;
    }
    .badge-tag .dot {
        width: 6px; height: 6px; border-radius: 50%;
        background: #38BDF8; display: inline-block;
    }
    h1.title {
        font-size: 52px;
        font-weight: 800;
        margin: 0 0 6px 0;
        line-height: 1.1;
    }
    p.subtitle {
        color: #94A3B8;
        font-size: 17px;
        margin-bottom: 28px;
    }
    .label-small {
        color: #64748B;
        font-size: 12px;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    .verdict-text { font-size: 26px; font-weight: 700; margin: 4px 0 16px 0; }
    .verdict-orange { color: #FB923C; }
    .verdict-green { color: #4ADE80; }
    .stat-value { font-size: 22px; font-weight: 700; color: #E6F1FF; }
    div[data-testid="stFileUploaderDropzone"] {
        background-color: #0F2438;
        border: 1px solid #38BDF844;
        border-radius: 12px;
    }
    div.stProgress > div > div > div > div {
        background: linear-gradient(90deg, #38BDF8, #22D3EE);
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #0F1E30;
        border-radius: 14px;
        border: 1px solid #1E3A5F !important;
    }
</style>
""", unsafe_allow_html=True)

left, right = st.columns([3, 2], gap="large")

with left:
    st.markdown('<div class="badge-tag"><span class="dot"></span>THE STANDARD FOR VOICE AUTHENTICITY</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="title">परख<br>AI Voice Clone<br>Detector.</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Drop in any voice clip. Get a real-time risk score on whether it\'s genuine or AI-cloned.</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a"], label_visibility="collapsed")

    result = None
    if uploaded_file is not None:
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        st.audio(uploaded_file)

        with st.spinner("Analyzing voice..."):
            result = analyze_audio(tmp_path)

        risk = result["risk_score"]
        if risk >= 0.65:
            action = "⚠️ High risk — recommend verifying via callback before proceeding."
        elif risk >= 0.5:
            action = "⚠️ Moderate risk — proceed with caution, consider secondary verification."
        else:
            action = "✅ Low risk — voice appears genuine."

        st.progress(risk)
        st.markdown(f"**Risk Score:** {risk:.1%} chance of being AI-cloned")
        st.markdown(action)

        os.remove(tmp_path)

with right:
    with st.container(border=True):
        if result is not None:
            risk = result["risk_score"]
            if risk >= 0.65:
                dot_color, label, verdict, vclass = "#FB923C", "SYNTHETIC DETECTED", "Likely fabricated.", "verdict-orange"
            elif risk >= 0.5:
                dot_color, label, verdict, vclass = "#FBBF24", "MODERATE RISK", "Possibly fabricated.", "verdict-orange"
            else:
                dot_color, label, verdict, vclass = "#4ADE80", "GENUINE VOICE", "Likely genuine.", "verdict-green"

            st.markdown(f'<div class="badge-tag"><span class="dot" style="background:{dot_color}"></span>{label}</div>', unsafe_allow_html=True)
            st.markdown('<div class="label-small">VERDICT</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="verdict-text {vclass}">{verdict}</div>', unsafe_allow_html=True)

            st.line_chart(result["segment_scores"], height=180)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<div class="label-small">CONFIDENCE</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="stat-value">{risk:.1%}</div>', unsafe_allow_html=True)
            with c2:
                st.markdown('<div class="label-small">SEGMENTS ANALYZED</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="stat-value">{len(result["segment_scores"])}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="label-small">AWAITING INPUT</div>', unsafe_allow_html=True)
            st.markdown('<div class="verdict-text" style="color:#64748B;">Upload a clip to see the verdict.</div>', unsafe_allow_html=True)