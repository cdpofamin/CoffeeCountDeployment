import streamlit as st
import base64
import os

def render_about():
    st.set_page_config(page_title="About - Coffee Count Lite", layout="wide")

    # Hide sidebar, header, and footer
    st.markdown("""
    <style>
        div[data-testid="stSidebar"] {display: none;}
        header, footer {display: none !important;}
    </style>
    """, unsafe_allow_html=True)

    # Convert local image to Base64
    image_path = os.path.join("assets", "bg2.jpg")
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()

    # Custom styles and HTML content
    st.markdown(f"""
    <style>
    .about-container {{
        background: url("data:image/png;base64,{encoded}") no-repeat center center;
        background-size: cover;
        min-height: 100vh;
        padding: 40px 20px 60px 20px;
        color: white;
        text-shadow: 1px 1px 3px black;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: flex-start;
    }}

    .about-box {{
        background: rgba(0, 0, 0, 0.7);
        padding: 50px;
        border-radius: 15px;
        width: 70vw;
        max-width: 1100px;
        text-align: center;
    }}
    .about-box h2 {{
        font-size: 36px;
        margin-bottom: 25px;
    }}
    .about-box p, .about-box li {{
        font-size: 20px;
        line-height: 1.6;
        margin-bottom: 10px;
    }}
    .about-box ul {{
        padding-left: 20px;
        text-align: left;
        margin-top: 15px;
        margin-bottom: 20px;
    }}
    .about-btn {{
        margin-top: 25px;
        padding: 12px 30px;
        background: #ffffff;
        color: #000000;
        font-size: 18px;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        cursor: pointer;
    }}
    </style>

    <div class="about-container">
        <div class="about-box">
            <h2>Disclaimer, Accuracy, and Intended Use:</h2>
            <p><strong>Coffee Count Lite version 1.0</strong> employs computer vision algorithms to estimate tree counts and spacing of coffee trees from user-supplied orthomosaics. While the software is rigorously trained and validated, results are not guaranteed to be 100% accurate.</p>
            <p>Detection accuracy can be affected by factors outside our control, including—but not limited to—image resolution, lighting and shadow conditions, canopy overlap, terrain variation, and orthomosaic alignment errors.</p>
            <ul>
                <li>Verify figures through ground sampling or high-res imagery.</li>
                <li>Review the annotated preview provided by the software.</li>
                <li>Re-process the orthomosaic if accuracy warnings appear.</li>
            </ul>
            <p>Mercurio Aerospace Corporation (MACO) is not liable for losses or damages from reliance on the outputs. By proceeding, you acknowledge these limitations and agree to use results as decision support, not as professional judgment or field verification.</p>
            <button class="about-btn" onclick="window.location.href='/?detect'">Detect Trees</button>
        </div>
    </div>
    """, unsafe_allow_html=True)
