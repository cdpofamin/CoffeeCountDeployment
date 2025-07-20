import streamlit as st

def render_instruction():
    st.markdown("""
    <style>
    header, .st-emotion-cache-18ni7ap {
        display: none;
    }
    .block-container {
        padding-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    # 📌 GENERAL INSTRUCTIONS
    Follow these guidelines before uploading your orthomosaic image.

    ## 🗺️ PREPARE YOUR ORTHOMOSAIC
    - Ensure the image is fully-stitched and geo-referenced (not individual tiles).
    - Ground-sampling distance (GSD) should be 2–5 cm/pixel for optimal detection.

    ## 📁 CHECK FILE FORMAT & SIZE
    - Accepted formats: **GeoTIFF** (.tif / .tiff)
    - Must contain an embedded coordinate system (e.g., WGS 84 / EPSG:4326)

    ## 🖼️ VERIFY IMAGE SPECIFICATIONS
    - Resolution: 2–5 cm GSD
    - Bit depth: 8-bit or 16-bit
    - Projection: EPSG:4326 (WGS 84)

    ## ⬆️ UPLOAD & ANALYZE
    - Drag and drop the file or click **Browse** to select.
    - Once uploaded, press **Run Detection**.
    - Wait for the progress bar to complete.

    ## 📊 REVIEW RESULTS
    After processing, the system will show:
    - Total tree count
    - Tree classifications: Mature, Young, Dead
    - Tree spacing statistics (average, smallest, largest)
    - Annotated image
    - Download links (image, PDF report, GeoJSON, CSV)

    ## ❗ TROUBLESHOOTING
    - No detections? Check your image resolution or clarity.
    - Warnings? Recheck the image quality or metadata.
    - Still having issues? Contact info@maco-3d.com with your run log.

    ## 🔒 DATA PRIVACY
    - No user data is retained after each session.
    - Always save your detection results after each run.

    <p style='font-size:12px; text-align:center; margin-top:40px;'>
    &copy; 2025 Mercurio Aerospace Corporation. All Rights Reserved.License No. 2025-03-05
    </p>
    """, unsafe_allow_html=True)
