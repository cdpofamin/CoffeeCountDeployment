import os
import streamlit as st
from db import fetch_all_detections

def render_history():
    st.markdown("""
    <style>
        .block-container {
            padding-top: 0rem;
        }
        header, .st-emotion-cache-18ni7ap {
            display: none;
        }
    </style>
    """, unsafe_allow_html=True)

    st.title("📊 Past Detections History")

    try:
        records = fetch_all_detections()
    except Exception as e:
        st.warning(f"Database not initialized or table missing. Error: {e}")
        records = []

    if not records:
        st.info("No past detections recorded yet.")
        return

    for rec in records:
        # rec[13] = timestamp, rec[1] = filename
        with st.expander(f"{rec[1]} - {rec[13]}"):
            st.write(f"Total Trees: {rec[2]} | Young: {rec[3]} | Mature: {rec[4]} | Dead: {rec[5]}")
            st.write(f"Avg Spacing: {rec[6]:.2f} m | Smallest: {rec[7]:.2f} m | Largest: {rec[8]:.2f} m")

            download_available = False
            st.markdown("### 📥 Available Downloads")

            try:
                if rec[9] and os.path.exists(rec[9]):
                    st.download_button("Download Annotated Image", open(rec[9], "rb"), file_name="annotated.jpg", key=f"annot_img_{rec[0]}")
                    download_available = True
                if rec[10] and os.path.exists(rec[10]):
                    st.download_button("Download PDF Report", open(rec[10], "rb"), file_name="report.pdf", key=f"pdf_{rec[0]}")
                    download_available = True
                if rec[11] and os.path.exists(rec[11]):
                    st.download_button("Download GeoJSON", open(rec[11], "rb"), file_name="detections.geojson", key=f"geojson_{rec[0]}")
                    download_available = True
                if rec[12] and os.path.exists(rec[12]):
                    st.download_button("Download CSV", open(rec[12], "rb"), file_name="detections.csv", key=f"csv_{rec[0]}")
                    download_available = True
            except Exception as e:
                st.error(f"Some files not accessible for this record: {e}")

            if not download_available:
                st.info("No files available for download for this record.")

            st.markdown("---")
