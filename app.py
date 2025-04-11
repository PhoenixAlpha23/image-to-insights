import streamlit as st
from PIL import Image
from src.object_detection import detect_objects
from src.diff_analyzer import compare_detections
from src.summary_generator import generate_summary

st.set_page_config(page_title="Image Diff Summarizer", layout="centered")
st.title("🧠 AI-Powered Image Difference Summarizer")

# Upload UI
before_img = st.file_uploader("Upload 'Before' Image", type=["jpg", "png", "jpeg"])
after_img = st.file_uploader("Upload 'After' Image", type=["jpg", "png", "jpeg"])

if before_img and after_img:
    before_pil = Image.open(before_img)
    after_pil = Image.open(after_img)

    st.image([before_pil, after_pil], caption=["Before", "After"], width=300)

    with st.spinner("Running YOLO object detection..."):
        before_objects = detect_objects(before_pil)
        after_objects = detect_objects(after_pil)

    with st.spinner("Analyzing differences..."):
        diff_description = compare_detections(before_objects, after_objects)

    with st.spinner("Generating summary..."):
        summary = generate_summary(diff_description)

    st.subheader("📄 Summary of Changes:")
    st.write(summary)
