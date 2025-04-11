import streamlit as st
from PIL import Image
from src.object_detection import detect_objects
from src.diff_analyzer import compare_detections
from src.summary_generator import generate_summary
from src.pixel_comparison import detect_pixel_differences  

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
        before_objects, before_annotated = detect_objects(before_pil)
        after_objects, after_annotated = detect_objects(after_pil)
        
    # Add OpenCV-based detection
    with st.spinner("Detecting pixel-level differences..."):
        pixel_changes_detected, num_changes, diff_image = detect_pixel_differences(before_pil, after_pil)
    
    st.subheader("📦 Detected Changes (YOLOv8)")
    col1, col2 = st.columns(2)
    with col1:
        st.image(before_annotated, caption="Before", use_container_width=True)
    with col2:
        st.image(after_annotated, caption="After", use_container_width=True)
    
    #OpenCV visualization for detected changes
    if pixel_changes_detected:
        st.subheader("📊 Pixel-level Differences (OpenCV)")
        st.image(diff_image, caption="Changes Highlighted", use_container_width=True)
    
    with st.spinner("Analyzing differences..."):
        diff_description = compare_detections(before_objects, after_objects)
        
        if pixel_changes_detected:
            diff_description += f" Additionally, {num_changes} areas with pixel-level differences were detected using OpenCV."
        
    with st.spinner("Generating summary..."):
        summary = generate_summary(diff_description)
        
    st.subheader("📄 Summary of Changes:")
    st.write(summary)
