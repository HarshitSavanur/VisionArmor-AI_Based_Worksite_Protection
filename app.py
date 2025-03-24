import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import streamlit as st
import tempfile, os, glob, time
from ultralytics import YOLO

# Inject custom CSS for styling
st.markdown(
    """
    <style>
    .main {background-color: #f4f4f4; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }
    h1 {color: #333333; font-weight: bold;}
    h2 {color: #333333; margin-top: 20px;}
    h3 {color: #333333; margin-top: 15px;}
    .stButton>button {background-color: #007bff; color: white; border-radius: 8px; border: none; padding: 8px 16px; }
    .stButton>button:hover {background-color: #0056b3; }
    </style>
    """,
    unsafe_allow_html=True
)

#####################
# Video Processing  #
#####################
def process_video(video_path):
    """
    Process a video using YOLOv8 and return a dictionary containing:
      - 'video': the path to the processed video (raw output from YOLO)
      - 'images': a list of sample image file paths with detection results.
      
    The YOLO model saves the output to a subfolder in "runs/detect".
    """
    model = YOLO("models/best.pt")
    results = model.predict(source=video_path, show=False, save=True)
    time.sleep(3)
    
    output_folders = glob.glob("runs/detect/*")
    if not output_folders:
        return None
    latest_folder = max(output_folders, key=os.path.getmtime)
    
    video_files = glob.glob(os.path.join(latest_folder, "*.mp4"))
    if not video_files:
        video_files = glob.glob(os.path.join(latest_folder, "*.avi"))
    if not video_files:
        return None
    processed_video = video_files[0]
    
    image_files = glob.glob(os.path.join(latest_folder, "*.jpg"))
    if not image_files:
        image_files = glob.glob(os.path.join(latest_folder, "*.png"))
    image_files.sort()
    sample_images = image_files[:3]
    
    return {"video": processed_video, "images": sample_images}

#####################
# Image Processing  #
#####################
def process_image(image_path):
    """
    Process a single image using YOLOv8 and return:
      - processed_image: the path to the processed image.
      - detection_info: a dictionary containing counts for each of the 10 classes.
      
    The YOLO model saves the output image to a subfolder in "runs/detect".
    """
    model = YOLO("models/best.pt")
    results = model.predict(source=image_path, show=False, save=True)
    time.sleep(2)
    
    output_folders = glob.glob("runs/detect/*")
    if not output_folders:
        return None, None
    latest_folder = max(output_folders, key=os.path.getmtime)
    
    image_files = glob.glob(os.path.join(latest_folder, "*.jpg"))
    if not image_files:
        image_files = glob.glob(os.path.join(latest_folder, "*.png"))
    if not image_files:
        return None, None
    processed_image_path = image_files[0]
    
    # Define target classes (all in lowercase)
    target_classes = ['hardhat', 'mask', 'no-hardhat', 'no-mask', 'no-safety vest', 
                      'person', 'safety cone', 'safety vest', 'machinery', 'vehicle']
    detection_info = {cls: 0 for cls in target_classes}
    
    if results and len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        if boxes.cls is not None:
            # Convert each detected class to lowercase and count occurrences
            detected_classes = [results[0].names[int(cls)].lower() for cls in boxes.cls.cpu().numpy()]
            for d in detected_classes:
                if d in detection_info:
                    detection_info[d] += 1
                    
    return processed_image_path, detection_info

#####################
# Sidebar Navigation#
#####################
page = st.sidebar.radio("Navigation", ["Detection (Video)", "Image Detection", "Safety Measures Blog"])

#####################
# Video Detection   #
#####################
if page == "Detection (Video)":
    st.title("PPE Detection on Construction Sites - Video")
    st.write("Upload a video file (MP4 or AVI) to detect PPE using the YOLOv8 model.")
    
    if "processed_data" not in st.session_state:
        st.session_state.processed_data = None
    
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi"])
    
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.flush()
        
        st.subheader("Original Uploaded Video")
        st.video(tfile.name)
        
        if st.session_state.processed_data is None:
            with st.spinner("Processing video, please wait..."):
                st.session_state.processed_data = process_video(tfile.name)
        
        if st.session_state.processed_data is not None:
            st.success("Video processed successfully!")
            
            with open(st.session_state.processed_data["video"], "rb") as f:
                processed_video_bytes = f.read()
            st.download_button(
                label="Download Processed Video",
                data=processed_video_bytes,
                file_name="processed_video.mp4",
                mime="video/mp4"
            )
            
            if st.session_state.processed_data["images"]:
                st.subheader("Sample Processed Images")
                for img_file in st.session_state.processed_data["images"]:
                    st.image(img_file, caption=os.path.basename(img_file), use_container_width=True)
            else:
                st.warning("No processed images found to display.")
            
            st.subheader("Recommendations")
            st.markdown("""
            - **Lighting & Angle:** Ensure that the camera angle and lighting offer a clear view of the personnel.
            - **Model Calibration:** Adjust the detection confidence threshold if you experience too many false positives or negatives.
            - **Training Data:** Include varied examples of PPE in different environments to improve detection accuracy.
            - **Regular Updates:** Periodically update the model with new data to adapt to changing conditions on-site.
            """)
        else:
            st.error("Error: Could not locate the processed video. Please try again.")

#####################
# Image Detection   #
#####################
elif page == "Image Detection":
    st.title("PPE Detection System - Image")
    st.write("Upload an image file (JPG or PNG) to detect PPE using the YOLOv8 model.")
    
    uploaded_image = st.file_uploader("Choose an image file", type=["jpg", "png"])
    
    if uploaded_image is not None:
        img_temp = tempfile.NamedTemporaryFile(delete=False, suffix="." + uploaded_image.name.split('.')[-1])
        img_temp.write(uploaded_image.read())
        img_temp.flush()
        
        st.markdown("<h4>Uploaded Image:</h4>", unsafe_allow_html=True)
        st.image(img_temp.name, use_container_width=True)
        
        with st.spinner("Processing image, please wait..."):
            processed_img, detection_info = process_image(img_temp.name)
        
        if processed_img:
            st.markdown("<h4>Detected Results:</h4>", unsafe_allow_html=True)
            st.image(processed_img, use_container_width=True)
            
            # Display detection summary (only nonzero values)
            st.markdown("#### Detection Summary:")
            for cls, count in detection_info.items():
                if count > 0:
                    st.write(f"**{cls.capitalize()}:** {count}")
            
            # Trigger alerts for missing PPE based on "no-" classes
            alerts = []
            if detection_info["no-hardhat"] > 0:
                alerts.append(f"Warning: {detection_info['no-hardhat']} person(s) without a hardhat detected!")
            if detection_info["no-mask"] > 0:
                alerts.append(f"Warning: {detection_info['no-mask']} person(s) without a mask detected!")
            if detection_info["no-safety vest"] > 0:
                alerts.append(f"Warning: {detection_info['no-safety vest']} person(s) without a safety vest detected!")
            
            if alerts:
                for a in alerts:
                    st.error(a)
            else:
                st.success("All required PPE detected!")
        else:
            st.error("Error: Could not process the image. Please try again.")

##############################
# Safety Measures Blog Page  #
##############################
elif page == "Safety Measures Blog":
    st.title("Safety Measures Blog")
    st.markdown("""
    ### Construction Site Safety: Key Measures to Protect Your Workforce
    
    In today's dynamic construction environment, safety is paramount. Here are essential measures to ensure a secure work site:
    
    1. **Personal Protective Equipment (PPE):**  
       Ensure that every worker is equipped with the necessary PPE such as hardhats, masks, gloves, high-visibility vests, and protective footwear.
    
    2. **Site Inspections and Hazard Identification:**  
       Conduct thorough and regular site inspections to promptly identify and address potential hazards.
    
    3. **Training and Awareness:**  
       Provide regular training on safety practices and emergency procedures, emphasizing the correct use of PPE.
    
    4. **Emergency Preparedness:**  
       Establish clear emergency protocols. Make sure that emergency exits, first aid kits, and communication devices are easily accessible.
    
    5. **Regular Maintenance:**  
       Maintain all equipment and machinery properly with routine checks to prevent malfunctions that could cause accidents.
    
    By integrating these measures, construction sites can significantly reduce risks and protect workers, ensuring a safer work environment for everyone involved.
    """)
