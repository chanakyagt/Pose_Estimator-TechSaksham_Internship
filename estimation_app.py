import streamlit as st
from PIL import Image
import numpy as np
import cv2

# Constants and model setup
BODY_PARTS = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
    "LEye": 15, "REar": 16, "LEar": 17, "Background": 18
}

POSE_PAIRS = [
    ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
    ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
    ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
    ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]
]

width, height, inWidth, inHeight = 368, 368, 368, 368

def load_model():
    model_path = "graph_opt.pb"
    try:
        net = cv2.dnn.readNetFromTensorflow(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None
    return net

def process_and_annotate(image, net):
    frameWidth, frameHeight = image.shape[1], image.shape[0]
    net.setInput(cv2.dnn.blobFromImage(image, 1.0, (width, height), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]

    assert(len(BODY_PARTS) == out.shape[1])
    points = []
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > 0.2 else None)

    for pair in POSE_PAIRS:
        partFrom, partTo = pair
        idFrom, idTo = BODY_PARTS[partFrom], BODY_PARTS[partTo]
        if points[idFrom] and points[idTo]:
            cv2.line(image, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv2.ellipse(image, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            cv2.ellipse(image, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
    
    return image

def main():
    st.set_page_config(page_title="Pose Estimation Hub", page_icon="🤸", layout="wide")

    net = load_model()
    if net is None:
        return

    st.title("Pose Estimation Hub")
    st.markdown("""
    **Analyze human poses using advanced AI-based techniques.**
    """)

    st.markdown("---")

    st.subheader("Upload an Image")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    

    if uploaded_file:
        with st.spinner('Processing...'):
            image = np.array(Image.open(uploaded_file))
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            annotated_image = process_and_annotate(image_bgr, net)
            annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        st.markdown("### Results")
        st.markdown("Below are your original and annotated images:")

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_column_width=True)
        with col2:
            st.image(annotated_image_rgb, caption="Pose Annotated Image", use_column_width=True)

        st.success("Pose analysis completed! View the annotated image on the right.")
    else:
        st.info("Upload an image to start pose estimation.")

    st.markdown("""
    ---
    ### How It Works
    1. **Upload:** Choose an image above.
    2. **Process:** Our model processes it using [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose).
    3. **Results:** The annotated image is displayed alongside the original.
    """)

if __name__ == "__main__":
    main()
