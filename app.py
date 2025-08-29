import streamlit as st
import requests
from PIL import Image

st.title("🦷 Oral Cancer Detection 🦷")
st.write("Upload an oral cavity image, and the model will predict if it is **Normal**, **Cancer**, or **Doubtful**.")

uploaded_file = st.file_uploader("📂 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        backend_url = "https://oc-detect-2.onrender.com/predict/"
        files = {"file": uploaded_file.getvalue()}
        response = requests.post(backend_url, files={"file": ("image.jpg", uploaded_file.getvalue(), "image/jpeg")})

        if response.status_code == 200:
            result = response.json()
            st.success(f"🔎 Prediction: **{result['label']}** (Confidence: {result['confidence']:.2f})")
        else:
            st.error("❌ Failed to connect to backend")
