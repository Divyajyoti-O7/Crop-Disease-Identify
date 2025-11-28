import streamlit as st
import tensorflow as tf
import numpy as np
from disease_info import  disease_data

# Function to load and make predictions using the trained model
def predict_disease(image_file):
    model = tf.keras.models.load_model("models/crop_disease_identification_model_pwp.keras")
    image = tf.keras.preprocessing.image.load_img(image_file, target_size=(160, 160))
    image_arr = tf.keras.preprocessing.image.img_to_array(image)
    image_arr = np.expand_dims(image_arr, axis=0)  # Convert to batch format
    prediction = model.predict(image_arr)
    return np.argmax(prediction)  # Return the predicted class index


# Sidebar navigation
st.sidebar.title("Navigation")
selected_page = st.sidebar.selectbox("Go to", ["Home", "About", "Detect Disease"])

# Home Page
if selected_page == "Home":
    st.header("🌱 Plant Disease Detection for Healthy Farmig")
    st.image("home_page.jpeg", use_container_width=True)
    st.markdown("""
    Welcome to our smart **Plant Disease Detection System**! 

    This tool allows farmers, gardeners, and researchers to identify plant diseases from images. Upload a picture of a plant, and our AI will analyze it to detect possible infections or issues.

    ###  How It Works
    - **Step 1:** Head over to the **Detect Disease** tab and upload a plant leaf image.
    - **Step 2:** The system processes the image using a deep learning model trained on thousands of examples.
    - **Step 3:** You’ll instantly receive the predicted disease class and can take appropriate action.

    ###  Features
    - **High Accuracy:** Powered by a state-of-the-art neural network model.
    - **Fast Results:** Get predictions in real-time.
    - **Easy to Use:** Clean and simple interface.

    ### Get Started
    Use the sidebar to navigate to the **Detect Disease** section and test it out. You can also learn more about this project from the **About** tab.
    """)

# About Page
elif selected_page == "About":
    st.header(" About This Project")
    st.markdown("""
    ### 📊 Dataset Information
    This project uses an enhanced version of a popular plant disease image dataset. Data augmentation was applied to increase variety.

    - **Total Images:** 87,000+ RGB images
    - **Classes:** 38 categories including healthy and infected leaves
    - **Structure:** 80% for training, 10% for training, 10% for validation.

    **Folders:**
    1. `train/` 
    2. `val/`  
    3. `test/`  

    """)

# Disease Detection Page
elif selected_page == "Detect Disease":
    st.header("🦠 Plant Disease Detection")
    uploaded_image = st.file_uploader("Upload a leaf image to analyze:")

    if st.button("Display Image"):
        if uploaded_image:
            st.image(uploaded_image, use_column_width=True)

    if st.button("Detect"):
        if uploaded_image:
            with st.spinner("Analyzing image..."):
                prediction_index = predict_disease(uploaded_image)

                labels = list(disease_data.keys())  # Correct class label list
                predicted_label = labels[prediction_index]

                st.success(f" Prediction: **{predicted_label.replace('___', ' - ')}**")

                # Show Cause and Cure
                info = disease_data.get(predicted_label, {})
                if info:
                    st.markdown("### Cause")
                    st.info(info.get("cause", "Cause not available."))

                    st.markdown("###  Cure")
                    st.success(info.get("cure", "Cure not available."))
                else:
                    st.warning("No additional information found for this disease.")
        else:
            st.warning("Please upload an image before clicking detect.")


