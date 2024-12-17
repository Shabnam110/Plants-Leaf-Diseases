import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# TensorFlow Model Prediction
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("trained_plant_disease_model.keras")

model = load_model()

def model_prediction(test_image):
    image = Image.open(test_image)
    image = image.resize((128, 128))
    input_arr = np.array(image)
    input_arr = np.expand_dims(input_arr, axis=0)  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition", "Precautions"])

# App Modes
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"  # Replace with your home image path
    st.image(image_path, use_container_width=True)  # Updated here
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. Upload Image: Go to the Disease Recognition page and upload an image of a plant with suspected diseases.
    2. Analysis: Our system will process the image using advanced algorithms to identify potential diseases.
    3. Results: View the results and recommendations for further action.

    ### Why Choose Us?
    - Accuracy: Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - User-Friendly: Simple and intuitive interface for seamless user experience.
    - Fast and Efficient: Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the Disease Recognition page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the About page.
    """)

elif app_mode == "About":
    st.header("About")
    st.markdown("""
    #### About Dataset
    This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this GitHub repo.
    This dataset consists of about 87K RGB images of healthy and diseased crop leaves, categorized into 38 different classes. The total dataset is divided into an 80/20 ratio for training and validation sets, preserving the directory structure.
    A new directory containing 33 test images is created later for prediction purposes.
    #### Content
    1. train (70295 images)
    2. test (33 images)
    3. validation (17572 images)
    """)

elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if st.button("Show Image"):
        if test_image is not None:
            st.image(test_image, use_container_width=True)  # Updated here
        else:
            st.warning("Please upload an image to show.")

    # Predict button
    if st.button("Predict"):
        if test_image is not None:
            st.snow()
            st.write("Our Prediction")
            result_index = model_prediction(test_image)
            # Reading Labels
            class_name = ['Apple_Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple_healthy',
                          'Blueberry_healthy', 'Cherry(including_sour)__Powdery_mildew',
                          'Cherry_(including_sour)_healthy', 'Corn(maize)__Cercospora_leaf_spot Gray_leaf_spot',
                          'Corn_(maize)_Common_rust', 'Corn(maize)_Northern_Leaf_Blight', 'Corn(maize__healthy',
                          'Grape_Black_rot', 'Grape_Esca(Black_Measles)', 'Grape_Leaf_blight(Isariopsis_Leaf_Spot)',
                          'Grape_healthy', 'Orange_Haunglongbing(Citrus_greening)', 'Peach__Bacterial_spot',
                          'Peach_healthy', 'Pepper,_bell_Bacterial_spot', 'Pepper,_bell_healthy',
                          'Potato_Early_blight', 'Potato_Late_blight', 'Potato_healthy',
                          'Raspberry_healthy', 'Soybean_healthy', 'Squash_Powdery_mildew', 'Strawberry_Leaf_scorch', 
                          'Strawberry_healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 
                          'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 
                          'Tomato_Spider_mites Two-spotted_spider_mite', 'Tomato_Target_Spot', 
                          'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_Tomato_mosaic_virus', 
                          'Tomato___healthy']
            st.success("Model is predicting it's a {}".format(class_name[result_index]))
        else:
            st.warning("Please upload an image before predicting.")

# Precautions Section
elif app_mode == "Precautions":
    st.header("Precautions for Crops")
    st.subheader("Select a Crop for Specific Precautions")   
    crop = st.selectbox("Choose a Crop", ["Select", "Potato", "Tomato", "Apple", "Grapes", "Corn", "Raspberry"])

    if crop == "Potato":
        st.markdown("### Potato Precautions:\n- Rotate crops annually.\n- Inspect for blight regularly.\n- Use disease-resistant varieties.")
    elif crop == "Tomato":
        st.markdown("### Tomato Precautions:\n- Use resistant varieties.\n- Ensure proper spacing for air circulation.\n- Regularly check for pests.")
    elif crop == "Apple":
        st.markdown("### Apple Precautions:\n- Prune trees annually.\n- Use pest control measures.\n- Maintain proper spacing for sunlight.")
    elif crop == "Grapes":
        st.markdown("### Grape Precautions:\n- Remove infected leaves promptly.\n- Use drip irrigation to avoid wet leaves.\n- Spray appropriate fungicides.")
    elif crop == "Corn":
        st.markdown("### Corn Precautions:\n- Practice crop rotation.\n- Monitor for pests and diseases regularly.\n- Use resistant varieties.")
    elif crop == "Raspberry":
        st.markdown("### Raspberry Precautions:\n- Ensure good air circulation.\n- Remove old canes after harvest.\n- Monitor for pests like spider mites.")
    elif crop == "Select":
        st.warning("Please select a crop to view precautions.")