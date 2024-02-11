import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

PREDICTED_LABELS = ['0_normal', '1_ulcerative colitis', '2_polyps', '3_esophagitis']

model = tf.keras.models.load_model('model.h5')
st.title("Detecting Colon Cancer in its Early Stages")
uploaded_file = st.sidebar.file_uploader("Upload colonoscopy image...", type="jpg")

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    processed_image = np.array(image.resize((224, 224))) / 255.0  # Example resize and scale
    processed_image = np.expand_dims(processed_image, axis=0)  # Add batch dimension
    output = model.predict(processed_image)

    # Display the output
    highest_prob_index = np.argmax(output[0])
    forecast=PREDICTED_LABELS[highest_prob_index]
    st.markdown(f"# Diagnosis: {forecast} \n# Confidence: {round(100*output[0][highest_prob_index])}%")
