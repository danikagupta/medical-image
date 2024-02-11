import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

PREDICTED_LABELS = ['0_normal', '1_ulcerative colitis', '2_polyps', '3_esophagitis']

def main():
    model = tf.keras.models.load_model('model.h5')
    st.title("Image Processing App")
    # File uploader
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        # Display the image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Process the image
        # Note: The processing depends on your model requirements
        processed_image = np.array(image.resize((224, 224))) / 255.0  # Example resize and scale
        processed_image = np.expand_dims(processed_image, axis=0)  # Add batch dimension

        # Make prediction or process image
        output = model.predict(processed_image)

        # Display the output
        #st.write("Processed Output:")
        #st.write(output)  # You might want to format this output
        highest_prob_index = np.argmax(output[0])
        forecast=PREDICTED_LABELS[highest_prob_index]
        print(f"File {uploaded_file.name} predicts {forecast} for output {output}")
        st.markdown(f"# Diagnosis: {forecast} \n# Confidence: {round(100*output[0][highest_prob_index])}%")




def predict(sess, image):
    """ Function to predict the class of an image using ONNX model """
    image = np.array(image)
    # Resize the image to match the input shape of the model
    image = np.resize(image, (224, 224, 3))
    image = image.astype('float32')
    image /= 255.0

    # No need to transpose the image since the model expects channels last
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Get the input name for the ONNX model
    input_name = sess.get_inputs()[0].name

    # Predict
    preds = sess.run(None, {input_name: image})[0]
    highest_prob_index = np.argmax(preds[0])
    print(f"Array returned as {preds}")
    return highest_prob_index
    
if __name__ == "__main__":
    main()