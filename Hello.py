import streamlit as st
import numpy as np
import onnxruntime as ort
from PIL import Image

def load_model(model_path):
    """ Load the ONNX model """
    sess = ort.InferenceSession(model_path)
    return sess

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
    
def main():
    st.title("Image Classification with ONNX Model")

    # Load ONNX model
    model_path = 'model.onnx'  # Update this path
    sess = load_model(model_path)

    # Upload image
    uploaded_image = st.sidebar.file_uploader("Upload image...", type=["jpg", "jpeg", "png"])
    descriptions = ["Description for Disease 1", "Description for Disease 2", "Description for Disease 3", "Description for Disease 4"]
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', width=200)
        prediction=st.empty()
        prediction.write("Classifying...")

        # Make prediction
        preds = predict(sess, image)
        
        # Show predictions
        # Note: Update this part based on how you want to display the predictions
        print(f"Got prediction as {preds}")
        prediction.write(f"# AI analysis: \n## {descriptions[preds]}")

if __name__ == "__main__":
    main()