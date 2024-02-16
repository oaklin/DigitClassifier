import os
import requests
import cv2
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas

CANVAS_SIZE = 256
BACKEND_URL = os.environ.get("URL_BACKEND")

def predict_digit(img):
    """
    Sends a request to the backend with the image data
    to perform a classification on the image.
    """
    if not BACKEND_URL:
        st.error("Backend URL not provided.")
        return None
    
    try:
        # Send a GET request to the backend with the image data as a JSON payload
        request = requests.get(BACKEND_URL, json={"image": img.tolist()})
        # Retrieve the predicted probabilities from the response
        answer = request.json()
        prob = answer["prob"]
        return np.array(prob)
    except Exception as e:
        st.error(f"Error occurred: {e}")
        return None

def main():
    """
    Main Streamlit function
    Read an image and show a probability
    """

    st.set_page_config(page_title="Handwritten Digit Classifier")
    st.title("Instructions")
  
    st.markdown(
        """
        ## Digit Image Classifier

        This is a simple interface for a Handwritten Digit Classifier that utilizes a pretrained Vision Transformer (ViT) model from HuggingFace, along with FastAPI and Streamlit, to classify images into digits ranging from 0 to 9. The ViT model employed in this system has been fine-tuned based on the google/vit-base-patch16-224-in21k model using the mnist dataset.

        """
    )

    prob = None
    canvas_image = None

    # Draw a number
    st.markdown("### Draw a Digit:")
    canvas_image = st_canvas(
        fill_color="black",
        stroke_width=20,
        stroke_color="gray",
        width=CANVAS_SIZE,
        height=CANVAS_SIZE,
        drawing_mode="freedraw",
        key="canvas",
        display_toolbar=True,
    )
    
    if canvas_image is not None and canvas_image.image_data is not None:
        if st.button("Classify"):
            img = cv2.cvtColor(canvas_image.image_data, cv2.COLOR_RGBA2RGB)
            with st.spinner("Wait for it..."):
                prob = predict_digit(img)
                
    st.markdown("---")
    
    # Prediction
    st.markdown("### Prediction:")
    if prob is not None:
        predicted_digit = prob.argmax()
        st.metric(label="Predicted digit:", value=str(predicted_digit))

if __name__ == "__main__":
    main()