# Digit Image Classifier
A Digit Image Classifier built with a pretrained Vision Transformer (ViT) model from HuggingFace, along with FastAPI and Streamlit, to classify images into digits ranging from 0 to 9. The ViT model employed in this system has been fine-tuned based on the google/vit-base-patch16-224-in21k model using the mnist dataset.

## Docker
This was tested on Windows 10 with Docker Desktop 4.27.2 installed.
From a command line within the project directory (see screenshot)
- Execute the command to build the docker:

  ``docker-compose -f "docker-compose.yml" up -d --build``

![alt text](https://github.com/oaklin/DigitClassifier/blob/master/pics/docker%20compose.jpg?raw=true)


## Streamlit Web App
Test the application using the webpage at ``http://127.0.0.1:8501/``
- It was built using Streamlit. The Streamlit webapp will send a HTTP request with the image to a FastAPI server. 

![alt text](https://github.com/oaklin/DigitClassifier/blob/master/pics/streamlit.jpg?raw=true)



