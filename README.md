# Digit Image Classifier
A Digit Image Classifier built with a pretrained Vision Transformer (ViT) model from HuggingFace, along with FastAPI and Streamlit, to classify images into digits ranging from 0 to 9. The ViT model employed in this system has been fine-tuned based on the google/vit-base-patch16-224-in21k model using the mnist dataset.


## Streamlit Web App
![alt text](https://github.com/oaklin/DigitClassifier/blob/master/pics/streamlit.jpg?raw=true)

- Streamlit is framework for building webapps. The Streamlit webapp will send a HTTP request with the image to a FastAPI server. Go to ``http://127.0.0.1:8501/`` to try the Digit Image Classifier web app.

## FastAPI server
- FastAPI is a framework for building API in Python. Go to ``http://127.0.0.1:8000/docs`` to view the API served. FastAPI will call the VIT model for prediction. FastAPI server will then send back the response to the Streamlit webapp

![alt text](https://github.com/TLIJUN99/DigitRecognizer/blob/main/pics/FastAPI.png?raw=true)

## Docker
- Execute the command to build the docker:

  ``docker-compose -f "docker-compose.yml" up -d --build``

![alt text](https://github.com/oaklin/DigitClassifier/blob/master/pics/docker%20compose.jpg?raw=true)

- Execute the command to stop the docker:
  
  ``docker compose stop``

- Use the following command to debug docker:

  ``docker logs -f <container name>`` : to view docker logs if container failed to start up

  ``docker ps -a``: to check if the docker containers are running

  ``docker exec -it <container name> bash`` : to enter into the docker containers after docker container are up

  



