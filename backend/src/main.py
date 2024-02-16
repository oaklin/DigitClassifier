import numpy as np
from fastapi import FastAPI, HTTPException, Request
from model.predict import predict_digit

# Create FastAPI instance
app = FastAPI()

# Define GET route to predict the digit
@app.get("/")
async def get_prediction(request: Request):
    try:
        # Parse input image from the request JSON
        request_body = await request.json()
        image_data = np.array(request_body["image"])
        
        # Predict the digit using the imported function
        probabilities = predict_digit(image_data)
        
        # Return probability of each class in a JSON format
        return {"prob": probabilities.tolist()}
    
    except KeyError:
        # If "image" key is not found in the request JSON
        raise HTTPException(status_code=400, detail="Image data is missing in the request.")
    
    except Exception as e:
        # Handle any other unexpected errors
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")