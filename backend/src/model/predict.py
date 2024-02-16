import torch
from transformers import ViTForImageClassification, ViTFeatureExtractor

# Define the model name or identifier
model_name = "farleyknight-org-username/vit-base-mnist"

# Load the ViT model and feature extractor from Hugging Face
model = ViTForImageClassification.from_pretrained(model_name)
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

def predict_digit(img):
    """
    Predict a digit image.

    Args:
        img (PIL.Image.Image): The input image to be classified.

    Returns:
        A numpy array of probabilities for each digit class (0-9).
    """
    # Preprocess the input image using the feature extractor
    inputs = feature_extractor(images=img, return_tensors="pt")

    # Pass the preprocessed inputs through the model
    outputs = model(**inputs)

    # Get the predicted probabilities for each class
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]

    # Convert the probabilities to a numpy array
    probabilities = probabilities.detach().numpy()

    return probabilities