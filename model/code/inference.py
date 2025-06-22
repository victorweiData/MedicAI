import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import logging

# Set up logging
logger = logging.getLogger(__name__)

NUM_CLASSES = 2  # pneumonia or normal

def model_fn(model_dir):
    """Load model for inference"""
    try:
        logger.info(f"Loading model from {model_dir}")
        
        # Create model architecture
        model = models.resnet18(weights=None)  # Use weights=None instead of pretrained=False
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        
        # Load checkpoint
        model_path = f"{model_dir}/model.pth"
        logger.info(f"Loading checkpoint from {model_path}")
        
        checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
        
        # Handle DataParallel wrapper
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
            
        # Remove 'module.' prefix if model was trained with DataParallel
        if any(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
        model.eval()
        
        logger.info("Model loaded successfully")
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def input_fn(request_body, content_type="application/x-image"):
    """Transform input data for inference"""
    try:
        logger.info(f"Processing input with content_type: {content_type}")
        
        if content_type in ["application/x-image", "image/jpeg", "image/png"]:
            # Open and convert image
            img = Image.open(io.BytesIO(request_body)).convert("RGB")
            
            # Apply same transforms as validation
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225]),
            ])
            
            tensor = transform(img).unsqueeze(0)  # Add batch dimension
            logger.info(f"Input tensor shape: {tensor.shape}")
            return tensor
            
        else:
            raise ValueError(f"Unsupported content type: {content_type}")
            
    except Exception as e:
        logger.error(f"Error in input_fn: {str(e)}")
        raise

def predict_fn(input_data, model):
    """Run inference on input data"""
    try:
        logger.info("Running inference")
        
        with torch.no_grad():
            outputs = model(input_data)
            probs = torch.softmax(outputs, dim=1)
            
        # Convert to list and get prediction
        probs_list = probs.squeeze().tolist()  # Remove batch dimension
        predicted_class = torch.argmax(probs, dim=1).item()
        
        # Class names
        class_names = ["NORMAL", "PNEUMONIA"]
        
        result = {
            "probabilities": probs_list,
            "predicted_class": predicted_class,
            "predicted_label": class_names[predicted_class],
            "confidence": max(probs_list)
        }
        
        logger.info(f"Prediction result: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error in predict_fn: {str(e)}")
        raise

def output_fn(prediction, accept="application/json"):
    """Format output"""
    try:
        if accept == "application/json":
            return prediction, accept
        else:
            raise ValueError(f"Unsupported accept type: {accept}")
    except Exception as e:
        logger.error(f"Error in output_fn: {str(e)}")
        raise