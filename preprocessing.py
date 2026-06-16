import cv2
import numpy as np

IMG_SIZE = 224  # EfficientNetB0 input size

# ImageNet normalization values (same as torchvision)
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

def load_and_preprocess_image(image_path=None, image_bytes=None):
    """
    Load image from file path or bytes, resize, convert to RGB, and normalize.
    Returns preprocessed numpy array of shape (1, 224, 224, 3) with ImageNet normalization.
    """
    if image_path:
        img = cv2.imread(image_path)
    elif image_bytes:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    else:
        raise ValueError("Provide either image_path or image_bytes")

    # Convert BGR (OpenCV default) to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to 224x224
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Convert to float32 and normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    # Apply ImageNet normalization
    img = (img - MEAN) / STD
    
    # Add batch dimension
    img_array = np.expand_dims(img, axis=0)
    
    return img_array