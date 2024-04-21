import tensorflow as tf
from PIL import Image
import numpy as np

# Load the trained ResNet50 model
model = tf.saved_model.load('C:\\Users\\vishn\\OneDrive\\Documents\\c c++\\Kavya_projects\\JPPY2309-Digital Image\\SOURCE CODE\\logo_pred\\ResNet50')

# Define the classes
classes = ["Fake", "Genuine"]

image_path = "C:\\Users\\vishn\\OneDrive\\Documents\\c c++\\Kavya_projects\\JPPY2309-Digital Image\\SOURCE CODE\\logo_pred\\train\\Fake\\000001_0c24510ac5084d43866538a629efeab1.jpg"  # Replace with the path to your image
img = Image.open(image_path).convert('RGB')
img = img.resize((300, 300 * img.size[1] // img.size[0]), Image.LANCZOS)
inp_numpy = np.array(img)[None]
inp = tf.constant(inp_numpy, dtype='float32')
class_scores = model(inp)[0].numpy()
predicted_class = classes[class_scores.argmax()]
print("Predicted Class:", predicted_class)