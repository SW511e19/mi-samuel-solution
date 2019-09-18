from pathlib import Path
import keras
import numpy as np
import joblib
from keras.preprocessing import image
from keras.applications import vgg16

# Path to folders with training data
not_cards = Path("training_data") / "not_cards"
green_cards = Path("training_data") / "green_cards"
red_cards = Path("training_data") / "red_cards"
blue_cards = Path("training_data") / "blue_cards"
black_cards = Path("training_data") / "black_cards"

def add_image_and_label(image_path, label_number):
    for img in image_path.glob("*.png"):
        # Load the image from disk
        img = image.load_img(img)

        # Convert the image to a numpy array
        image_array = image.img_to_array(img)

        # Add the image to the list of images
        images.append(image_array)

        # For each 'not card' image, the expected value should be 0
        labels.append(formatted_card_labels[label_number])

images = []
card_labels = [
    0,
    1,
    2,
    3,
    4
]

formatted_card_labels = keras.utils.to_categorical(card_labels, 5)
print(formatted_card_labels)


labels = []
# Load all the not-card images
add_image_and_label(not_cards, 0)
add_image_and_label(green_cards, 1)
add_image_and_label(red_cards, 2)
add_image_and_label(blue_cards, 3)
add_image_and_label(black_cards, 4)

# Create a single numpy array with all the images we loaded
x_train = np.array(images)

# Also convert the labels to a numpy array
y_train = np.array(labels)

# Normalize image data to 0-to-1 range
x_train = vgg16.preprocess_input(x_train)

# Load a pre-trained neural network to use as a feature extractor
pretrained_nn = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Extract features for each image (all in one pass)
features_x = pretrained_nn.predict(x_train)

# Save the array of extracted features to a file
joblib.dump(features_x, "x_train.dat")

# Save the matching array of expected values to a file
joblib.dump(y_train, "y_train.dat")
