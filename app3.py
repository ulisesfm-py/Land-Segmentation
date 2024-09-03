from flask import Flask, request, render_template_string
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalFocalCrossentropy
from PIL import Image
import numpy as np
import io
import base64
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load your models
logistic_model = joblib.load('models/lr_balanced_model.joblib')
random_forest_model = joblib.load('models/rf_balanced_model.joblib')
unet_model = load_model(
    'models/land_segmentation_unet_50epochs_ClassWeights_Bsize32_adam_categoricalfocal_FULL_resolution.hdf5', compile=False)

# Recompile the U-Net model with the original loss function
unet_model.compile(
    optimizer=Adam(), loss=CategoricalFocalCrossentropy(), metrics=['accuracy']
)

# Define the class-to-RGB mapping (reverse of your RGB-to-class mapping)
rgb_to_class = {
    (0, 0, 0): 5,         # Black pixels as background or ignored class
    (41, 169, 226): 0,    # Water
    (58, 221, 254): 1,    # Land
    (152, 16, 60): 2,     # Road
    (228, 193, 110): 3,   # Building
    (246, 41, 132): 4,    # Vegetation
    (155, 155, 155): 5    # Unlabeled
}
class_to_rgb = {v: k for k, v in rgb_to_class.items()}


def crop_image(image, size=256):
    """
    Crop the image to the specified size (default 256x256).
    """
    width, height = image.size
    left = (width - size) / 2
    top = (height - size) / 2
    right = (width + size) / 2
    bottom = (height + size) / 2
    return image.crop((left, top, right, bottom))


def preprocess_image_for_unet(image):
    """
    Preprocess the image for the U-Net model:
    - Normalize for U-Net (0-1 range)
    """
    # Convert to RGB and normalize
    image_array = np.array(image.convert('RGB')) / 255.0
    return np.expand_dims(image_array, axis=0)


def preprocess_image_for_classifiers(image):
    """
    Preprocess the image for the Logistic Regression and Random Forest models:
    - Extract RGB channels and flatten for model input.
    """
    # Convert to RGB
    image_array = np.array(image.convert('RGB'))
    # Extract the red, green, and blue channels and flatten
    red = image_array[:, :, 0].flatten()
    green = image_array[:, :, 1].flatten()
    blue = image_array[:, :, 2].flatten()
    features = np.array([red, green, blue]).T
    return features


def class_to_rgb_conversion(mask, class_to_rgb):
    """
    Convert class labels back to RGB values.
    """
    # Initialize an empty array for the RGB image
    rgb_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    # Iterate through the class-to-RGB mapping and replace class labels with RGB values
    for class_label, rgb in class_to_rgb.items():
        rgb_image[mask == class_label] = rgb

    return rgb_image


@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        model_choice = request.form['model']
        file = request.files['file']
        if not file:
            return 'No file uploaded', 400

        # Open the uploaded image and crop to 256x256
        image = crop_image(Image.open(file), size=256)

        if model_choice == 'unet':
            # Preprocess the image for U-Net
            processed_image = preprocess_image_for_unet(image)
            # Predict using the U-Net model
            unet_result = unet_model.predict(processed_image)
            # Convert softmax probabilities to class labels
            predicted_mask = np.argmax(unet_result, axis=-1)[0]
            # Convert class labels back to RGB
            result_image = class_to_rgb_conversion(
                predicted_mask, class_to_rgb)

        elif model_choice in ['logistic', 'random_forest']:
            # Preprocess the image for the classifiers
            features = preprocess_image_for_classifiers(image)
            model = logistic_model if model_choice == 'logistic' else random_forest_model
            # Predict using the classifier model
            predicted_mask = model.predict(features)
            # Reshape the flat array back to the image size
            predicted_mask_rgb = np.array([class_to_rgb[int(
                hex_color)] for hex_color in predicted_mask]).reshape(256, 256, 3).astype(np.uint8)
            result_image = predicted_mask_rgb

        # Convert the original and masked images to display
        original_image_buffer = io.BytesIO()
        image.save(original_image_buffer, format='PNG')
        original_image_buffer.seek(0)
        encoded_original_image = base64.b64encode(
            original_image_buffer.getvalue()).decode()

        masked_image_buffer = io.BytesIO()
        Image.fromarray(result_image).save(masked_image_buffer, format='PNG')
        masked_image_buffer.seek(0)
        encoded_masked_image = base64.b64encode(
            masked_image_buffer.getvalue()).decode()

        return render_template_string('''
        <!doctype html>
        <title>Result</title>
        <h1>Original and Masked Images</h1>
        <div>
            <h2>Original Image</h2>
            <img src="data:image/png;base64,{{original_image}}" alt="Original Image">
        </div>
        <div>
            <h2>Masked Image ({{model}})</h2>
            <img src="data:image/png;base64,{{masked_image}}" alt="Masked Image">
        </div>
        ''', original_image=encoded_original_image, masked_image=encoded_masked_image, model=model_choice.capitalize())

    return '''
    <!doctype html>
    <title>Upload an Image</title>
    <h1>Upload an Image for Land Segmentation</h1>
    <form method="post" enctype="multipart/form-data">
      <input type="file" name="file" required>
      <select name="model" required>
        <option value="unet">U-Net</option>
        <option value="logistic">Logistic Regression</option>
        <option value="random_forest">Random Forest</option>
      </select>
      <input type="submit" value="Upload">
    </form>
    '''


if __name__ == '__main__':
    app.run(debug=True)
