from flask import Flask, request, render_template_string
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalFocalCrossentropy
from PIL import Image
import numpy as np
import io
import base64

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


def hex2rgb(hex_color):
    """
    Convert a hexadecimal color to an RGB tuple.
    """
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def preprocess_image_for_unet(image):
    """
    Preprocess the image for the U-Net model:
    - Convert to RGB
    - Normalize for U-Net (0-1 range)
    - Crop the image into a maximum of 4 patches of 256x256 each (2x2 grid).
    """
    # Convert to RGB in case it's a different format
    image = image.convert('RGB')

    # Convert to numpy array and normalize for U-Net (0-1 range)
    image_array = np.array(image) / 255.0

    # Limit the crop area to 512x512 if the image is larger than that
    max_crop_size = 512
    crop_size = 256
    height, width, _ = image_array.shape

    # Determine the crop dimensions
    crop_height = min(height, max_crop_size)
    crop_width = min(width, max_crop_size)

    # Crop the image to the determined dimensions
    cropped_images = []
    for y in range(0, crop_height, crop_size):
        for x in range(0, crop_width, crop_size):
            if y + crop_size <= crop_height and x + crop_size <= crop_width:
                cropped_image = image_array[y:y+crop_size, x:x+crop_size]
                cropped_images.append(cropped_image)

    return cropped_images


def preprocess_image_for_classifiers(image):
    """
    Preprocess the image for the Logistic Regression and Random Forest models:
    - Convert to RGB
    - Crop the image into a maximum of 4 patches of 256x256 each (2x2 grid).
    - Extract RGB channels and flatten for model input
    """
    # Convert to RGB in case it's a different format
    image = image.convert('RGB')

    # Convert to numpy array
    image_array = np.array(image)

    # Limit the crop area to 512x512 if the image is larger than that
    max_crop_size = 512
    crop_size = 256
    height, width, _ = image_array.shape

    # Determine the crop dimensions
    crop_height = min(height, max_crop_size)
    crop_width = min(width, max_crop_size)

    # Crop the image to the determined dimensions
    cropped_images = []
    for y in range(0, crop_height, crop_size):
        for x in range(0, crop_width, crop_size):
            if y + crop_size <= crop_height and x + crop_size <= crop_width:
                cropped_image = image_array[y:y+crop_size, x:x+crop_size]
                # Extract the red, green, and blue channels and flatten
                red = cropped_image[:, :, 0].flatten()
                green = cropped_image[:, :, 1].flatten()
                blue = cropped_image[:, :, 2].flatten()
                features = np.array([red, green, blue]).T
                cropped_images.append(features)

    return cropped_images


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

        # Open the uploaded image
        image = Image.open(file)

        if model_choice == 'unet':
            # Preprocess the image for U-Net
            cropped_images = preprocess_image_for_unet(image)

            predicted_masks = []
            for cropped_image in cropped_images:
                # U-Net requires input to be in 4D (batch_size, height, width, channels)
                unet_result = unet_model.predict(
                    np.expand_dims(cropped_image, axis=0))
                # Convert softmax probabilities to class labels
                predicted_mask = np.argmax(unet_result, axis=-1)[0]
                # Convert class labels back to RGB
                predicted_mask_rgb = class_to_rgb_conversion(
                    predicted_mask, class_to_rgb)
                predicted_masks.append(predicted_mask_rgb)

            # Stitch the patches back together
            # Prepare a 512x512 canvas
            result_image = np.zeros((512, 512, 3), dtype=np.uint8)
            crop_size = 256
            idx = 0
            for y in range(0, 512, crop_size):
                for x in range(0, 512, crop_size):
                    if idx < len(predicted_masks):
                        result_image[y:y+crop_size, x:x +
                                     crop_size] = predicted_masks[idx]
                        idx += 1
            # Crop the final result to match original size
            result_image = result_image[:image.size[1], :image.size[0], :]

        elif model_choice in ['logistic', 'random_forest']:
            # Preprocess the image for the classifiers
            cropped_features = preprocess_image_for_classifiers(image)

            predicted_masks = []
            model = logistic_model if model_choice == 'logistic' else random_forest_model

            for features in cropped_features:
                # Predict using the classifier model
                predicted_mask = model.predict(features)
                # Reshape the flat array back to the image size (256x256)
                predicted_mask_rgb = np.array([hex2rgb(
                    hex_color) for hex_color in predicted_mask]).reshape(256, 256, 3).astype(np.uint8)
                predicted_masks.append(predicted_mask_rgb)

            # Stitch the patches back together
            # Prepare a 512x512 canvas
            result_image = np.zeros((512, 512, 3), dtype=np.uint8)
            crop_size = 256
            idx = 0
            for y in range(0, 512, crop_size):
                for x in range(0, 512, crop_size):
                    if idx < len(predicted_masks):
                        result_image[y:y+crop_size, x:x +
                                     crop_size] = predicted_masks[idx]
                        idx += 1
            # Crop the final result to match original size
            result_image = result_image[:image.size[1], :image.size[0], :]

        # Convert the predicted mask back to an image
        masked_image = Image.fromarray(result_image)

        # Convert the original image to display
        original_image_buffer = io.BytesIO()
        image.save(original_image_buffer, format='PNG')
        original_image_buffer.seek(0)
        encoded_original_image = base64.b64encode(
            original_image_buffer.getvalue()).decode()

        # Convert the masked image to display
        masked_image_buffer = io.BytesIO()
        masked_image.save(masked_image_buffer, format='PNG')
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
