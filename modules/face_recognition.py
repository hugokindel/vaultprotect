import os
import shutil

from flask import render_template, request
from keras.src.preprocessing.image import ImageDataGenerator
from werkzeug.utils import secure_filename
from app import app, model, img_width, img_height, batch_size, train_generator
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import numpy as np

# Location of the folder that will contain temporary data during analysis.
UPLOAD_FOLDER = "temporary"

# Label of the person to recognize.
# In a real use case, it should be the ID of the user fetch from the database.
TEST_RECOGNITION = "4044"


def authenticate_face(file):
    """
    Recognize face from image.

    :param file: File to recognize.

    :return: True if recognized, False otherwise.
    """
    filename_with_ext = secure_filename(file.filename)
    filename = os.path.splitext(filename_with_ext)[0]
    os.mkdir(os.path.join(UPLOAD_FOLDER, filename))
    filepath = os.path.join(UPLOAD_FOLDER, filename, filename_with_ext)
    file.save(filepath)

    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    test_generator = test_datagen.flow_from_directory(
        UPLOAD_FOLDER,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode="categorical")

    res = model.predict(test_generator)

    predicted_class_indices = np.argmax(res, axis=1)
    labels = dict((v, k) for k, v in train_generator.class_indices.items())
    predictions = [labels[k] for k in predicted_class_indices]

    shutil.rmtree(os.path.join(UPLOAD_FOLDER, filename))

    print("Found: " + predictions[0] + "; Expected: " + TEST_RECOGNITION + ".")

    return predictions[0] == TEST_RECOGNITION


@app.route("/api/v1/authenticate_face", methods=["POST"])
def backend_authenticate_face():
    """
    Authenticate face from image.

    It expects a file in the request.

    :return: JSON response.
    """
    if "file" not in request.files:
        return {
            "status": "error",
            "message": "No file part",
            "recognized": False
        }

    file = request.files["file"]

    if not file or file.filename == "":
        return {
            "status": "error",
            "message": "No file selected",
            "recognized": False
        }

    return {
        "status": "success",
        "message": "File successfully uploaded",
        "recognized": authenticate_face(file)
    }


@app.route("/authenticate_face")
def frontend_authenticate_face():
    """
    Render face recognition page.

    :return: Rendered page.
    """
    return render_template("authenticate_face.html")
