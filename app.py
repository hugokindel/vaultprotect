# VaultProtect
#
# You can define environment variables to change server settings:
# - `VAULTPROTECT_SERVER_IP`: The IP address to bind to.
# - `VAULTPROTECT_SERVER_PORT`: The port to bind to.

import os
import datetime
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
# noinspection PyUnresolvedReferences
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
# noinspection PyUnresolvedReferences
from tensorflow.python.keras.regularizers import l2
from matplotlib import pyplot as plt

# Change these settings to enable/disable features.
load = True
save = True
train = False
test = False
server = True

# Global settings.
train_dir = "datasets/train"
validation_dir = "datasets/validation"
test_dir = "datasets/test"
model_path = "models/vaultprotect.keras"
img_width = 224
img_height = 224
batch_size = 32
epochs = 25
num_people = len([f for f in os.listdir(train_dir) if not f.startswith(".")])

if server:
    from flask import Flask

base_model = VGG16(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3))

for layer in base_model.layers:
    layer.trainable = False

base_model_output = Flatten()(base_model.output)

base_model_output = Dense(1024, activation='relu')(base_model_output)
base_model_output = BatchNormalization()(base_model_output)
base_model_output = Dropout(0.5)(base_model_output)

base_model_output = Dense(1024, activation='relu')(base_model_output)
base_model_output = BatchNormalization()(base_model_output)
base_model_output = Dropout(0.5)(base_model_output)

base_model_output = Dense(num_people, activation='softmax')(base_model_output)

model = Model(inputs=base_model.input, outputs=base_model_output)

if load and os.path.exists(model_path):
    model.load_weights(model_path)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="categorical")

if train:
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    validation_generator = val_datagen.flow_from_directory(
        validation_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode="categorical")

    tl_checkpoint_1 = ModelCheckpoint(filepath=model_path.replace(".keras", ".checkpoint.keras"),
                                      save_best_only=True,
                                      verbose=1)

    early_stop = EarlyStopping(monitor="val_loss",
                               patience=50,
                               restore_best_weights=True,
                               mode="min")

    print(datetime.datetime.now())

    history = model.fit(train_generator,
              steps_per_epoch=train_generator.samples // batch_size,
              epochs=epochs,
              validation_data=validation_generator,
              validation_steps=validation_generator.samples // batch_size,
              callbacks=[tl_checkpoint_1, early_stop])

    print(datetime.datetime.now())

    if save:
        model.save_weights(model_path)

    accuracy_stats = plt.figure()
    print(history.history['accuracy'], history.history['val_accuracy'])
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['training accuracy', 'validation accuracy'], loc='upper left')
    plt.show()
    accuracy_stats.savefig("accuracy_stats.pdf")

    plt.clf()

    loss_stats = plt.figure()
    print(history.history['loss'], history.history['val_loss'])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['training loss', 'validation loss'], loc='upper left')
    plt.show()
    loss_stats.savefig("loss_stats.pdf")

if test:
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode="categorical")

    test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // batch_size)

if server:
    # Create the web application.
    app = Flask(__name__)

    # Run the web application.
    app.run(os.getenv("VAULTPROTECT_SERVER_IP", "0.0.0.0"), port=os.getenv("VAULTPROTECT_SERVER_PORT", 8000))

    # Import all web endpoints.
    # This needs to be at the end to avoid circular imports.
    # noinspection PyUnresolvedReferences
    import modules
