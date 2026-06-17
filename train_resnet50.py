import pathlib
import tensorflow as tf # type: ignore
from tensorflow import keras # type: ignore
from tensorflow.keras import layers # type: ignore
from tensorflow.keras.utils import image_dataset_from_directory # type: ignore
import matplotlib.pyplot as plt # type: ignore

dir = pathlib.Path("tomato-leaf-dataset")

train_dataset = image_dataset_from_directory(
    dir / "train",
    image_size=(256, 256),
    batch_size=32)

validation_dataset = image_dataset_from_directory(
    dir / "val",
    image_size=(256, 256),
    batch_size=32)

test_dataset = image_dataset_from_directory(
    dir / "test",
    image_size=(256, 256),
    batch_size=32)

class_names = train_dataset.class_names
num_classes = len(class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(AUTOTUNE)
validation_dataset = validation_dataset.prefetch(AUTOTUNE)
test_dataset = test_dataset.prefetch(AUTOTUNE)

train_dataset = train_dataset.ignore_errors()
validation_dataset = validation_dataset.ignore_errors()
test_dataset = test_dataset.ignore_errors()

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

base_model = keras.applications.ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(256, 256, 3)
)

base_model.trainable = False

inputs = keras.Input(shape=(256, 256, 3))
x = data_augmentation(inputs)
x = keras.applications.resnet.preprocess_input(x)
x = base_model(x, training = False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes,  activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callback = [
    keras.callbacks.ModelCheckpoint(
        filepath="outputs/Tomato-Leaf-Disease-Detection_resnet50.keras",
        save_best_only=True,
        monitor="val_loss"),

    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )
]

history = model.fit(
    train_dataset,
    epochs=20,
    validation_data=validation_dataset,
    callbacks=callback
    )

test_model = keras.models.load_model("outputs/Tomato-Leaf-Disease-Detection_resnet50.keras")

test_loss, test_acc = test_model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.3f}")

# Extract training history
accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(accuracy) + 1)

plt.plot(epochs, accuracy, "bo", label="Training accuracy")
plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.figure()

plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.show()