import pathlib
from tensorflow import keras # type: ignore
from tensorflow.keras import layers # type: ignore
from tensorflow.keras.utils import image_dataset_from_directory # type: ignore
import matplotlib.pyplot as plt

dir = pathlib.Path("tomato-leaf-dataset")

inputs = keras.Input(shape=(256, 256, 3))
x = layers.Rescaling(1./255)(inputs)
x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
x = layers.Flatten()(x)
outputs = layers.Dense(10, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)

model.summary()

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

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

# Ignore any corrupted images (prevents crashes)
train_dataset = train_dataset.ignore_errors()
validation_dataset = validation_dataset.ignore_errors()
test_dataset = test_dataset.ignore_errors()

callback = [
    keras.callbacks.ModelCheckpoint(
        filepath="Tomato-Leaf-Disease-Detection.keras",
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
    epochs=50,
    validation_data=validation_dataset,
    callbacks=callback
    )

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

test_model = keras.models.load_model("Tomato-Leaf-Disease-Detection.keras")

test_loss, test_acc = test_model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.3f}")