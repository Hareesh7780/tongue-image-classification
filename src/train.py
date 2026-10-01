import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from dataset import load_datasets, build_augmentation, get_class_weights, IMG_SIZE

tf.keras.utils.set_random_seed(42)

train_ds, val_ds, test_ds, class_names = load_datasets()
class_weights = get_class_weights(class_names)

with open("models/class_names.json", "w") as f:
    json.dump(class_names, f)

base = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet"
)
base.trainable = False

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = build_augmentation()(inputs)
x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1)(x)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
model = tf.keras.Model(inputs, outputs)


def get_metrics():
    return ["accuracy", tf.keras.metrics.AUC(name="auc")]


def get_callbacks(path):
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(path, monitor="val_loss", save_best_only=True),
    ]


print("=== Stage 1: training the new top layer ===")
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=get_metrics(),
)
h1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    class_weight=class_weights,
    callbacks=get_callbacks("models/stage1.keras"),
)

print("=== Stage 2: fine-tuning the last layers ===")
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False
for layer in base.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="binary_crossentropy",
    metrics=get_metrics(),
)
h2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    class_weight=class_weights,
    callbacks=get_callbacks("models/stage2.keras"),
)

model.save("models/tongue_model.keras")
print("Saved models/tongue_model.keras")

acc = h1.history["accuracy"] + h2.history["accuracy"]
val_acc = h1.history["val_accuracy"] + h2.history["val_accuracy"]
loss = h1.history["loss"] + h2.history["loss"]
val_loss = h1.history["val_loss"] + h2.history["val_loss"]
stage_change = len(h1.history["loss"]) - 0.5

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(acc, label="train")
axes[0].plot(val_acc, label="validation")
axes[0].axvline(stage_change, color="gray", linestyle="--")
axes[0].set_title("Accuracy")
axes[0].legend()
axes[1].plot(loss, label="train")
axes[1].plot(val_loss, label="validation")
axes[1].axvline(stage_change, color="gray", linestyle="--")
axes[1].set_title("Loss")
axes[1].legend()
plt.tight_layout()
plt.savefig("notebooks/training_curves.png", dpi=100)
print("Saved notebooks/training_curves.png")
print("Best validation accuracy:", round(max(val_acc), 4))