import json
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

IMG_SIZE = (224, 224)
model = tf.keras.models.load_model("models/tongue_model.keras")
with open("models/class_names.json") as f:
    class_names = json.load(f)

base = [l for l in model.layers
        if isinstance(l, tf.keras.Model) and l.name.startswith("mobilenet")][0]
rescale = [l for l in model.layers if isinstance(l, tf.keras.layers.Rescaling)][0]
gap = [l for l in model.layers if isinstance(l, tf.keras.layers.GlobalAveragePooling2D)][0]
dense = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)][0]


def preprocess(pil_img):
    gray = pil_img.convert("L").resize(IMG_SIZE, Image.BILINEAR)
    return np.stack([np.array(gray)] * 3, axis=-1).astype("float32")


def gradcam(rgb):
    batch = tf.convert_to_tensor(rgb[None, ...])
    p = float(model(batch, training=False).numpy()[0][0])
    idx = int(p >= 0.5)
    with tf.GradientTape() as tape:
        fmap = base(rescale(batch), training=False)
        tape.watch(fmap)
        p_t = dense(gap(fmap))[:, 0]
        target = p_t if idx == 1 else 1.0 - p_t
    grads = tape.gradient(target, fmap)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.reduce_sum(fmap * weights[:, None, None, :], axis=-1)[0]
    cam = tf.nn.relu(cam).numpy()
    cam = cam / (cam.max() + 1e-8)
    cam = cv2.resize(cam, IMG_SIZE)
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(rgb.astype("uint8"), 0.6, heat, 0.4, 0)
    return p, idx, overlay


def run(cls_name):
    files = sorted(Path("data/split/test/" + cls_name).glob("*.png"))[:6]
    fig, axes = plt.subplots(2, len(files), figsize=(3 * len(files), 6))
    for j, f in enumerate(files):
        rgb = preprocess(Image.open(f))
        p, idx, overlay = gradcam(rgb)
        conf = p if idx == 1 else 1 - p
        axes[0, j].imshow(rgb.astype("uint8"))
        axes[0, j].set_title("true: " + cls_name)
        axes[1, j].imshow(overlay)
        axes[1, j].set_title("pred: " + class_names[idx] + " " + str(round(conf * 100)) + "%")
        axes[0, j].axis("off")
        axes[1, j].axis("off")
    plt.tight_layout()
    plt.savefig("notebooks/gradcam_" + cls_name + ".png", dpi=100)
    plt.close()
    print("Saved notebooks/gradcam_" + cls_name + ".png")


run("marked")
run("unmarked")