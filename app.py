import json
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

IMG_SIZE = (224, 224)
UNCERTAIN_BELOW = 0.65

st.set_page_config(page_title="Tongue Image Classifier (Experimental)", layout="centered")


@st.cache_resource
def load_everything():
    model = tf.keras.models.load_model("models/tongue_model.keras")
    with open("models/class_names.json") as f:
        names = json.load(f)
    try:
        with open("models/test_metrics.json") as f:
            metrics = json.load(f)
    except FileNotFoundError:
        metrics = None
    return model, names, metrics


def preprocess(pil_img):
    gray = pil_img.convert("L").resize(IMG_SIZE, Image.BILINEAR)
    rgb = np.stack([np.array(gray)] * 3, axis=-1).astype("float32")
    return rgb


def predict_with_gradcam(model, rgb):
    batch = tf.convert_to_tensor(rgb[None, ...])
    p = float(model(batch, training=False).numpy()[0][0])
    idx = int(p >= 0.5)

    base = [l for l in model.layers
            if isinstance(l, tf.keras.Model) and l.name.startswith("mobilenet")][0]
    rescale = [l for l in model.layers if isinstance(l, tf.keras.layers.Rescaling)][0]
    gap = [l for l in model.layers if isinstance(l, tf.keras.layers.GlobalAveragePooling2D)][0]
    dense = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)][0]

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


model, class_names, metrics = load_everything()

st.title("Tongue Image Classifier")
st.warning(
    "EXPERIMENTAL PROJECT. This is a learning and portfolio project, not a validated "
    "medical diagnostic tool. It must not be used to diagnose, treat, or rule out any "
    "condition. Please consult a qualified healthcare professional for medical concerns."
)

with st.sidebar:
    st.header("About this model")
    st.write(
        "Binary classification of tooth-marked ('marked') vs 'unmarked' tongue images. "
        "MobileNetV2 transfer learning, trained on 1,250 grayscale tongue images."
    )
    if metrics:
        st.write("Test accuracy: " + str(round(metrics["accuracy"] * 100, 1)) + "%")
        st.write("Test ROC AUC: " + str(round(metrics["roc_auc"], 3)))
    st.caption(
        "The test set has only 190 images, so expect roughly plus or minus 5 points "
        "of uncertainty. Works best on cropped, grayscale tongue images like the "
        "training data. Ordinary phone photos are a very different input and "
        "predictions on them are unreliable."
    )

uploaded = st.file_uploader("Upload a tongue image (PNG or JPG)", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    img = Image.open(uploaded)
    rgb = preprocess(img)
    p, idx, overlay = predict_with_gradcam(model, rgb)
    confidence = p if idx == 1 else 1 - p

    col1, col2 = st.columns(2)
    col1.image(rgb.astype("uint8"), caption="Input (resized, grayscale)")
    col2.image(overlay, caption="Grad-CAM: regions that influenced the prediction")

    st.subheader("Predicted class: " + class_names[idx])
    st.metric("Confidence", str(round(confidence * 100, 1)) + "%")
    if confidence < UNCERTAIN_BELOW:
        st.info("Low confidence: treat this result as uncertain.")

    probs = pd.DataFrame({"probability": [1 - p, p]}, index=class_names)
    st.bar_chart(probs)
    st.caption("Confidence is the model's own score, not a true probability of being correct.")