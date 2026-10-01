import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    ConfusionMatrixDisplay,
)
from dataset import load_datasets

_, _, test_ds, class_names = load_datasets()
model = tf.keras.models.load_model("models/tongue_model.keras")

images, y_true, y_prob = [], [], []
for batch_images, batch_labels in test_ds:
    probs = model.predict(batch_images, verbose=0).ravel()
    images.append(batch_images.numpy().astype("uint8"))
    y_true.extend(batch_labels.numpy().ravel().astype(int))
    y_prob.extend(probs)

images = np.concatenate(images)
y_true = np.array(y_true)
y_prob = np.array(y_prob)
y_pred = (y_prob >= 0.5).astype(int)

print()
print(classification_report(y_true, y_pred, target_names=class_names, digits=3))
report = classification_report(
    y_true, y_pred, target_names=class_names, output_dict=True
)
auc = float(roc_auc_score(y_true, y_prob))
print("ROC AUC:", round(auc, 3))

cm = confusion_matrix(y_true, y_pred)
print("Confusion matrix (rows = true, columns = predicted):")
print(cm)

ConfusionMatrixDisplay(cm, display_labels=class_names).plot(cmap="Blues")
plt.title("Test set confusion matrix")
plt.savefig("notebooks/confusion_matrix.png", dpi=100)
plt.close()

with open("models/test_metrics.json", "w") as f:
    json.dump(
        {
            "accuracy": float(report["accuracy"]),
            "roc_auc": auc,
            "report": report,
            "confusion_matrix": cm.tolist(),
        },
        f,
        indent=2,
    )

wrong = np.where(y_pred != y_true)[0][:8]
print("Total mistakes:", int((y_pred != y_true).sum()), "out of", len(y_true))
if len(wrong) > 0:
    fig, axes = plt.subplots(2, 4, figsize=(12, 7))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, idx in zip(axes.ravel(), wrong):
        ax.imshow(images[idx])
        conf = y_prob[idx] if y_pred[idx] == 1 else 1 - y_prob[idx]
        ax.set_title(
            "true: " + class_names[y_true[idx]]
            + "\npred: " + class_names[y_pred[idx]]
            + " (" + str(round(float(conf) * 100)) + "%)"
        )
    plt.tight_layout()
    plt.savefig("notebooks/misclassified.png", dpi=100)
    print("Saved notebooks/misclassified.png")

print("Saved notebooks/confusion_matrix.png and models/test_metrics.json")