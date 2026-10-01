import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataset import load_datasets, build_augmentation, get_class_weights

train_ds, val_ds, test_ds, class_names = load_datasets()
print("Classes:", class_names)
print("Class weights:", get_class_weights(class_names))

aug = build_augmentation()
images, labels = next(iter(train_ds))
print("Batch shape:", images.shape)
print("Pixel range:", float(images.numpy().min()), "to", float(images.numpy().max()))

fig, axes = plt.subplots(2, 4, figsize=(12, 7))
for i, ax in enumerate(axes[0]):
    ax.imshow(images[i].numpy().astype("uint8"))
    ax.set_title(class_names[int(labels[i][0])])
    ax.axis("off")

augmented = aug(images, training=True)
for i, ax in enumerate(axes[1]):
    ax.imshow(augmented[i].numpy().astype("uint8"))
    ax.set_title("augmented")
    ax.axis("off")

plt.tight_layout()
plt.savefig("notebooks/sample_grid.png", dpi=100)
print("Saved notebooks/sample_grid.png")
