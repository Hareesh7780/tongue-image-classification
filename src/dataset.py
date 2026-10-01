from pathlib import Path
import tensorflow as tf

DATA_DIR = Path("data/split")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42


def load_datasets():
    common = dict(
        labels="inferred",
        label_mode="binary",
        color_mode="rgb",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )
    train = tf.keras.utils.image_dataset_from_directory(
        str(DATA_DIR / "train"), shuffle=True, seed=SEED, **common
    )
    val = tf.keras.utils.image_dataset_from_directory(
        str(DATA_DIR / "val"), shuffle=False, **common
    )
    test = tf.keras.utils.image_dataset_from_directory(
        str(DATA_DIR / "test"), shuffle=False, **common
    )
    class_names = train.class_names
    autotune = tf.data.AUTOTUNE
    return (
        train.prefetch(autotune),
        val.prefetch(autotune),
        test.prefetch(autotune),
        class_names,
    )


def build_augmentation():
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )


def get_class_weights(class_names):
    counts = [len(list((DATA_DIR / "train" / c).glob("*.png"))) for c in class_names]
    total = sum(counts)
    return {i: total / (len(counts) * n) for i, n in enumerate(counts)}