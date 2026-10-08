import time
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from app.inference import CLASS_NAMES


def evaluate_model(model_path: str, test_dir: str) -> None:
    model = tf.keras.models.load_model(model_path)
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        image_size=(224, 224),
        batch_size=32,
        shuffle=False,
    )

    y_true = []
    y_pred = []
    y_top2_hit = []

    for batch_imgs, batch_labels in test_ds:
        probs = model.predict(batch_imgs, verbose=0)
        preds = np.argmax(probs, axis=1)
        top2 = np.argsort(probs, axis=1)[:, -2:]
        labels_np = batch_labels.numpy()

        y_true.extend(labels_np.tolist())
        y_pred.extend(preds.tolist())
        for lbl, t2 in zip(labels_np, top2):
            y_top2_hit.append(int(lbl in t2))

    acc = np.mean(np.array(y_true) == np.array(y_pred))
    top2_acc = np.mean(y_top2_hit)

    # Benchmark single-image CPU inference latency across 50 warm runs
    dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
    _ = model.predict(dummy, verbose=0)
    latencies = []
    for _ in range(50):
        t0 = time.perf_counter()
        _ = model.predict(dummy, verbose=0)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    print(f"Overall Top-1 Accuracy: {acc * 100:.2f}% (Target >= 85.0%)")
    print(f"Overall Top-2 Accuracy: {top2_acc * 100:.2f}%")
    print(f"CPU Latency p50: {np.percentile(latencies, 50):.2f} ms | p95: {np.percentile(latencies, 95):.2f} ms")
    print("\nPer-Class Classification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/crop_disease_mobilenetv2.keras")
    parser.add_argument("--test_dir", default="data/splits/test")
    args = parser.parse_args()
    evaluate_model(args.model, args.test_dir)
