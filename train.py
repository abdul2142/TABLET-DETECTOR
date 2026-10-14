from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score
from sklearn.model_selection import train_test_split

from image_utils import (DEFECT_TYPES, detect, extract_features, make_blister_pack_image,
                         make_multi_tablet_image, make_tablet_image, save_image)

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"

#: Share of the dataset taken from whole-photo detections rather than single
#: centred tiles. At inference the model only ever sees crops that `detect`
#: produced, so training on them removes a train/serve mismatch that cost ~16
#: points of chip recall.
DETECTOR_SHARE = 0.5

PACK_LAYOUTS = ((5, 2), (4, 3), (3, 4), (6, 2), (4, 1))


def _isolated(count: int, preview_dir: Path):
    """Single centred tablets: half normal, half spread over the defect types.

    Half sit loose on a table, half in a blister-pack pocket.
    """
    X, y = [], []
    defects = DEFECT_TYPES[1:]
    for i in range(count):
        defect_type = "none" if i % 2 == 0 else defects[(i // 2) % len(defects)]
        image, _, _ = make_tablet_image(defect_type=defect_type, seed=1000 + i,
                                        packaged=(i // 8) % 2 == 1)
        X.append(extract_features(image))
        y.append(DEFECT_TYPES.index(defect_type))
        if i < 12:
            save_image(image, preview_dir / f"sample_{i:02d}_{defect_type.replace(' ', '_')}.png")
    return X, y


def _from_detections(target: int):
    """Crops taken the way the app takes them: run `detect` over a whole photo.

    Images whose tablet count does not match the ground truth are skipped, so a
    crop is never paired with the wrong label.
    """
    X, y, seed, skipped = [], [], 50_000, 0
    while len(X) < target:
        seed += 1
        if seed % 2:
            rows, cols = PACK_LAYOUTS[seed % len(PACK_LAYOUTS)]
            image, truth = make_blister_pack_image(rows=rows, cols=cols, defect_rate=0.5, seed=seed)
        else:
            image, truth = make_multi_tablet_image(count=4 + seed % 7, defect_rate=0.5, seed=seed)

        boxes, _ = detect(image)
        if len(boxes) != len(truth):
            skipped += 1
            continue
        for box, defect_type in zip(boxes, truth):
            X.append(extract_features(image.crop(box)))
            y.append(DEFECT_TYPES.index(defect_type))
    return X, y, skipped


def build_dataset(samples: int, preview_dir: Path):
    preview_dir.mkdir(parents=True, exist_ok=True)
    for old in preview_dir.glob("sample_*.png"):
        old.unlink()

    detector_target = int(samples * DETECTOR_SHARE)
    print(f"  {samples - detector_target} isolated tablets...")
    X_iso, y_iso = _isolated(samples - detector_target, preview_dir)
    print(f"  {detector_target} crops from whole-photo detections...")
    X_det, y_det, skipped = _from_detections(detector_target)
    if skipped:
        print(f"  (skipped {skipped} photos where the detected count did not match ground truth)")

    return np.asarray(X_iso + X_det), np.asarray(y_iso + y_det)


def main():
    parser = argparse.ArgumentParser(description="Train a synthetic tablet defect classifier")
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--output", default=str(ARTIFACTS / "tablet_defect_model.joblib"))
    args = parser.parse_args()

    print(f"Generating {args.samples} synthetic training crops...")
    X, y = build_dataset(args.samples, ARTIFACTS / "previews")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Gradient boosting gives far better-calibrated confidence than a random forest
    # here, so very few tablets end up in the app's "uncertain" band.
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Collapse defect types to the binary Normal / Defective decision as well.
    binary_true, binary_pred = (y_test > 0).astype(int), (predictions > 0).astype(int)
    binary_acc = accuracy_score(binary_true, binary_pred)
    type_acc = accuracy_score(y_test, predictions)

    print("\n=== Normal vs. Defective ===")
    print(classification_report(binary_true, binary_pred, target_names=["Normal", "Defective"], zero_division=0))
    print(confusion_matrix(binary_true, binary_pred))
    print("\n=== Defect type ===")
    print(classification_report(y_test, predictions, target_names=DEFECT_TYPES, zero_division=0))
    print(confusion_matrix(y_test, predictions))

    # Per-class recall is recorded because an aggregate score hides a weak
    # class, and a missed defect is the expensive error in an inspection tool.
    recalls = recall_score(y_test, predictions, average=None,
                           labels=range(len(DEFECT_TYPES)), zero_division=0)
    per_class = {name: round(float(r), 4) for name, r in zip(DEFECT_TYPES, recalls)}
    missed = {name: round(float(((y_test == i) & (predictions == 0)).sum() / max((y_test == i).sum(), 1)), 4)
              for i, name in enumerate(DEFECT_TYPES) if i > 0}

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path, compress=3)
    metadata = {
        "task": "Educational synthetic tablet defect classification",
        "labels": {"0": "Normal Tablet", "1": "Defective Tablet"},
        "defect_types": DEFECT_TYPES,
        "model": "Gradient Boosting",
        "samples": args.samples,
        "detector_sourced_share": DETECTOR_SHARE,
        "features": int(X.shape[1]),
        "test_accuracy_binary": round(binary_acc, 4),
        "test_accuracy_defect_type": round(type_acc, 4),
        "recall_per_class": per_class,
        "missed_as_normal": missed,
        "warning": "Not suitable for pharmaceutical QA or regulatory deployment.",
    }
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nBinary accuracy: {binary_acc:.2%} | Defect-type accuracy: {type_acc:.2%}")
    print("Recall per class: " + " · ".join(f"{k} {v:.1%}" for k, v in per_class.items()))
    print("Missed as normal: " + " · ".join(f"{k} {v:.1%}" for k, v in missed.items()))
    print(f"Saved model to: {output_path}")


if __name__ == "__main__":
    main()
