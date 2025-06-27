#!/usr/bin/env python3
"""
Single‑image Plant–Disease Detector (with 4‑level health grading)
=================================================================

Usage
-----
    python infer_single_grades.py path/to/leaf.jpg

Produces
--------
    annotated_leaf.jpg   # original image with green overlay text

Health‑status mapping
---------------------
Based on *disease probability* (0 → perfectly healthy, 1 → certainly diseased):

    < 0.50                      → Healthy
    0.50 ≤ p < 0.70             → Moderately Healthy
    0.70 ≤ p < 0.90             → Close to Deterioration
    ≥ 0.90                      → Diseased
"""

import argparse
import os
import cv2
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# ----------------------------- Settings ------------------------------------ #
MODEL_NAME       = "Diginsa/Plant-Disease-Detection-Project"
CONF_THRESHOLD   = 0.95   # original override threshold
# Grading thresholds (disease probability bands)
TH_MODERATE      = 0.50
TH_CLOSE         = 0.70
TH_DISEASED      = 0.90

# ------------------------- Model & Processor ------------------------------- #
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor  = AutoImageProcessor.from_pretrained(MODEL_NAME, use_fast=True)
model      = AutoModelForImageClassification.from_pretrained(MODEL_NAME).to(device)
model.eval()

# -------------------- Helper: run model on PIL image ----------------------- #
@torch.inference_mode()
def predict(pil_img, conf_thresh: float = CONF_THRESHOLD):
    """
    Returns (final_label, model_conf, disease_prob)

    final_label – 'Healthy' OR disease name
    model_conf  – probability associated with final_label
    disease_prob– probability that leaf is diseased (for grading)
    """
    inputs = processor(images=pil_img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)
    probs   = torch.softmax(outputs.logits, dim=1)[0]
    idx     = outputs.logits.argmax(-1).item()
    label   = model.config.id2label[idx]
    conf    = probs[idx].item()

    # Override: low‑confidence disease ⇒ call it healthy
    if label.lower() != "healthy" and conf < conf_thresh:
        label = "Healthy"
        conf  = 1.0 - conf   # switch to healthy confidence

    # Disease probability for grading
    disease_prob = conf if label.lower() != "healthy" else 1.0 - conf
    return label, conf, disease_prob


def grade_health(disease_prob: float) -> str:
    """Map disease probability to one of four health-status labels."""
    if disease_prob >= TH_DISEASED:
        return "Diseased"
    elif disease_prob >= TH_CLOSE:
        return "Close to Deterioration"
    elif disease_prob >= TH_MODERATE:
        return "Moderately Healthy"
    else:
        return "Confidence Score"


# --------------------------- Main routine ---------------------------------- #
def annotate_image(img_path: str):
    if not os.path.exists(img_path):
        raise FileNotFoundError(img_path)

    bgr = cv2.imread(img_path)
    if bgr is None:
        raise ValueError(f"Could not load {img_path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    label, conf, disease_prob = predict(Image.fromarray(rgb))
    status = grade_health(disease_prob)

    # Compose overlay text
    if label.lower() == "healthy":
        text = f"{status} ({(1 - disease_prob):.2%})"
    else:
        text = f"{label} – {status} ({conf:.2%})"

    cv2.putText(bgr, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out_path = os.path.join(
        os.path.dirname(img_path),
        f"annotated_{os.path.basename(img_path)}"
    )
    cv2.imwrite(out_path, bgr)
    print(f"✅ {text}  →  {out_path}")


# ----------------------------- CLI ----------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Plant‑disease inference with 4‑level health grading.")
    parser.add_argument("image", help="Path to a leaf image")
    args = parser.parse_args()
    annotate_image(args.image)


if __name__ == "__main__":
    main()
