# Cattle Vision — Cow Detection & Segmentation

Fine-tuning YOLOv11 and SAM 2 for automated cow detection and segmentation in farm camera footage.

> Prerequisites: Basic Python · Introductory ML  
> Tools: PyTorch · Ultralytics YOLOv11 · SAM 2 · Label Studio

---

## Overview

This project builds a deep learning pipeline to detect and segment individual cows in farm camera footage — a core capability for automated herd monitoring, counting, and body condition scoring.

Two camera views are supported:

- Top-view (overhead): useful for counting, tracking movement, and detecting lying vs. standing behaviour.  
- Side-view (lateral): useful for body condition scoring, lameness detection, and posture analysis.  

| Model | Task | Output |
|-------|------|--------|
| YOLOv11 | Object Detection | Bounding boxes + confidence scores |
| SAM 2 | Instance Segmentation | Binary pixel masks per cow |

---

## Project Workflow

| Stage | Description | Script / Tool |
|-------|-------------|---------------|
| 1 | Extract frames from farm videos at 1 FPS | `src/utils/extract_frames.py` |
| 2 | Annotate frames in Label Studio | Label Studio |
| 3 | Export annotations and split dataset 70/20/10 | `src/utils/dataset_split.py` |
| 4 | Fine-tune YOLOv11 for cow detection | `src/detection/train_yolo.py` |
| 5 | Fine-tune SAM 2 for cow segmentation | `src/segmentation/train_sam2.py` |
| 6 | Evaluate both models and save results | `evaluate_yolo.py` / `evaluate_sam2.py` |

---

## Setup

### 1. Clone and create environment

```bash
git clone git@github.com:nivethjunnithan/cattle-vision-finetuning.git
cd cattle-vision-finetuning

python -m venv .cattle-vision-env
source .cattle-vision-env/bin/activate        # Mac/Linux
# .cattle-vision-env\Scripts\activate         # Windows

pip install -r requirements.txt
```

> ⚠️ Note: Label Studio is installed globally using pipx, not inside this virtual environment.

---

### 2. Install SAM 2

```bash
pip install git+https://github.com/facebookresearch/sam2.git
```

---

### 3. Install Label Studio (via pipx)

```bash
brew install pipx
pipx ensurepath
source ~/.zprofile

pipx install label-studio --python /opt/homebrew/bin/python3.10
label-studio
```

Open: http://localhost:8080

---

### 4. Place video files

Put your farm videos in `data/raw/videos/`.

---

## Stage 1 — Frame Extraction

```bash
python src/utils/extract_frames.py \
  --video_dir data/raw/videos \
  --output_dir data/frames \
  --fps 1
```

---

## Stage 2 — Annotation with Label Studio

- Draw tight bounding boxes  
- Use polygon masks for segmentation  
- Export YOLO + COCO formats  

---

## Stage 3 — Dataset Split

```bash
python src/utils/dataset_split.py
```

---

## Stage 4 — Fine-Tune YOLOv11

```bash
python src/detection/train_yolo.py \
  --config configs/yolo_config.yaml \
  --model yolo11m.pt \
  --epochs 100
```

---

## Stage 5 — Fine-Tune SAM 2

```bash
python src/segmentation/train_sam2.py
```

---

## Stage 6 — Evaluate

```bash
python src/detection/evaluate_yolo.py
```

---

## Running Full Pipeline

```bash
python src/segmentation/predict_sam2.py
```

---

## Git Workflow

```bash
git checkout -b feature/annotation-boundary-refinement
git push -u origin feature/annotation-boundary-refinement
```

---

## Important Rules

❌ Never commit:
- data/raw/videos/
- data/frames/
- *.pth
- .cattle-vision-env/

---

## Hardware

| Component | Recommended |
|----------|------------|
| GPU | 16GB+ |
| RAM | 32GB |
| Python | 3.10 |

---

## Annotation Checklist

- [ ] All cows labeled  
- [ ] Tight bounding boxes  
- [ ] Accurate polygon masks  
- [ ] Correct YOLO format  
- [ ] Valid COCO JSON  

---

## Troubleshooting

| Problem | Fix |
|--------|-----|
| CUDA out of memory | Reduce batch size |
| YOLO not learning | Check label format |
| SAM loss NaN | Reduce LR |
| No frames found | Run extraction |

---
