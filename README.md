# Cattle Vision — Cow Detection & Segmentation

Fine-tuning **YOLOv11** and **SAM 2** for automated cow detection and segmentation in farm camera footage.

> **Prerequisites:** Basic Python · Introductory ML  
> **Tools:** PyTorch · Ultralytics YOLOv11 · SAM 2 · Label Studio

---

## Overview

This project builds a deep learning pipeline to detect and segment individual cows in farm camera footage — a core capability for automated herd monitoring, counting, and body condition scoring.

Two camera views are supported:

- **Top-view (overhead):** useful for counting, tracking movement, and detecting lying vs. standing behaviour.
- **Side-view (lateral):** useful for body condition scoring, lameness detection, and posture analysis.

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
git clone https://github.com/<your-username>/cow-detection-segmentation.git
cd cow-detection-segmentation

python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 2. Install SAM 2

SAM 2 must be installed separately:

```bash
pip install git+https://github.com/facebookresearch/sam2.git
```

### 3. Place video files

Put your farm videos in `data/raw/videos/`. File naming conventions:

- Top-view files: name must contain `top` or `overhead`
- Side-view files: name must contain `side` or `lateral`

---

## Stage 1 — Frame Extraction

```bash
python src/utils/extract_frames.py --video_dir data/raw/videos --output_dir data/frames --fps 1
```

Extracting at 1 FPS from a 10-minute video gives ~600 frames. Adjust with `--max_frames` if needed. Use `--fps 0.5` if cows barely move between frames, or `--fps 2` for more diversity.

---

## Stage 2 — Annotation with Label Studio

[Label Studio](https://labelstud.io) is a free, open-source annotation tool that supports bounding boxes and polygon segmentation — both of which are needed for this project.

### Installation

```bash
pip install label-studio
label-studio start
```

Label Studio runs locally at `http://localhost:8080`. Create a free account on first launch.

### Create a Project

1. Click **Create Project** and give it a name (e.g. `Cow Detection Segmentation`).
2. Under **Labeling Setup**, choose **Object Detection with Bounding Boxes** (for detection) or **Semantic Segmentation with Polygons** (for segmentation).
3. Add a label called `cow`.
4. Under **Data Import**, upload your extracted `.jpg` frames from `data/frames/`.

### Annotating Bounding Boxes (for YOLOv11)

1. Open a task. You will see the first frame.
2. Select the **Rectangle** tool from the toolbar.
3. Choose the `cow` label.
4. Click and drag to draw a tight box around each cow.
5. One box per cow. Move to the next frame with the arrow key.

**Guidelines:**
- Draw the box as tight as possible around the cow's body.
- Include the head; slight clipping at image edges is fine.
- Skip cows that are more than 50% occluded.
- Aim for at least 300–500 annotated instances total.

### Annotating Polygon Masks (for SAM 2)

1. Select the **Polygon** tool from the toolbar.
2. Choose the `cow` label.
3. Click around the cow's body boundary (10–20 points is sufficient).
4. Close the polygon by clicking the first point or pressing `Enter`.
5. Adjust any point by clicking and dragging.

> **Tip:** Segmentation annotation is slower. It is fine to annotate a subset of frames with polygon masks (150–200 images) while annotating all frames with bounding boxes. SAM 2 generalises well from fewer but high-quality masks.

### Exporting Annotations

**For detection (YOLO format):**

1. In your project, go to **Export**.
2. Choose **YOLO** format.
3. Download the ZIP and copy the `.txt` label files to `data/annotations/detection/`.

**For segmentation (COCO format):**

1. In your project, go to **Export**.
2. Choose **COCO JSON** format.
3. Download the ZIP and copy:
   - `instances_default.json` → `data/annotations/segmentation/train_coco.json`
   - Repeat for val and test splits: `val_coco.json`, `test_coco.json`

> **Important:** YOLO label files must share the same filename stem as the image. For example, `frame_000001.jpg` must have `frame_000001.txt`. Label Studio handles this automatically on YOLO export.

---

## Stage 3 — Dataset Split

```bash
python src/utils/dataset_split.py
```

Splits the annotated data 70% train / 20% validation / 10% test automatically.

---

## Stage 4 — Fine-Tune YOLOv11

```bash
python src/detection/train_yolo.py \
    --config  configs/yolo_config.yaml \
    --model   yolo11m.pt              \
    --epochs  100                     \
    --imgsz   640                     \
    --batch   16                      \
    --device  0
```

The medium model `yolo11m.pt` is recommended for this project. Weights are downloaded automatically on first run.

| Model | Parameters | mAP COCO | Use when |
|-------|-----------|----------|----------|
| yolo11n | 2.6M | 39.5 | Edge / very limited GPU |
| yolo11s | 9.4M | 47.0 | Limited GPU memory |
| **yolo11m** | **20.1M** | **51.5** | **This project (default)** |
| yolo11l | 25.3M | 53.4 | More data available |
| yolo11x | 56.9M | 54.7 | Maximum accuracy |

---

## Stage 5 — Fine-Tune SAM 2

```bash
python src/segmentation/train_sam2.py \
    --config configs/sam2_config.yaml  \
    --epochs 50                        \
    --device cuda:0
```

Training uses a two-phase strategy:

- **Phase 1 (epochs 1–10):** Image encoder is frozen; only the mask decoder and prompt encoder train.
- **Phase 2 (epochs 11+):** Image encoder is unfrozen at 10× lower learning rate.

---

## Stage 6 — Evaluate

```bash
# Detection
python src/detection/evaluate_yolo.py \
    --weights results/detection/cow_yolo11/weights/best.pt \
    --split   test

# Segmentation
python src/segmentation/evaluate_sam2.py \
    --checkpoint results/segmentation/best_sam2.pth \
    --split      test
```

Outputs saved to `results/`:

- `metrics_summary.json`
- `metrics_bar_chart.png`
- `iou_distribution.png`
- Confusion matrix and PR curve (YOLO, generated automatically)

### Target Metrics

| Metric | Good range |
|--------|-----------|
| mAP@50 | > 0.80 |
| mAP@50-95 | > 0.55 |
| Precision | > 0.85 |
| Recall | > 0.80 |
| Mean IoU (masks) | > 0.75 |
| Dice coefficient | > 0.85 |

---

## Running the Full Pipeline (Detection + Segmentation)

After training both models, run automated detection followed by segmentation:

```bash
python src/segmentation/predict_sam2.py \
    --checkpoint results/segmentation/best_sam2.pth \
    --image      data/frames/test/frame_000050.jpg  \
    --prompt_mode box                               \
    --yolo_weights results/detection/cow_yolo11/weights/best.pt
```

YOLOv11 detects cows and returns bounding boxes → each box is passed to SAM 2 as a prompt → SAM 2 returns a precise pixel mask per cow.

---

## Deliverables

1. **GitHub repository** — all code and configs committed.
2. **Annotated dataset** — Label Studio exported YOLO labels + COCO JSON in `data/annotations/`.
3. **YOLOv11 weights** — `results/detection/cow_yolo11/weights/best.pt`
4. **SAM 2 checkpoint** — `results/segmentation/best_sam2.pth`
5. **Evaluation outputs** — metric JSON files and plots in `results/`
6. **Poster presentation** — approach, results, and discussion.

---

## Git Workflow

Commit regularly — after each significant step is a good rule of thumb.

```bash
git add data/annotations/ configs/
git commit -m "Add Label Studio annotations for topview_barn1"
git push origin main

# After training:
git add results/detection/eval_test/ results/segmentation/eval/
git add results/detection/cow_yolo11/weights/best.pt
git commit -m "Add evaluation results and best model weights"
git push origin main
```

**Never commit:**
- Raw video files (`data/raw/videos/`)
- Extracted frame images (`data/frames/**/*.jpg`)
- Intermediate checkpoints (`checkpoint_epoch*.pth`)
- Virtual environment (`venv/`)

These are already excluded in `.gitignore`.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| CUDA out of memory | Reduce `--batch` to 8 or 4; set `batch_size: 2` in `configs/sam2_config.yaml` |
| `ultralytics` / `sam2` not found | Activate your venv and re-run pip installs |
| No images found in `data/frames/` | Run the frame extraction script first |
| YOLO mAP stays near 0 | Check `.txt` label format (`class_id cx cy w h`, normalised 0–1) and `yolo_config.yaml` paths |
| SAM 2 loss is NaN | Set `mixed_precision: false` in config, reduce LR to `5e-5`, reduce batch size |

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA 8 GB (RTX 3070) | NVIDIA 16+ GB (RTX 3090 / A100) |
| RAM | 16 GB | 32 GB |
| Storage | 50 GB | 100+ GB |
| Python | 3.10 | 3.11 |
| PyTorch | 2.0 | 2.2+ |
| CUDA | 11.8 | 12.1+ |

> **No GPU?** Use [Google Colab](https://colab.research.google.com) — free T4 GPU access. Training YOLOv11m for 100 epochs takes roughly 30–60 minutes on a T4 with 500 images.

---

## Annotation Quality Checklist

Before training, verify:

- [ ] Every cow in a frame has a bounding box — none missed
- [ ] Boxes are tight (not including excessive background or clipping the cow)
- [ ] Polygon masks trace the actual body outline
- [ ] YOLO `.txt` files have exactly 5 values per line: `class_id cx cy w h`
- [ ] COCO JSON has a `segmentation` field as a list of polygon coordinates
- [ ] Number of YOLO label files roughly matches number of annotated frames
