# 🪵 TimberGrade - AI-Based Wooden Plank Defect Detection and Quality Grading System

ROSPL Lab Project (2026-27) | Python + Open-Source Tools

TimberGrade detects surface defects on wooden planks from a photo using a YOLOv8 model and gives
the plank a quality grade (A / B / C) with its recommended use.

## Features
- **Inspect plank** - upload a photo, take one with the webcam, or pick one from the dataset
- **AI detection** - 8 defect types: Live knot, Dead knot, Knot with crack, Knot missing, Crack, Resin, Marrow, Quartzity
- **Grading** - quality score (0-100), Grade A / B / C and recommended use
- **Dashboard** - inspection history (SQLite), grade charts, CSV report download
- **Dataset & Model** - dataset details, training graphs and confusion matrix

## Project structure
| File / folder | Purpose |
|---|---|
| `app.py` | Streamlit website |
| `grading.py` | Grading logic (penalty points -> score -> grade) |
| `models/best.pt` | Trained YOLOv8n model |
| `training_results/` | Training graphs, confusion matrix, sample predictions |
| `TimberGrade_Training.ipynb` | Google Colab notebook used to train the model |
| `dataset/` | (not in repo) put dataset images here to use "From dataset" |

## How to run (Windows)
```
python -m pip install -r requirements.txt
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
python -m streamlit run app.py
```
The website opens at http://localhost:8501

If you get a `c10.dll` error, install the Microsoft Visual C++ Redistributable:
https://aka.ms/vs/17/release/vc_redist.x64.exe

## Dataset (optional, 3.3 GB)
Kaggle: [Large Scale Image Dataset of Wood Surface Defects](https://www.kaggle.com/datasets/nomihsa965/large-scale-image-dataset-of-wood-surface-defects)
- 4,000 images (2800 x 1024), YOLO labels, 8 defect classes
- Download, extract, and put the images inside a folder named `dataset` next to `app.py`

## Grading rules
Score = 100 - sum of penalties (bigger defect = bigger penalty)

| Grade | Condition | Use |
|---|---|---|
| A | Score >= 85 and no crack | Furniture, interiors, export |
| B | Score >= 60 | Construction, doors, general use |
| C | Score < 60 | Packaging, pallets, crates |

## Tools
Python, Ultralytics YOLOv8, PyTorch, OpenCV, Streamlit, Pandas, SQLite, Google Colab
