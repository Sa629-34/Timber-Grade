# 🪵 TimberGrade - AI-Based Wooden Plank Defect Detection and Quality Grading System

ROSPL Lab Project | Python + Open-Source Tools

## Folder mein kya hai

| File | Kaam |
|---|---|
| `TimberGrade_Training.ipynb` | Google Colab notebook: dataset download + YOLOv8 training |
| `grading.py` | Defects ke hisaab se Grade A / B / C dena |
| `app.py` | Streamlit website: photo upload/webcam, result, dashboard, SQLite history |
| `models/` | Yahan Colab se download ki `best.pt` file daalni hai |
| `requirements.txt` | Libraries ki list |

## PART 1: Model train karna (Google Colab, ~1 ghanta)

1. https://colab.research.google.com kholo -> **File -> Upload notebook** -> `TimberGrade_Training.ipynb` chuno
2. **Runtime -> Change runtime type -> T4 GPU** -> Save
3. Har cell ek-ek karke upar se neeche chalao (▶️)
4. Last cell 2 files download karega:
   - `best.pt` -> is folder ke andar `models` folder mein daalo
   - `training_results.zip` -> report ke graphs (accuracy, confusion matrix)

## PART 2: Website chalana (apne laptop par)

VS Code mein TimberGrade folder kholo -> right-click -> **Open in Integrated Terminal**:
```
pip install -r requirements.txt
python -m streamlit run app.py
```
Browser mein `localhost:8501` khulega.

## Grading kaise hota hai

Har defect ke penalty points hain (crack sabse zyada, live knot sabse kam). Bada defect = zyada penalty.
Quality score = 100 - penalties

| Grade | Condition | Use |
|---|---|---|
| A | Score >= 85 aur koi crack nahi | Furniture, export |
| B | Score >= 60 | Construction, darwaze |
| C | Score < 60 | Packaging, pallets |

## Dataset
Kaggle: "Large Scale Image Dataset of Wood Surface Defects" (4,000 photos, 8 defect types:
Live knot, Dead knot, Knot with crack, Knot missing, Crack, Resin, Marrow, Quartzity)

## Tools
Python, Ultralytics YOLOv8, OpenCV, Streamlit, SQLite, Google Colab
