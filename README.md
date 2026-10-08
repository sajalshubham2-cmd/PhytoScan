# PhytoScan — AI Crop-Disease Detector & Engineering Specification

Farmer's assistant web application and complete 14-section ML engineering specification for diagnosing **Tomato** and **Maize** foliar diseases from leaf photos using **MobileNetV2 transfer learning** on the **PlantVillage** dataset, with crop-aware softmax renormalization, confidence/margin fallback gates, a farmer-friendly agronomy knowledge base, and a `localStorage`-based history of the last 5 diagnostic results.

---

## Quick Start

### 1. Interactive Web Console (React + Vite + TypeScript)
```bash
npm install
npm run dev
```
Open `http://localhost:3000` to use:
- **Diagnostic Main Screen**: Select **Tomato** or **Maize**, upload a `.jpg`/`.jpeg`/`.png` leaf image (up to 5 MB) or click any of the 10 calibrated PlantVillage & edge-case presets.
- **Last 5 Diagnostic Results (`localStorage`)**: Automatically stores and displays the 5 most recent diagnoses under key `phytoscan_diagnostic_history_v1` on the main screen. Click any history card to reload that diagnosis.
- **8-Class Agronomy Knowledge Base**: Complete symptoms, organic remedies, generic chemical active ingredients, and prevention tips.
- **Full 14-Section ML Specification**: Interactive specification viewer with tables, split math, ASCII architecture & wireframes, runnable Python code skeletons, test cases, and risk register.

### 2. Python 3.10 + Flask + TensorFlow/Keras Backend
```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 1. Prepare 8-class PlantVillage subset (capped at 1,000 images/class, 70/15/15 split)
python scripts/prepare_data.py --source /path/to/PlantVillage --output data/splits

# 2. Train Two-Phase MobileNetV2 model (Phase 1 head-only 8 epochs, Phase 2 fine-tune top 34 layers 12 epochs)
python scripts/train.py --data_dir data/splits --output models/crop_disease_mobilenetv2.keras

# 3. Evaluate on 1,125-image test split & verify CPU latency (< 250 ms)
python scripts/evaluate.py --model models/crop_disease_mobilenetv2.keras --test_dir data/splits/test

# 4. Run Flask Server on CPU Laptop
FLASK_APP=app/main.py flask run --host=0.0.0.0 --port=5000
```

---

## Scope & Dataset Summary (8 PlantVillage Classes)

| Idx | Crop | Exact Dataset Folder Name | Display Name | Available | Used (Cap=1000) | Train (70%) | Val (15%) | Test (15%) | Class Weight |
|---:|:---|:---|:---|---:|---:|---:|---:|---:|---:|
| 0 | Tomato | `Tomato___Early_blight` | Tomato Early Blight | 1,000 | 1,000 | 700 | 150 | 150 | 0.9203 |
| 1 | Tomato | `Tomato___Late_blight` | Tomato Late Blight | 1,909 | 1,000 | 700 | 150 | 150 | 0.9203 |
| 2 | Tomato | `Tomato___Septoria_leaf_spot` | Tomato Septoria Leaf Spot | 1,771 | 1,000 | 700 | 150 | 150 | 0.9203 |
| 3 | Tomato | `Tomato___healthy` | Healthy Tomato Leaf | 1,591 | 1,000 | 700 | 150 | 150 | 0.9203 |
| 4 | Maize | `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` | Maize Gray Leaf Spot | 513 | 513 | 359 | 77 | 77 | 1.7945 |
| 5 | Maize | `Corn_(maize)___Common_rust_` | Maize Common Rust | 1,192 | 1,000 | 700 | 150 | 150 | 0.9203 |
| 6 | Maize | `Corn_(maize)___Northern_Leaf_Blight` | Maize Northern Leaf Blight | 985 | 985 | 689 | 148 | 148 | 0.9344 |
| 7 | Maize | `Corn_(maize)___healthy` | Healthy Maize Leaf | 1,162 | 1,000 | 700 | 150 | 150 | 0.9203 |
| **Total** | **2 Crops** | **8 Classes** | — | **10,123** | **7,498** | **5,248** | **1,125** | **1,125** | **mean = 1.0326** |
