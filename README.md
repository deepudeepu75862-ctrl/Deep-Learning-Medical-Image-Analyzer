# Deep Learning Image Analyzer (Educational Demo)

A small Streamlit project demonstrating an end-to-end image-classification workflow using TensorFlow/Keras and MobileNetV2.

## Important limitation
This starter project uses MobileNetV2 pretrained on **ImageNet**, a general-object dataset. It is **not trained, tested, or validated for medical images** and must not be used for diagnosis, treatment, or clinical decisions. The app labels predictions as general ImageNet labels.

For a legitimate medical-imaging research extension, use a properly licensed public dataset, patient-level train/validation/test splits, expert-reviewed labels, relevant metrics, and ethics/privacy safeguards. Do not upload identifiable patient data to public services.

## Requirements
- Python 3.10–3.12 recommended
- Internet connection for first-time dependency installation and pretrained model download

## Windows setup (PowerShell)

```powershell
cd path\to\Deep_Learning_Medical_Image_Analyzer
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If `py -3.12` is unavailable, install Python 3.12 and select it during setup. If PowerShell blocks activation, run this once in the current terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

The browser should open at the local Streamlit address shown in the terminal (usually `http://localhost:8501`).

## Project structure

```text
Deep_Learning_Medical_Image_Analyzer/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## GitHub upload

```powershell
git init
git add .
git commit -m "Add educational deep learning image analyzer"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

Replace the URL with your own repository URL. If the repository already contains a README or other commits, pull and reconcile those changes before pushing.

## Suggested advanced extensions
- Train a CNN or fine-tune a suitable pretrained model on a permitted medical-image dataset.
- Report sensitivity, specificity, precision, recall, F1-score, ROC-AUC, and confusion matrix.
- Add patient-level splitting to reduce data leakage.
- Add Grad-CAM only after choosing and validating a medically appropriate model.
- Document intended use, dataset license, limitations, and model-card information.
