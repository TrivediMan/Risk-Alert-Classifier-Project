# 🛡️ Risk Alert Classifier

An early-warning decision-support tool that estimates whether a customer may be **high risk**, built with scikit-learn and served through an interactive **Streamlit** web app.

🔗 **Live App:** [https://robust-5fd7qcgpjn6hxvnha76eg4.streamlit.app/](https://risk-alert-classifier-project-klkq486smwv8ztbfv23akz.streamlit.app/)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://risk-alert-classifier-project-klkq486smwv8ztbfv23akz.streamlit.app/)

> ⚠️ **Disclaimer:** This tool supports human review. It does **not** establish fraud or default. Always validate alerts and follow your institution's approved decision process before taking action.

---

## ✨ Features

- **Interactive risk assessment** – enter customer details through auto-generated form widgets (number inputs, dropdowns) built from the dataset's schema.
- **Probability-based output** – shows the estimated high-risk probability along with a clear *High Risk Alert* / *Low Risk* result.
- **Adjustable threshold** – a slider (10%–90%) lets you trade off between catching more risky customers and reducing false positives.
- **Smart model loading** – uses a saved model (`Best_Model.pkl` or `risk_alert_model.joblib`) if available; otherwise it automatically trains a Random Forest fallback from the dataset.
- **Robust preprocessing** – median imputation for numeric features, most-frequent imputation + one-hot encoding for categorical features, and `class_weight="balanced"` to handle class imbalance.
- **Dataset overview** – an expandable panel showing row/column counts, target classes and features used.

---

## 📁 Project Structure

```
Risk-Alert-Classifier-Project/
├── data/               # Dataset (Risk_Alert_Classifier_Dataset_4600.csv)
├── model/              # Saved models and metadata (fallback model is written here)
├── notebook/           # Exploration, training and evaluation notebook(s)
├── app.py              # Streamlit application
├── requirements.txt    # Python dependencies
└── README.md
```

---

## 🧰 Tech Stack

- Python 3.9+
- [Streamlit](https://streamlit.io/)
- scikit-learn (Random Forest, `ColumnTransformer`, `Pipeline`)
- pandas, NumPy
- joblib / pickle (model persistence)

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/TrivediMan/Risk-Alert-Classifier-Project.git
cd Risk-Alert-Classifier-Project
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

---

## 📊 Dataset

The app expects a CSV named **`Risk_Alert_Classifier_Dataset_4600.csv`**, placed in either the `data/` folder or the project root.

Requirements for the CSV:

| Column | Purpose |
| --- | --- |
| `risk_status` | **Target** column (must contain at least two classes) |
| `customer_id` | Identifier – dropped before training |
| `last_transaction_date` | Dropped before training |
| *all other columns* | Used as model features |

---

## 🧠 How It Works

1. **Load data** – reads the CSV and drops rows with a missing `risk_status`.
2. **Load or train a model**
   - Looks for `Best_Model.pkl` or `risk_alert_model.joblib` (plus optional `model_metadata.json` for input columns and threshold).
   - If none can be loaded, trains a Random Forest (200 trees, balanced class weights) and saves it as `model/Fallback_Model.pkl`.
3. **Collect inputs** – builds a form with defaults set to the dataset's median (numeric) or most common value (categorical).
4. **Predict** – computes the probability of the high-risk class and compares it to the chosen threshold.
5. **Display result** – shows the probability, threshold, a progress bar and a High Risk / Low Risk message.

The high-risk class is detected automatically from labels such as `1`, `high risk`, `high`, `risky`, `yes` or `true`.

---

## 🖥️ Usage

1. Launch the app with `streamlit run app.py`.
2. Fill in the customer information (defaults are pre-populated).
3. Adjust the **High-risk probability threshold** if needed.
4. Click **Assess Risk**.
5. Review the probability and recommendation, then validate with your standard review process.

---

## 🔧 Troubleshooting

| Problem | Fix |
| --- | --- |
| *Dataset CSV was not found* | Place `Risk_Alert_Classifier_Dataset_4600.csv` in `data/` or the project root. |
| *Existing model could not be loaded* | Usually a scikit-learn version mismatch. Rename the old model file and restart – a compatible fallback is trained automatically. |
| *Model does not support probability predictions* | Use a classifier that implements `predict_proba`, or remove the saved model so the fallback is used. |

---

## 🗺️ Future Improvements

- Add model evaluation metrics (precision, recall, ROC-AUC) to the app
- Feature-importance / SHAP explanations for each alert
- Batch prediction via CSV upload
- Dockerfile and cloud deployment (Streamlit Community Cloud)

---

## 🤝 Contributing

Contributions, issues and feature requests are welcome. Feel free to open an issue or submit a pull request.

---

## 📄 License

No license has been specified yet. Consider adding one (e.g. MIT) to clarify how others may use this project.

---

## 👤 Author

**TrivediMan** – [GitHub](https://github.com/TrivediMan)
