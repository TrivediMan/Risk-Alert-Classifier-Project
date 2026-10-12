
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# --------------------------------------------------
# 1. APPLICATION CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Risk Alert Classifier",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Risk Alert Classifier")
st.write(
    "Early-warning decision support for identifying "
    "customers who may be high risk."
)

st.warning(
    "This tool supports review; it does not establish fraud "
    "or default. Validate alerts before taking action."
)

# Automatically use the folder containing app.py.
# No /mnt/data path or hard-coded Windows username is needed.
BASE_DIR = Path(__file__).resolve().parent
DATA_FILENAME = "Risk_Alert_Classifier_Dataset_4600.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DROP_COLUMNS = [
    "customer_id",
    "last_transaction_date",
    "risk_status",
]


# --------------------------------------------------
# 2. FIND FILES
# --------------------------------------------------

def find_file(filename):
    """Find a file in the project folder or its subfolders."""

    direct_path = BASE_DIR / filename

    if direct_path.is_file():
        return direct_path

    ignored = {
        ".git",
        ".venv",
        "venv",
        "env",
        "__pycache__",
    }

    for path in BASE_DIR.rglob(filename):
        relative_parts = path.relative_to(BASE_DIR).parts

        if any(
            part in ignored or part.startswith(".")
            for part in relative_parts
        ):
            continue

        if path.is_file():
            return path

    return None


# --------------------------------------------------
# 3. LOAD DATASET
# --------------------------------------------------

data_path = find_file(DATA_FILENAME)

if data_path is None:
    st.error("Dataset CSV was not found.")

    st.write("Put your dataset in either location:")

    st.code(
        str(BASE_DIR / "data" / DATA_FILENAME)
        + "\n"
        + str(BASE_DIR / DATA_FILENAME)
    )

    st.write("Your application folder:")
    st.code(str(BASE_DIR))

    st.stop()

try:
    df = pd.read_csv(data_path)

except Exception as exc:
    st.error(f"Unable to read the dataset: {exc}")
    st.stop()

if df.empty:
    st.error("The dataset is empty.")
    st.stop()

if "risk_status" not in df.columns:
    st.error(
        "The CSV must contain a column named 'risk_status'."
    )
    st.write("Available columns:", df.columns.tolist())
    st.stop()

df = df.dropna(subset=["risk_status"]).copy()

if df["risk_status"].nunique() < 2:
    st.error(
        "The risk_status column must contain at least "
        "two different classes."
    )
    st.stop()

X = df.drop(columns=DROP_COLUMNS, errors="ignore")
y = df["risk_status"]

if X.empty:
    st.error("No usable feature columns were found.")
    st.stop()


# --------------------------------------------------
# 4. TRAIN A COMPATIBLE FALLBACK MODEL
# --------------------------------------------------

def train_model(data):
    """Train a model without requiring imbalanced-learn."""

    features = data.drop(
        columns=DROP_COLUMNS,
        errors="ignore",
    )

    target = data["risk_status"]

    numeric_columns = features.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_columns = [
        column
        for column in features.columns
        if column not in numeric_columns
    ]

    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                )
            ]
        )

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_columns,
            )
        )

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    ),
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    ),
                ),
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
    )

    classifier = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    pipeline.fit(features, target)

    return pipeline, features.columns.tolist()


# --------------------------------------------------
# 5. LOAD EXISTING MODEL OR TRAIN A FALLBACK
# --------------------------------------------------

metadata = {}

metadata_path = find_file("model_metadata.json")

if metadata_path is not None:
    try:
        with metadata_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            metadata = json.load(file)

    except Exception:
        metadata = {}

model = None
model_message = ""

# Reuse the fallback model during Streamlit reruns.
if "risk_classifier_model" in st.session_state:
    model = st.session_state["risk_classifier_model"]
    model_message = st.session_state.get(
        "risk_classifier_message",
        "Fallback model trained by this application",
    )

else:
    saved_model_path = (
        find_file("Best_Model.pkl")
        or find_file("risk_alert_model.joblib")
    )

    if saved_model_path is not None:
        try:
            if saved_model_path.suffix.lower() == ".joblib":
                import joblib

                model = joblib.load(saved_model_path)

            else:
                with saved_model_path.open("rb") as file:
                    model = pickle.load(file)

            model_message = (
                f"Existing model: {saved_model_path.name}"
            )

        except Exception as exc:
            st.warning(
                "The existing model could not be loaded. "
                "A compatible replacement will be trained "
                "from your dataset. "
                f"Original error: {exc}"
            )

    # Automatically train when no saved model is usable.
    if model is None:
        try:
            with st.spinner(
                "Training a compatible model from your dataset..."
            ):
                model, feature_columns = train_model(df)

            st.session_state["risk_classifier_model"] = model
            st.session_state["risk_classifier_message"] = (
                "Random Forest fallback model"
            )

            model_message = "Random Forest fallback model"

            # Save separately; do not overwrite the original model.
            try:
                with (
                    MODEL_DIR / "Fallback_Model.pkl"
                ).open("wb") as file:
                    pickle.dump(model, file)

                with (
                    MODEL_DIR / "fallback_metadata.json"
                ).open("w", encoding="utf-8") as file:
                    json.dump(
                        {
                            "input_columns": feature_columns,
                            "threshold": 0.5,
                            "model_family": "Random Forest",
                        },
                        file,
                        indent=2,
                    )

            except Exception as save_error:
                st.info(
                    "The model works for this session but "
                    "could not be saved: "
                    f"{save_error}"
                )

            st.success("A compatible model is ready.")

        except Exception as exc:
            st.error(f"Model training failed: {exc}")
            st.info(
                "Check that scikit-learn is installed and "
                "that the dataset contains valid target classes."
            )
            st.stop()


# --------------------------------------------------
# 6. PREPARE CUSTOMER INPUTS
# --------------------------------------------------

st.caption(f"Model: {model_message}")
st.caption(f"Dataset: {data_path.name}")
st.caption(f"Dataset rows: {len(df):,}")

feature_columns = metadata.get("input_columns")

if not isinstance(feature_columns, list):
    feature_columns = X.columns.tolist()

# Use the dataset schema if old metadata is incompatible.
if (
    not feature_columns
    or any(column not in X.columns for column in feature_columns)
):
    feature_columns = X.columns.tolist()

st.subheader("Enter customer information")

st.write(
    "Values are initialized using typical dataset values. "
    "Change them as required."
)


def typical_value(series):
    """Calculate a default input value."""

    valid = series.dropna()

    if valid.empty:
        return (
            0
            if pd.api.types.is_numeric_dtype(series)
            else "Unknown"
        )

    if pd.api.types.is_numeric_dtype(series):
        median = float(valid.median())

        if pd.api.types.is_integer_dtype(series):
            return int(round(median))

        return median

    modes = valid.astype(str).mode()

    return (
        str(modes.iloc[0])
        if not modes.empty
        else "Unknown"
    )


def render_input(column, series, index):
    """Create a widget appropriate for each feature."""

    label = str(column).replace("_", " ").title()
    default = typical_value(series)
    key = f"customer_feature_{index}"

    if pd.api.types.is_bool_dtype(series):
        return st.selectbox(
            label,
            [False, True],
            index=1 if bool(default) else 0,
            key=key,
        )

    if pd.api.types.is_numeric_dtype(series):
        valid = pd.to_numeric(
            series,
            errors="coerce",
        ).dropna()

        minimum = (
            float(valid.min()) if not valid.empty else 0.0
        )

        maximum = (
            float(valid.max()) if not valid.empty else 100.0
        )

        if minimum == maximum:
            minimum = min(minimum, 0.0)
            maximum = max(maximum, minimum + 1.0)

        default = min(max(float(default), minimum), maximum)

        if pd.api.types.is_integer_dtype(series):
            low = int(np.floor(minimum))
            high = int(np.ceil(maximum))

            return st.number_input(
                label,
                min_value=low,
                max_value=high,
                value=min(
                    max(int(round(default)), low),
                    high,
                ),
                step=1,
                key=key,
            )

        return st.number_input(
            label,
            min_value=float(minimum),
            max_value=float(maximum),
            value=float(default),
            format="%.4f",
            key=key,
        )

    options = sorted(
        series.dropna().astype(str).unique().tolist()
    )

    if not options:
        options = ["Unknown"]

    default = str(default)

    if default not in options:
        options.insert(0, default)

    return st.selectbox(
        label,
        options,
        index=options.index(default),
        key=key,
    )


# --------------------------------------------------
# 7. ASSESS CUSTOMER RISK
# --------------------------------------------------

threshold = float(metadata.get("threshold", 0.5))

if not np.isfinite(threshold):
    threshold = 0.5

threshold = min(max(threshold, 0.10), 0.90)

with st.form("customer_risk_form"):
    customer_values = {}
    columns = st.columns(3)

    for index, column in enumerate(feature_columns):
        with columns[index % 3]:
            customer_values[column] = render_input(
                column,
                X[column],
                index,
            )

    threshold = st.slider(
        "High-risk probability threshold",
        min_value=0.10,
        max_value=0.90,
        value=threshold,
        step=0.05,
        help=(
            "Lower thresholds may flag more customers "
            "and increase false positives."
        ),
    )

    submitted = st.form_submit_button(
        "Assess Risk",
        type="primary",
    )


# --------------------------------------------------
# 8. SHOW PREDICTION
# --------------------------------------------------

if submitted:
    try:
        input_df = pd.DataFrame(
            [customer_values],
            columns=feature_columns,
        )

        for column in feature_columns:
            if pd.api.types.is_numeric_dtype(X[column]):
                input_df[column] = pd.to_numeric(
                    input_df[column],
                    errors="coerce",
                )

        if not hasattr(model, "predict_proba"):
            st.error(
                "The saved model does not support probability "
                "predictions. Rename the incompatible saved "
                "model in the model folder and restart the app."
            )
            st.stop()

        probabilities = model.predict_proba(input_df)

        classes = list(
            getattr(model, "classes_", [])
        )

        if not classes and hasattr(model, "named_steps"):
            estimator = model.named_steps.get(
                "classifier",
                model.named_steps.get("model"),
            )

            if estimator is not None:
                classes = list(
                    getattr(estimator, "classes_", [])
                )

        # Identify the high-risk class by its label.
        positive_index = None

        for index, value in enumerate(classes):
            normalized = (
                str(value)
                .strip()
                .lower()
                .replace("_", " ")
                .replace("-", " ")
            )

            if normalized in {
                "1",
                "high risk",
                "high",
                "risky",
                "yes",
                "true",
            }:
                positive_index = index
                break

        # For binary numeric 0/1 targets, class 1 is positive.
        if positive_index is None:
            positive_index = (
                1 if probabilities.shape[1] > 1 else 0
            )

        probability = float(
            probabilities[0, positive_index]
        )

        probability = min(
            max(probability, 0.0),
            1.0,
        )

        first, second = st.columns(2)

        first.metric(
            "Estimated high-risk probability",
            f"{probability:.1%}",
        )

        second.metric(
            "Classification threshold",
            f"{threshold:.0%}",
        )

        if probability >= threshold:
            st.error(
                "High Risk Alert — review this customer."
            )
        else:
            st.success(
                "Low Risk prediction — continue routine monitoring."
            )

        st.progress(probability)

        st.info(
            "This probability is a model estimate, not a guarantee. "
            "Review the evidence and follow your institution's "
            "approved decision process."
        )

    except Exception as exc:
        st.error(f"Could not assess this customer: {exc}")

        st.info(
            "The saved model may use a different feature schema "
            "or package version. Rename the old saved model and "
            "restart the app to train a compatible fallback."
        )


# --------------------------------------------------
# 9. DATASET DETAILS
# --------------------------------------------------

with st.expander("Dataset information"):
    st.write("Rows:", len(df))
    st.write("Columns:", len(df.columns))
    st.write(
        "Target classes:",
        [str(value) for value in y.unique()],
    )
    st.write("Features used:", feature_columns)
