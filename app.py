
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


# =========================================================
# 1. APP CONFIGURATION
# =========================================================

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

# Automatically detect the folder containing app.py.
BASE_DIR = Path(__file__).resolve().parent

DATA_FILENAME = "Risk_Alert_Classifier_Dataset_4600.csv"

MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "Risk_Alert_Fallback_Model.pkl"
METADATA_PATH = MODEL_DIR / "risk_alert_fallback_metadata.json"

DROP_COLUMNS = [
    "customer_id",
    "last_transaction_date",
    "risk_status",
]


# =========================================================
# 2. FIND DATASET
# =========================================================

def find_dataset():
    """Search the project directory and its subdirectories."""

    direct_path = BASE_DIR / DATA_FILENAME

    if direct_path.is_file():
        return direct_path

    ignored = {
        ".git",
        ".venv",
        "venv",
        "env",
        "__pycache__",
    }

    for path in BASE_DIR.rglob(DATA_FILENAME):
        relative_parts = path.relative_to(BASE_DIR).parts

        if any(
            part in ignored or part.startswith(".")
            for part in relative_parts
        ):
            continue

        if path.is_file():
            return path

    return None


data_path = find_dataset()

if data_path is None:
    st.error("Dataset CSV not found.")

    st.write("Place your CSV in this folder:")

    st.code(
        str(BASE_DIR / "data" / DATA_FILENAME)
    )

    st.write("Or place it directly beside app.py:")

    st.code(
        str(BASE_DIR / DATA_FILENAME)
    )

    st.stop()


# =========================================================
# 3. LOAD DATA
# =========================================================

try:
    df = pd.read_csv(data_path)

except Exception:
    st.error(
        "Unable to read the dataset. Check that the CSV "
        "file is valid and is not open in another program."
    )
    st.stop()


if df.empty:
    st.error("The dataset contains no records.")
    st.stop()


if "risk_status" not in df.columns:
    st.error(
        "The dataset must contain a 'risk_status' column."
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


X = df.drop(
    columns=DROP_COLUMNS,
    errors="ignore",
)

y = df["risk_status"]

if X.empty:
    st.error("No usable input features were found.")
    st.stop()


# =========================================================
# 4. TRAIN A NEW COMPATIBLE MODEL
# =========================================================

def train_model(data):
    """
    Train a fresh model using scikit-learn.

    This deliberately does not load the old saved model,
    avoiding the incompatible _RemainderColsList object.
    """

    X_train = data.drop(
        columns=DROP_COLUMNS,
        errors="ignore",
    )

    y_train = data["risk_status"]

    numeric_columns = X_train.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_columns = [
        column
        for column in X_train.columns
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

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    model.fit(X_train, y_train)

    return model, X_train.columns.tolist()


# =========================================================
# 5. TRAIN ONLY ONCE PER SESSION
# =========================================================

if "risk_alert_model" not in st.session_state:

    try:
        with st.spinner(
            "Preparing the risk classification model..."
        ):
            model, feature_columns = train_model(df)

        st.session_state["risk_alert_model"] = model

        st.session_state["risk_alert_features"] = (
            feature_columns
        )

        # Save the newly trained model separately.
        # The old Best_Model.pkl is never loaded or overwritten.
        try:
            with MODEL_PATH.open("wb") as file:
                pickle.dump(model, file)

            metadata = {
                "input_columns": feature_columns,
                "threshold": 0.50,
                "model_family": "Random Forest",
            }

            with METADATA_PATH.open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    metadata,
                    file,
                    indent=2,
                )

        except OSError:
            # The model can still be used during this session.
            pass

    except Exception:
        st.error(
            "The model could not be trained. Check that the "
            "CSV contains valid feature columns and multiple "
            "risk_status classes."
        )
        st.stop()


model = st.session_state["risk_alert_model"]

feature_columns = st.session_state["risk_alert_features"]


# =========================================================
# 6. APPLICATION INFORMATION
# =========================================================

st.success("Risk classification model is ready.")

st.caption(
    "Model: Random Forest Classifier"
)

st.caption(
    f"Dataset: {data_path.name} | "
    f"Records: {len(df):,}"
)

st.subheader("Enter Customer Information")

st.write(
    "The input fields use typical values from the dataset. "
    "Change the values and click Assess Risk."
)


# =========================================================
# 7. INPUT FIELD HELPERS
# =========================================================

def typical_value(series):
    """Find a reasonable default for a feature."""

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

    if not modes.empty:
        return str(modes.iloc[0])

    return "Unknown"


def create_input(column, series, index):
    """Create a Streamlit input based on the feature type."""

    label = column.replace("_", " ").title()

    default = typical_value(series)

    key = f"customer_input_{index}"

    # Boolean feature
    if pd.api.types.is_bool_dtype(series):
        return st.selectbox(
            label,
            options=[False, True],
            index=1 if bool(default) else 0,
            key=key,
        )

    # Numeric feature
    if pd.api.types.is_numeric_dtype(series):

        numeric_values = pd.to_numeric(
            series,
            errors="coerce",
        ).dropna()

        minimum = (
            float(numeric_values.min())
            if not numeric_values.empty
            else 0.0
        )

        maximum = (
            float(numeric_values.max())
            if not numeric_values.empty
            else 100.0
        )

        if minimum == maximum:
            minimum = min(minimum, 0.0)
            maximum = max(maximum, minimum + 1.0)

        default = min(
            max(float(default), minimum),
            maximum,
        )

        # Integer feature
        if pd.api.types.is_integer_dtype(series):

            min_int = int(np.floor(minimum))
            max_int = int(np.ceil(maximum))

            return st.number_input(
                label,
                min_value=min_int,
                max_value=max_int,
                value=min(
                    max(int(round(default)), min_int),
                    max_int,
                ),
                step=1,
                key=key,
            )

        # Decimal feature
        return st.number_input(
            label,
            min_value=float(minimum),
            max_value=float(maximum),
            value=float(default),
            format="%.4f",
            key=key,
        )

    # Categorical feature
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
        options=options,
        index=options.index(default),
        key=key,
    )


# =========================================================
# 8. CUSTOMER RISK FORM
# =========================================================

with st.form("customer_risk_form"):

    customer_values = {}

    columns = st.columns(3)

    for index, column in enumerate(feature_columns):

        with columns[index % 3]:

            customer_values[column] = create_input(
                column,
                X[column],
                index,
            )

    threshold = st.slider(
        "High-risk probability threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help=(
            "Lower thresholds may flag more customers "
            "and may increase false positives."
        ),
    )

    submitted = st.form_submit_button(
        "Assess Risk",
        type="primary",
    )


# =========================================================
# 9. PREDICT RISK
# =========================================================

if submitted:

    try:
        input_df = pd.DataFrame(
            [customer_values],
            columns=feature_columns,
        )

        # Ensure numerical fields have the expected data types.
        for column in feature_columns:

            if pd.api.types.is_numeric_dtype(X[column]):

                input_df[column] = pd.to_numeric(
                    input_df[column],
                    errors="coerce",
                )

        probabilities = model.predict_proba(input_df)

        classes = list(model.classes_)

        # For numeric 0/1 targets, class 1 is treated as high risk.
        # Also recognize common text labels.
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

        if positive_index is None:

            positive_index = (
                1 if len(classes) > 1 else 0
            )

        probability = float(
            probabilities[0, positive_index]
        )

        probability = min(
            max(probability, 0.0),
            1.0,
        )

        # Display results
        col1, col2 = st.columns(2)

        col1.metric(
            "Estimated High-Risk Probability",
            f"{probability:.1%}",
        )

        col2.metric(
            "Classification Threshold",
            f"{threshold:.0%}",
        )

        if probability >= threshold:

            st.error(
                "High Risk Alert — review this customer."
            )

        else:

            st.success(
                "Low Risk Prediction — continue routine monitoring."
            )

        st.progress(probability)

        st.info(
            "This probability is a model estimate, not a guarantee. "
            "Review supporting evidence and follow the approved "
            "decision-making process."
        )

    except Exception:
        st.error(
            "Prediction failed. Check the dataset feature types "
            "and restart the application."
        )


# =========================================================
# 10. DATASET DETAILS
# =========================================================

with st.expander("Dataset Information"):

    st.write("Total records:", len(df))

    st.write("Total columns:", len(df.columns))

    st.write(
        "Risk classes:",
        [str(value) for value in y.unique()],
    )

    st.write(
        "Features used:",
        feature_columns,
    )