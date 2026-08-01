from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# =========================================================
# Page Configuration
# =========================================================

st.set_page_config(
    page_title="Bank Account Fraud Detection",
    page_icon="🏦",
    layout="wide",
)


# =========================================================
# Paths and Artifact Loading
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

# The app will look in both:
# 1) the same folder as app.py
# 2) a models/ folder beside app.py
MODEL_DIR = BASE_DIR / "models"


def find_artifact(filename: str) -> Path:
    """Return the first existing artifact path."""
    candidates = [
        BASE_DIR / filename,
        MODEL_DIR / filename,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    checked_paths = "\n".join(f"- {path}" for path in candidates)

    raise FileNotFoundError(
        f"Could not find '{filename}'. Checked:\n{checked_paths}"
    )


@st.cache_resource
def load_artifacts():
    model_path = find_artifact("best_xgb_model.pkl")
    encoders_path = find_artifact("label_encoders.pkl")

    loaded_model = joblib.load(model_path)
    loaded_encoders = joblib.load(encoders_path)

    return loaded_model, loaded_encoders, model_path, encoders_path


try:
    model, label_encoders, model_path, encoders_path = load_artifacts()
except Exception as error:
    st.error("The trained model files could not be loaded.")
    st.code(str(error))
    st.markdown(
        """
        Place these files either beside `app.py` or inside a `models` folder:

        ```text
        best_xgb_model.pkl
        label_encoders.pkl
        ```
        """
    )
    st.stop()


# =========================================================
# Sidebar
# =========================================================

st.sidebar.title("🏦 Fraud Detection")

st.sidebar.markdown(
    """
### Model
**XGBoost Classifier**

### Dataset
NeurIPS 2022 Bank Account Fraud Dataset

- 1,000,000 applications
- Highly imbalanced dataset

### Best Performance

- ROC-AUC: **0.893**
- Recall: **78.9%**

---

### Author

**Nourallah Ghonim**
"""
)

with st.sidebar.expander("Loaded files"):
    st.write("Model:", model_path.name)
    st.write("Encoders:", encoders_path.name)


# =========================================================
# Header
# =========================================================

st.title("🏦 Bank Account Fraud Detection")

st.markdown(
    """
This application predicts whether a **bank account application** is fraudulent
using a Machine Learning model trained on the **NeurIPS 2022 Bank Account Fraud Dataset**.

The deployed model is **XGBoost**, selected after comparing Logistic Regression,
Random Forest, SMOTE-based Random Forest, Tuned Random Forest, and XGBoost.

Fill in the customer information below, then click **Predict Fraud**.
"""
)

st.divider()


# =========================================================
# User Inputs
# =========================================================

st.header("Customer Information")

col1, col2 = st.columns(2)

with col1:
    income = st.slider(
        "Income",
        min_value=0.1,
        max_value=0.9,
        value=0.6,
        step=0.1,
    )

    customer_age = st.selectbox(
        "Customer Age",
        [10, 20, 30, 40, 50, 60, 70, 80, 90],
        index=3,
    )

    payment_type = st.selectbox(
        "Payment Type",
        ["AA", "AB", "AC", "AD", "AE"],
    )

    employment_status = st.selectbox(
        "Employment Status",
        ["CA", "CB", "CC", "CD", "CE", "CF", "CG"],
    )

    housing_status = st.selectbox(
        "Housing Status",
        ["BA", "BB", "BC", "BD", "BE", "BF", "BG"],
    )

    credit_risk_score = st.slider(
        "Credit Risk Score",
        min_value=-170,
        max_value=389,
        value=122,
    )

    proposed_credit_limit = st.selectbox(
        "Proposed Credit Limit",
        [200, 500, 1000, 1500, 2000],
    )

    name_email_similarity = st.slider(
        "Name & Email Similarity",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
    )

with col2:
    current_address_months_count = st.slider(
        "Current Address (Months)",
        min_value=-1,
        max_value=428,
        value=52,
    )

    prev_address_months_count = st.slider(
        "Previous Address (Months)",
        min_value=-1,
        max_value=383,
        value=-1,
    )

    device_os = st.selectbox(
        "Device Operating System",
        ["linux", "windows", "macintosh", "x11", "other"],
    )

    session_length_in_minutes = st.slider(
        "Session Length (Minutes)",
        min_value=0.0,
        max_value=86.0,
        value=5.0,
    )

    phone_home_valid = st.checkbox(
        "Home Phone Valid",
        value=False,
    )

    email_is_free = st.checkbox(
        "Free Email Provider",
        value=True,
    )

    keep_alive_session = st.checkbox(
        "Keep Alive Session",
        value=True,
    )

    foreign_request = st.checkbox(
        "Foreign Request",
        value=False,
    )


# =========================================================
# Input Data
# =========================================================

input_data = pd.DataFrame(
    {
        "income": [income],
        "name_email_similarity": [name_email_similarity],
        "prev_address_months_count": [prev_address_months_count],
        "current_address_months_count": [current_address_months_count],
        "customer_age": [customer_age],
        "days_since_request": [0.015],
        "intended_balcon_amount": [-0.83],
        "payment_type": [payment_type],
        "zip_count_4w": [1263],
        "velocity_6h": [5319],
        "velocity_24h": [4750],
        "velocity_4w": [4913],
        "bank_branch_count_8w": [9],
        "date_of_birth_distinct_emails_4w": [9],
        "employment_status": [employment_status],
        "credit_risk_score": [credit_risk_score],
        "email_is_free": [int(email_is_free)],
        "housing_status": [housing_status],
        "phone_home_valid": [int(phone_home_valid)],
        "phone_mobile_valid": [1],
        "bank_months_count": [5],
        "has_other_cards": [0],
        "proposed_credit_limit": [proposed_credit_limit],
        "foreign_request": [int(foreign_request)],
        "source": ["INTERNET"],
        "session_length_in_minutes": [session_length_in_minutes],
        "device_os": [device_os],
        "keep_alive_session": [int(keep_alive_session)],
        "device_distinct_emails_8w": [1],
        "device_fraud_count": [0],
        "month": [3],
    }
)


# =========================================================
# Input Summary
# =========================================================

st.divider()
st.subheader("Input Summary")
st.dataframe(input_data, use_container_width=True)


# =========================================================
# Prediction
# =========================================================

if st.button("🔍 Predict Fraud", use_container_width=True):
    categorical_cols = [
        "payment_type",
        "employment_status",
        "housing_status",
        "source",
        "device_os",
    ]

    input_encoded = input_data.copy()

    try:
        for column in categorical_cols:
            if column not in label_encoders:
                raise KeyError(
                    f"Encoder for '{column}' is missing from label_encoders.pkl"
                )

            encoder = label_encoders[column]
            raw_value = input_encoded.at[0, column]

            if raw_value not in encoder.classes_:
                raise ValueError(
                    f"Unknown value '{raw_value}' for '{column}'. "
                    f"Allowed values: {list(encoder.classes_)}"
                )

            input_encoded[column] = encoder.transform(
                input_encoded[column]
            )

        # Match the feature order expected by the trained model when available.
        if hasattr(model, "feature_names_in_"):
            expected_columns = list(model.feature_names_in_)

            missing_columns = [
                col for col in expected_columns
                if col not in input_encoded.columns
            ]

            if missing_columns:
                raise ValueError(
                    "Missing model input columns: "
                    + ", ".join(missing_columns)
                )

            input_encoded = input_encoded[expected_columns]

        probability = float(
            model.predict_proba(input_encoded)[0][1]
        )

        st.divider()
        st.header("Prediction Result")

        st.metric(
            "Fraud Probability",
            f"{probability * 100:.2f}%",
        )

        st.progress(min(max(probability, 0.0), 1.0))

        if probability >= 0.80:
            st.error("🔴 HIGH RISK FRAUD")
            st.markdown(
                """
### Recommended Action

- Block the transaction temporarily.
- Verify the customer's identity.
- Perform manual review before approval.
"""
            )

        elif probability >= 0.50:
            st.warning("🟠 MEDIUM RISK")
            st.markdown(
                """
### Recommended Action

- Perform additional customer verification before approval.
"""
            )

        else:
            st.success("🟢 LOW RISK")
            st.markdown(
                """
### Recommended Action

- Transaction appears legitimate.
- No additional verification is required.
"""
            )

    except Exception as error:
        st.error("Prediction failed.")
        st.code(str(error))


# =========================================================
# About
# =========================================================

st.divider()

with st.expander("About this Project"):
    st.markdown(
        """
### Project Workflow

- Exploratory Data Analysis (EDA)
- Feature Engineering
- Handling Class Imbalance
- Logistic Regression
- Random Forest
- Random Forest + SMOTE
- Hyperparameter Tuning
- XGBoost (Final Model)

### Final Model Performance

- Accuracy: **83.7%**
- Recall: **78.9%**
- ROC-AUC: **0.893**

This project was developed as an end-to-end Machine Learning
portfolio project for fraud detection.
"""
    )

st.divider()

st.caption(
    "© 2026 Nourallah Ghonim | "
    "Bank Account Fraud Detection using Machine Learning"
)
