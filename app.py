from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# =========================================================
# Page configuration
# =========================================================

st.set_page_config(
    page_title="FraudGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# Paths and artifact loading
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"


def find_artifact(filename: str) -> Path:
    """
    Search for an artifact beside app.py and inside models/.
    """
    candidates = [
        BASE_DIR / filename,
        MODEL_DIR / filename,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    checked = "\n".join(f"- {path}" for path in candidates)

    raise FileNotFoundError(
        f"Could not find '{filename}'. Checked:\n{checked}"
    )


@st.cache_resource
def load_artifacts():
    model_path = find_artifact("best_xgb_model.pkl")
    encoders_path = find_artifact("label_encoders.pkl")

    loaded_model = joblib.load(model_path)
    loaded_encoders = joblib.load(encoders_path)

    return loaded_model, loaded_encoders


try:
    model, label_encoders = load_artifacts()
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
# Styling
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background: #f4f7fb;
        color: #172033;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1.25rem;
        padding-bottom: 2rem;
    }

    [data-testid="stSidebar"] {
        background: #102344;
        color: white;
    }

    [data-testid="stSidebar"] * {
        color: white;
    }

    .hero {
        background: linear-gradient(135deg, #ffffff, #edf3ff);
        border: 1px solid #dce4f0;
        border-radius: 22px;
        padding: 1.5rem 1.7rem;
        box-shadow: 0 12px 35px rgba(35, 58, 98, 0.08);
        margin-bottom: 1rem;
    }

    .hero-title {
        font-size: 2.45rem;
        font-weight: 850;
        color: #193564;
        margin-bottom: 0.2rem;
    }

    .hero-subtitle {
        color: #65748d;
        font-size: 1rem;
    }

    .section-card {
        background: white;
        border: 1px solid #e1e7f0;
        border-radius: 20px;
        padding: 1.15rem 1.2rem;
        box-shadow: 0 10px 28px rgba(23, 50, 95, 0.06);
        margin-bottom: 1rem;
    }

    .section-label {
        font-size: 0.82rem;
        font-weight: 750;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #66748b;
        margin-bottom: 0.5rem;
    }

    .risk-card {
        border-radius: 22px;
        padding: 1.5rem;
        box-shadow: 0 10px 28px rgba(23, 50, 95, 0.06);
    }

    .risk-low {
        background: #eaf8ef;
        border: 1px solid #b7e3c5;
    }

    .risk-medium {
        background: #fff6e8;
        border: 1px solid #f2d49f;
    }

    .risk-high {
        background: #fff0f0;
        border: 1px solid #f0bcbc;
    }

    .risk-score {
        font-size: 3.35rem;
        font-weight: 850;
        line-height: 1;
        color: #17325f;
        margin: 0.45rem 0 0.75rem;
    }

    .risk-title {
        font-size: 1.65rem;
        font-weight: 800;
        margin-bottom: 0.45rem;
    }

    .risk-action {
        color: #40506a;
        font-size: 1rem;
    }

    .waiting-card {
        background: white;
        border: 1px solid #e1e7f0;
        border-radius: 20px;
        padding: 1.4rem;
        box-shadow: 0 10px 28px rgba(23, 50, 95, 0.06);
    }

    .stButton > button {
        width: 100%;
        min-height: 3.1rem;
        border-radius: 12px;
        border: none;
        background: #1f4f9a;
        color: white;
        font-weight: 750;
        font-size: 1rem;
    }

    .stButton > button:hover {
        background: #173f7d;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Sidebar
# =========================================================

with st.sidebar:
    st.title("🏦 Fraud Detection")

    st.markdown("### Model")
    st.write("XGBoost Classifier")

    st.markdown("### Dataset")
    st.write("NeurIPS 2022 Bank Account Fraud Dataset")

    st.markdown("### Performance")
    st.metric("ROC-AUC", "0.893")
    st.metric("Recall", "78.9%")
    st.metric("Accuracy", "83.7%")

    st.divider()

    st.markdown("### Author")
    st.write("Nourallah Ghonim")


# =========================================================
# Header and KPI cards
# =========================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">🛡️ FraudGuard AI</div>
        <div class="hero-subtitle">
            Professional bank-account application risk assessment powered by XGBoost.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Model", "XGBoost")
m2.metric("ROC-AUC", "0.893")
m3.metric("Recall", "78.9%")
m4.metric("Dataset Size", "1M applications")


# =========================================================
# Main layout
# =========================================================

form_col, result_col = st.columns([1.55, 1], gap="large")

with form_col:
    st.subheader("Applicant Information")

    with st.form("fraud_assessment_form"):
        st.markdown(
            '<div class="section-label">👤 Customer Information</div>',
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2)

        with c1:
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

            employment_status = st.selectbox(
                "Employment Status",
                ["CA", "CB", "CC", "CD", "CE", "CF", "CG"],
            )

        with c2:
            housing_status = st.selectbox(
                "Housing Status",
                ["BA", "BB", "BC", "BD", "BE", "BF", "BG"],
            )

            name_email_similarity = st.slider(
                "Name & Email Similarity",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
            )

            payment_type = st.selectbox(
                "Payment Type",
                ["AA", "AB", "AC", "AD", "AE"],
            )

        st.markdown(
            '<div class="section-label">📍 Address Information</div>',
            unsafe_allow_html=True,
        )

        a1, a2 = st.columns(2)

        with a1:
            current_address_months_count = st.slider(
                "Current Address (Months)",
                min_value=-1,
                max_value=428,
                value=52,
            )

        with a2:
            prev_address_months_count = st.slider(
                "Previous Address (Months)",
                min_value=-1,
                max_value=383,
                value=-1,
            )

        st.markdown(
            '<div class="section-label">💳 Financial Information</div>',
            unsafe_allow_html=True,
        )

        f1, f2 = st.columns(2)

        with f1:
            credit_risk_score = st.slider(
                "Credit Risk Score",
                min_value=-170,
                max_value=389,
                value=122,
            )

        with f2:
            proposed_credit_limit = st.selectbox(
                "Proposed Credit Limit",
                [200, 500, 1000, 1500, 2000],
            )

        st.markdown(
            '<div class="section-label">💻 Device Information</div>',
            unsafe_allow_html=True,
        )

        d1, d2 = st.columns(2)

        with d1:
            device_os = st.selectbox(
                "Device Operating System",
                ["linux", "windows", "macintosh", "x11", "other"],
            )

        with d2:
            session_length_in_minutes = st.slider(
                "Session Length (Minutes)",
                min_value=0.0,
                max_value=86.0,
                value=5.0,
            )

        st.markdown(
            '<div class="section-label">⚙️ Additional Checks</div>',
            unsafe_allow_html=True,
        )

        b1, b2 = st.columns(2)

        with b1:
            phone_home_valid = st.checkbox(
                "Home Phone Valid",
                value=False,
            )

            email_is_free = st.checkbox(
                "Free Email Provider",
                value=True,
            )

        with b2:
            keep_alive_session = st.checkbox(
                "Keep Alive Session",
                value=True,
            )

            foreign_request = st.checkbox(
                "Foreign Request",
                value=False,
            )

        submitted = st.form_submit_button("Assess Fraud Risk")


# =========================================================
# Build model input
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
# Risk result panel
# =========================================================

with result_col:
    st.subheader("Risk Assessment")

    if not submitted:
        st.markdown(
            """
            <div class="waiting-card">
                <h3>Waiting for assessment</h3>
                <p>
                    Complete the applicant form and click
                    <b>Assess Fraud Risk</b>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        st.markdown(
            """
            <div class="waiting-card">
                <h4>Risk thresholds</h4>
                <p><b>Low risk:</b> below 50%</p>
                <p><b>Medium risk:</b> 50% to 79.99%</p>
                <p><b>High risk:</b> 80% or above</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        categorical_columns = [
            "payment_type",
            "employment_status",
            "housing_status",
            "source",
            "device_os",
        ]

        encoded_input = input_data.copy()

        try:
            for column in categorical_columns:
                if column not in label_encoders:
                    raise KeyError(
                        f"Encoder for '{column}' is missing."
                    )

                encoder = label_encoders[column]
                raw_value = encoded_input.at[0, column]

                if raw_value not in encoder.classes_:
                    raise ValueError(
                        f"Unknown value '{raw_value}' for '{column}'."
                    )

                encoded_input[column] = encoder.transform(
                    encoded_input[column]
                )

            if hasattr(model, "feature_names_in_"):
                expected_columns = list(model.feature_names_in_)

                missing_columns = [
                    column
                    for column in expected_columns
                    if column not in encoded_input.columns
                ]

                if missing_columns:
                    raise ValueError(
                        "Missing model input columns: "
                        + ", ".join(missing_columns)
                    )

                encoded_input = encoded_input[expected_columns]

            probability = float(
                model.predict_proba(encoded_input)[0][1]
            )

            percentage = probability * 100

            if probability >= 0.80:
                css_class = "risk-high"
                icon = "🔴"
                label = "HIGH RISK"
                action = (
                    "Temporarily block the application, verify the applicant's "
                    "identity, and send it for manual review."
                )

            elif probability >= 0.50:
                css_class = "risk-medium"
                icon = "🟠"
                label = "MEDIUM RISK"
                action = (
                    "Request additional customer verification before approval."
                )

            else:
                css_class = "risk-low"
                icon = "🟢"
                label = "LOW RISK"
                action = (
                    "The application appears legitimate. "
                    "No additional verification is currently required."
                )

            st.markdown(
                f"""
                <div class="risk-card {css_class}">
                    <div class="section-label">Fraud Probability</div>
                    <div class="risk-score">{percentage:.2f}%</div>
                    <div class="risk-title">{icon} {label}</div>
                    <div class="risk-action">{action}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write("")
            st.progress(min(max(probability, 0.0), 1.0))

            st.markdown("### 🛡️ Recommended Action")
            st.write(action)

            with st.expander("Review encoded application"):
                st.dataframe(
                    encoded_input,
                    use_container_width=True,
                )

            with st.expander("View raw application input"):
                st.dataframe(
                    input_data,
                    use_container_width=True,
                )

        except Exception as error:
            st.error("Prediction failed.")
            st.code(str(error))


# =========================================================
# About section
# =========================================================

st.divider()

with st.expander("About this Project"):
    st.markdown(
        """
### Project Workflow

- Exploratory Data Analysis
- Feature Engineering
- Handling Class Imbalance
- Logistic Regression
- Random Forest
- Random Forest with SMOTE
- Hyperparameter Tuning
- XGBoost Final Model

### Final Model Performance

- Accuracy: **83.7%**
- Recall: **78.9%**
- ROC-AUC: **0.893**
"""
    )

st.caption(
    "© 2026 Nourallah Ghonim | FraudGuard AI"
)