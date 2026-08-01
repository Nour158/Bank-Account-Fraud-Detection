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
    initial_sidebar_state="collapsed",
)


# =========================================================
# Paths and model loading
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"


def find_artifact(filename: str) -> Path:
    candidates = [
        BASE_DIR / filename,
        MODEL_DIR / filename,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not find '{filename}' beside app.py or inside models/."
    )


@st.cache_resource
def load_artifacts():
    model = joblib.load(find_artifact("best_xgb_model.pkl"))
    encoders = joblib.load(find_artifact("label_encoders.pkl"))
    return model, encoders


try:
    model, label_encoders = load_artifacts()
except Exception as error:
    st.error("The trained model files could not be loaded.")
    st.code(str(error))
    st.stop()


# =========================================================
# Styling
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background: #f5f7fb;
        color: #162033;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    .hero {
        background: linear-gradient(135deg, #112b52, #1e4c8f);
        color: white;
        border-radius: 24px;
        padding: 1.6rem 1.8rem;
        box-shadow: 0 16px 40px rgba(20, 48, 94, 0.18);
        margin-bottom: 1rem;
    }

    .hero-title {
        font-size: 2.5rem;
        font-weight: 850;
        margin-bottom: 0.25rem;
    }

    .hero-subtitle {
        color: #d9e6fa;
        font-size: 1rem;
    }

    .info-card {
        background: white;
        border: 1px solid #e1e7f0;
        border-radius: 18px;
        padding: 1rem 1.1rem;
        box-shadow: 0 8px 24px rgba(28, 52, 95, 0.06);
        height: 100%;
    }

    .info-title {
        color: #6b7890;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 750;
    }

    .info-value {
        color: #17325f;
        font-size: 1.45rem;
        font-weight: 800;
        margin-top: 0.25rem;
    }

    .category-card {
        background: white;
        border: 1px solid #e1e7f0;
        border-radius: 20px;
        padding: 1rem 1.15rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 24px rgba(28, 52, 95, 0.05);
    }

    .category-title {
        color: #193564;
        font-size: 1.1rem;
        font-weight: 800;
        margin-bottom: 0.8rem;
        padding-bottom: 0.45rem;
        border-bottom: 2px solid #e7edf6;
    }

    .waiting-card {
        background: white;
        border: 1px solid #e1e7f0;
        border-radius: 20px;
        padding: 1.4rem;
        box-shadow: 0 10px 28px rgba(23, 50, 95, 0.06);
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
        font-size: 3.3rem;
        font-weight: 850;
        line-height: 1;
        color: #17325f;
        margin: 0.4rem 0 0.75rem;
    }

    .risk-title {
        font-size: 1.55rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
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
# Header
# =========================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">🛡️ FraudGuard AI</div>
        <div class="hero-subtitle">
            Bank-account application risk assessment powered by an XGBoost classifier.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Top information row
# =========================================================

i1, i2, i3, i4, i5 = st.columns(5)

info_items = [
    ("Model", "XGBoost"),
    ("Dataset", "NeurIPS 2022"),
    ("ROC-AUC", "0.893"),
    ("Recall", "78.9%"),
    ("Accuracy", "83.7%"),
]

for column, (title, value) in zip([i1, i2, i3, i4, i5], info_items):
    with column:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="info-title">{title}</div>
                <div class="info-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.write("")


# =========================================================
# Main layout
# =========================================================

form_col, result_col = st.columns([1.65, 1], gap="large")

with form_col:
    st.subheader("Applicant Information")

    with st.form("fraud_form"):

        st.markdown(
            '<div class="category-title">👤 Customer Profile</div>',
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
            '<div class="category-title">📍 Address History</div>',
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
            '<div class="category-title">💳 Financial Details</div>',
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
            '<div class="category-title">💻 Device and Session</div>',
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
            '<div class="category-title">⚙️ Verification Signals</div>',
            unsafe_allow_html=True,
        )

        v1, v2 = st.columns(2)

        with v1:
            phone_home_valid = st.checkbox(
                "Home Phone Valid",
                value=False,
            )

            email_is_free = st.checkbox(
                "Free Email Provider",
                value=True,
            )

        with v2:
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
# Result panel
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
                <h4>Risk Thresholds</h4>
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
                encoder = label_encoders[column]
                value = encoded_input.at[0, column]

                if value not in encoder.classes_:
                    raise ValueError(
                        f"Unknown value '{value}' for '{column}'."
                    )

                encoded_input[column] = encoder.transform(
                    encoded_input[column]
                )

            if hasattr(model, "feature_names_in_"):
                encoded_input = encoded_input[
                    list(model.feature_names_in_)
                ]

            probability = float(
                model.predict_proba(encoded_input)[0][1]
            )

            if probability >= 0.80:
                css_class = "risk-high"
                icon = "🔴"
                label = "HIGH RISK"
                action = (
                    "Temporarily block the application, verify identity, "
                    "and perform a manual review."
                )

            elif probability >= 0.50:
                css_class = "risk-medium"
                icon = "🟠"
                label = "MEDIUM RISK"
                action = (
                    "Request additional applicant verification before approval."
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
                    <div>Fraud Probability</div>
                    <div class="risk-score">{probability * 100:.2f}%</div>
                    <div class="risk-title">{icon} {label}</div>
                    <p>{action}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write("")
            st.progress(min(max(probability, 0.0), 1.0))

            st.markdown("### Recommended Action")
            st.write(action)

            with st.expander("View raw application input"):
                st.dataframe(input_data, use_container_width=True)

        except Exception as error:
            st.error("Prediction failed.")
            st.code(str(error))


# =========================================================
# Footer
# =========================================================

st.divider()

with st.expander("Project Details"):
    st.markdown(
        """
### Workflow

- Exploratory Data Analysis
- Feature Engineering
- Class-Imbalance Handling
- Logistic Regression
- Random Forest
- Random Forest with SMOTE
- Hyperparameter Tuning
- XGBoost Final Model

### Final Performance

- Accuracy: **83.7%**
- Recall: **78.9%**
- ROC-AUC: **0.893**
"""
    )

st.caption("© 2026 Nourallah Ghonim | FraudGuard AI")