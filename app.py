from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="FraudGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"


def find_artifact(filename):
    candidates = [
        BASE_DIR / filename,
        MODEL_DIR / filename,
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Could not find '{filename}' beside app.py or inside models/."
    )


@st.cache_resource
def load_artifacts():
    return (
        joblib.load(find_artifact("best_xgb_model.pkl")),
        joblib.load(find_artifact("label_encoders.pkl")),
    )


try:
    model, label_encoders = load_artifacts()
except Exception as error:
    st.error("Model files could not be loaded.")
    st.code(str(error))
    st.stop()


st.markdown(
    """
    <style>
    .stApp {
        background: #f4f7fb;
        color: #172033;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    .header {
        background: linear-gradient(135deg, #ffffff, #edf3ff);
        border: 1px solid #dce4f0;
        border-radius: 22px;
        padding: 1.5rem 1.7rem;
        box-shadow: 0 12px 35px rgba(35, 58, 98, 0.08);
        margin-bottom: 1rem;
    }

    .title {
        font-size: 2.4rem;
        font-weight: 850;
        color: #193564;
    }

    .subtitle {
        color: #65748d;
        font-size: 1rem;
    }

    .panel {
        background: white;
        border: 1px solid #e1e7f0;
        border-radius: 20px;
        padding: 1.3rem;
        box-shadow: 0 10px 28px rgba(23, 50, 95, 0.06);
    }

    .risk-low, .risk-medium, .risk-high {
        border-radius: 22px;
        padding: 1.5rem;
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

    .score {
        font-size: 3.2rem;
        font-weight: 850;
        line-height: 1;
        color: #17325f;
        margin: 0.4rem 0;
    }

    .stButton > button {
        width: 100%;
        min-height: 3rem;
        border-radius: 12px;
        border: none;
        background: #1f4f9a;
        color: white;
        font-weight: 750;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="header">
        <div class="title">🛡️ FraudGuard AI</div>
        <div class="subtitle">
            Professional bank-account application risk assessment using XGBoost.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Model", "XGBoost")
m2.metric("ROC-AUC", "0.893")
m3.metric("Recall", "78.9%")
m4.metric("Dataset", "1M applications")

form_col, result_col = st.columns([1.55, 1], gap="large")

with form_col:
    st.subheader("Application profile")

    with st.form("fraud_form"):
        left, right = st.columns(2)

        with left:
            income = st.slider("Income", 0.1, 0.9, 0.6, 0.1)

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
                -170,
                389,
                122,
            )

            proposed_credit_limit = st.selectbox(
                "Proposed Credit Limit",
                [200, 500, 1000, 1500, 2000],
            )

        with right:
            name_email_similarity = st.slider(
                "Name & Email Similarity",
                0.0,
                1.0,
                0.5,
            )

            current_address_months_count = st.slider(
                "Current Address (Months)",
                -1,
                428,
                52,
            )

            prev_address_months_count = st.slider(
                "Previous Address (Months)",
                -1,
                383,
                -1,
            )

            device_os = st.selectbox(
                "Device Operating System",
                ["linux", "windows", "macintosh", "x11", "other"],
            )

            session_length_in_minutes = st.slider(
                "Session Length (Minutes)",
                0.0,
                86.0,
                5.0,
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

        submitted = st.form_submit_button("Assess fraud risk")


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

with result_col:
    st.subheader("Risk assessment")

    if not submitted:
        st.markdown(
            """
            <div class="panel">
                <h3>Waiting for assessment</h3>
                <p>
                    Complete the application form and click
                    <b>Assess fraud risk</b>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        st.markdown(
            """
            <div class="panel">
                <h4>Risk thresholds</h4>
                <p><b>Low:</b> below 50%</p>
                <p><b>Medium:</b> 50% to 79.99%</p>
                <p><b>High:</b> 80% or above</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        categorical_cols = [
            "payment_type",
            "employment_status",
            "housing_status",
            "source",
            "device_os",
        ]

        encoded = input_data.copy()

        try:
            for col in categorical_cols:
                encoder = label_encoders[col]
                value = encoded.at[0, col]

                if value not in encoder.classes_:
                    raise ValueError(
                        f"Unknown value '{value}' for '{col}'."
                    )

                encoded[col] = encoder.transform(encoded[col])

            if hasattr(model, "feature_names_in_"):
                encoded = encoded[list(model.feature_names_in_)]

            probability = float(model.predict_proba(encoded)[0][1])

            if probability >= 0.80:
                css_class = "risk-high"
                label = "HIGH RISK"
                action = (
                    "Temporarily block the application, verify identity, "
                    "and send it for manual review."
                )

            elif probability >= 0.50:
                css_class = "risk-medium"
                label = "MEDIUM RISK"
                action = (
                    "Request additional customer verification before approval."
                )

            else:
                css_class = "risk-low"
                label = "LOW RISK"
                action = (
                    "The application appears legitimate. "
                    "No additional verification is required."
                )

            st.markdown(
                f"""
                <div class="{css_class}">
                    <div>Fraud probability</div>
                    <div class="score">{probability * 100:.2f}%</div>
                    <h3>{label}</h3>
                    <p>{action}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write("")
            st.progress(min(max(probability, 0.0), 1.0))

            with st.expander("Review encoded application"):
                st.dataframe(encoded, use_container_width=True)

        except Exception as error:
            st.error("Prediction failed.")
            st.code(str(error))

st.divider()

with st.expander("Project details"):
    st.markdown(
        """
- Exploratory Data Analysis
- Feature Engineering
- Class-Imbalance Handling
- Logistic Regression
- Random Forest
- Random Forest with SMOTE
- Hyperparameter Tuning
- XGBoost Final Model
"""
    )

st.caption("© 2026 Nourallah Ghonim ")