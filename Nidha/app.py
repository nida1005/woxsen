import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load data for visualization
@st.cache_data
def load_data():
    return pd.read_csv("loan_approval_dataset.csv")

# Load the trained model
@st.cache_resource
def load_model():
    with open("gb_model.pkl", "rb") as f:
        return pickle.load(f)

# Login authentication
def login(username, password):
    return username == "admin" and password == "admin"

# Main Streamlit application
def main():
    st.set_page_config("Loan Approval Dashboard", layout="wide")
    st.title("🏦 Loan Approval Prediction Dashboard")

    # Login session state
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        with st.form("login_form"):
            st.subheader("🔐 Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_btn = st.form_submit_button("Login")
            if login_btn:
                if login(username, password):
                    st.session_state.authenticated = True
                    st.success("✅ Logged in successfully!")
                else:
                    st.error("❌ Invalid username or password")

    # Dashboard pages
    if st.session_state.authenticated:
        st.sidebar.header("Navigation")
        page = st.sidebar.radio("Select Page", ["📊 Visualizations", "📈 Predict Loan Status"])

        if page == "📊 Visualizations":
            df = load_data()
            df.columns = df.columns.str.strip()  # Strip column names just in case

            st.subheader("📊 Data Visualizations")

            st.write("### Loan Status Count")
            fig1, ax1 = plt.subplots()
            sns.countplot(data=df, x="loan_status", palette="Set2", ax=ax1)
            st.pyplot(fig1)

            st.write("### Income vs Loan Amount")
            fig2, ax2 = plt.subplots()
            sns.scatterplot(data=df, x="income_annum", y="loan_amount", hue="loan_status", ax=ax2)
            st.pyplot(fig2)

            st.write("### CIBIL Score Distribution")
            fig3, ax3 = plt.subplots()
            sns.histplot(df["cibil_score"], kde=True, color="skyblue", ax=ax3)
            st.pyplot(fig3)

        elif page == "📈 Predict Loan Status":
            st.subheader("📈 Predict Loan Approval Outcome")

            with st.form("prediction_form"):
                no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, step=1)
                education = st.selectbox("Education", ["Graduate", "Not Graduate"])
                self_employed = st.selectbox("Self Employed", ["Yes", "No"])
                income_annum = st.number_input("Annual Income", min_value=0)
                loan_amount = st.number_input("Loan Amount", min_value=0)
                loan_term = st.number_input("Loan Term (Months)", min_value=1)
                cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900)
                residential_assets_value = st.number_input("Residential Asset Value", min_value=0)
                commercial_assets_value = st.number_input("Commercial Asset Value", min_value=0)
                luxury_assets_value = st.number_input("Luxury Asset Value", min_value=0)
                bank_asset_value = st.number_input("Bank Asset Value", min_value=0)

                submit_btn = st.form_submit_button("Predict")

                if submit_btn:
                    # Raw input dictionary with potential spaces
                    raw_input = {
                        "no_of_dependents":no_of_dependents,
                        "education":education,
                        "self_employed":self_employed,
                        "income_annum":income_annum,
                        "loan_amount":loan_amount,
                        "loan_term":loan_term,
                        "cibil_score":cibil_score,
                        "residential_assets_value":residential_assets_value,
                        "commercial_assets_value":commercial_assets_value,
                        "luxury_assets_value":luxury_assets_value,
                        "bank_asset_value":bank_asset_value
                    }

                    # Strip keys to match model training
                    clean_input = {k.strip(): v for k, v in raw_input.items()}
                    input_df = pd.DataFrame([clean_input])

                    # Load model and make prediction
                    model = load_model()
                    prediction = model.predict(input_df)[0]

                    st.success(f"🎯 Prediction Result: **{prediction}**")

if __name__ == "__main__":
    main()
