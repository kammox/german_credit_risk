import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("notebooks/credit_risk_model.pkl")

st.title("Prédiction de Risque de Crédit (German Credit)")

st.write("Remplissez les informations ci-dessous pour prédire si un client est à risque ou non.")

# Inputs
age = st.number_input("Âge", min_value=18, max_value=100, value=30)
sex = st.selectbox("Sexe", ["male", "female"])
job = st.selectbox("Emploi (0 = non qualifié, 3 = très qualifié)", [0, 1, 2, 3])
housing = st.selectbox("Logement", ["own", "free", "rent"])
saving_accounts = st.selectbox("Comptes d'épargne", ["NaN", "little", "moderate", "quite rich", "rich"])
checking_account = st.selectbox("Compte courant", ["NaN", "little", "moderate", "rich"])
credit_amount = st.number_input("Montant du crédit", min_value=100, max_value=100000, value=2000)
duration = st.number_input("Durée du crédit (en mois)", min_value=4, max_value=72, value=12)
purpose = st.selectbox("Objet du crédit", ["radio/TV", "education", "furniture/equipment", "car", "business", "domestic appliances", "repairs", "vacation/others"])

# Remplacer "NaN" par None pour le modèle
saving_accounts = None if saving_accounts == "NaN" else saving_accounts
checking_account = None if checking_account == "NaN" else checking_account

# Prédiction
if st.button("Prédire"):
    input_df = pd.DataFrame([{
        "Age": age,
        "Sex": sex,
        "Job": job,
        "Housing": housing,
        "Saving accounts": saving_accounts,
        "Checking account": checking_account,
        "Credit amount": credit_amount,
        "Duration": duration,
        "Purpose": purpose
    }])

    proba = model.predict_proba(input_df)[0]
    good_proba = proba[0]  # proba que le client est "good"
    bad_proba = proba[1]   # proba que le client est "bad"
    prediction = model.predict(input_df)[0]

    if bad_proba > 0.3:  # Seuil ajustable
        st.error(f"❌ Le client est **à risque** (probabilité de défaut : {bad_proba:.2f})")
    else:
        st.success(f"✅ Le client est **fiable** pour le crédit (probabilité : {good_proba:.2f})")
