import streamlit as st
import pandas as pd
import joblib
import numpy as np
import datetime
import os
import requests

#CONFIGURATION
st.set_page_config(
    page_title="Détection de Fraude Bancaire",
    page_icon="🏦",
    layout="centered"
)
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "fraud_model.pkl")
# Création du dossier model s'il n'existe pas sur le serveur Streamlit
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

#CHARGEMENT DES MODÈLES ET ENCODEURS
@st.cache_resource
def load_heavy_model():
    if not os.path.exists(MODEL_PATH):
        print("Le modèle n'existe pas localement. Début du téléchargement...")
        file_id = st.secrets["DRIVE_FILE_ID"]
        url = "https://docs.google.com/uc?export=download&confirm=t&id=" + file_id
        
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(MODEL_PATH, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            print("Téléchargement du modèle réussi!")
        except Exception as e:
            print(f"ERREUR lors du téléchargement : {e}")
            raise e
    else:
        print("Le modèle existe déjà localement.")
            
    print("Chargement du modèle principal avec joblib...")
    return joblib.load(MODEL_PATH)

# Exécution pas à pas avec logs dans le terminal
print("CHARGEMENT DES COMPOSANTS")
model = load_heavy_model()


le_type = joblib.load(os.path.join(MODEL_DIR, "le_type.pkl"))


le_status = joblib.load(os.path.join(MODEL_DIR, "le_status.pkl"))


le_local = joblib.load(os.path.join(MODEL_DIR, "le_localisation.pkl"))


scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

print("TOUS LES COMPOSANTS ONT ÉTÉ CHARGÉS AVEC SUCCÈS")
#INTERFACE UTILISATEUR
st.title("🏦 Détection de Fraude Bancaire")
st.write("Saisissez les caractéristiques de la transaction pour évaluer le risque de fraude.")

st.divider()

# Formulaire de saisie
col1, col2 = st.columns(2)

with col1:
    montant = st.number_input("Montant de la transaction (FCFA)", min_value=0.0, value=25000.0, step=500.0)
    type_trans = st.selectbox("Type de transaction", le_type.classes_)
    status_op = st.selectbox("Statut de l'opération", le_status.classes_)

with col2:
    localisation = st.selectbox("Localisation", le_local.classes_)
    date_saisie = st.date_input("Date de la transaction", datetime.date.today())
    heure_saisie = st.time_input("Heure de la transaction", datetime.time(12, 0))

if st.button("Lancer l'analyse...", type="primary"):
    try:
        #Traitement des variables de date
        jour = date_saisie.day
        mois = date_saisie.month
        jour_semaine = date_saisie.weekday()  # 0=Lundi, ..., 6=Dimanche
        heure = heure_saisie.hour

        # 2. Encodage des variables textuelles
        type_encoded = le_type.transform([type_trans])[0]
        status_encoded = le_status.transform([status_op])[0]
        local_encoded = le_local.transform([localisation])[0]

        features = np.array([[
            montant, 
            type_encoded, 
            status_encoded, 
            local_encoded, 
            jour, 
            mois, 
            jour_semaine, 
            heure
        ]])

        # Normalisation
        features_scaled = scaler.transform(features)

        # Prédiction et probabilités
        prediction = model.predict(features_scaled)[0]
        proba = model.predict_proba(features_scaled)[0][1]

        #résultats
        st.divider()
        # Exemple de structure pour afficher le verdict de l'IA
        if prediction == 1: # Si c'est une fraude
            st.error("🚨 **Alerte : Risque élevé de fraude détecté !**")
            st.metric(label="Statut de la transaction", value="SUSPECTE", delta="- Risque critique", delta_color="inverse")
        else: # Si la transaction est légitime
            st.success("✅ **Transaction validée : Aucun risque détecté.**")
            st.metric(label="Statut de la transaction", value="LÉGITIME", delta="Sécurisée")
            
        st.progress(float(proba))

    except Exception as e:
        st.error(f"Une erreur est survenue lors du traitement des données : {e}")

# --- PIED DE PAGE ---
st.sidebar.markdown("---")
st.sidebar.caption("Projet pédagogique — Détection de fraude bancaire par IA")
