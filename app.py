from __future__ import annotations

import streamlit as st
from pathlib import Path

# Configuration de la page
st.set_page_config(
    page_title="Credit Risk Classification",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre principal
st.title("🏦 Credit Risk Classification")
st.markdown("### Application d'analyse du risque de crédit")

# Navigation
page = st.sidebar.radio(
    "Navigation",
    ["Accueil", "Prédiction", "Statistiques", "À propos"]
)

if page == "Accueil":
    st.markdown("""
    #### Bienvenue !
    Cette application utilise un modèle de Machine Learning pour analyser le risque d'accorder un crédit à un client.
    
    **Fonctionnalités :**
    - 📊 Visualisation des données
    - 🔮 Prédictions individuelles
    - 📈 Statistiques et analyses
    """)

elif page == "Prédiction":
    st.header("Prédiction du risque")
    st.write("Entrez les informations du client pour obtenir une prédiction.")
    # À compléter avec ton code de prédiction

elif page == "Statistiques":
    st.header("Statistiques & Visualisations")
    st.write("Visualisations des données d'entraînement.")
    # À compléter avec ton code de stats

elif page == "À propos":
    st.header("À propos du modèle")
    st.markdown("""
    - **Modèle utilisé :** Gradient Boosting
    - **Précision :** À compléter
    - **Dataset :** Risque_data.xls
    """)
