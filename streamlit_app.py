import streamlit as st
import pandas as pd

st.markdown("# Une approche hybride pour l’analyse des documents médicaux")
# Chargement des données
data = pd.read_csv('mtsamples.csv')
data.rename(columns={"Unnamed: 0": "ID"}, inplace=True)
colonnes_a_visualiser = ['ID', 'description', 'medical_specialty', 'sample_name', 'transcription', 'keywords']
data_selection = data[colonnes_a_visualiser]

# Afficher les données
st.title('Données Médicales : Extraction des mots clés')

# Sélection des colonnes à afficher
colonnes_selectionnees = st.multiselect('Sélectionner les colonnes à visualiser', colonnes_a_visualiser)

# Filtrer les données en fonction des colonnes sélectionnées
data_affichee = data_selection[colonnes_selectionnees]
