import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
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
st.write(data_affichee)
# Compter le nombre de valeurs manquantes pour chaque colonne sélectionnée
missing_values = data_selection.isnull().sum()

# Créer un graphique à barres pour visualiser les valeurs manquantes
fig, ax = plt.subplots(figsize=(10, 6))
missing_values.plot(kind='bar', color='skyblue', ax=ax)
plt.title('Nombre de valeurs manquantes par colonne')
plt.xlabel('Colonnes')
plt.ylabel('Nombre de valeurs manquantes')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Afficher le graphique avec Streamlit
st.pyplot(fig)
