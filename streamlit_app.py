import streamlit as st
import pandas as pd
from keybert import KeyBERT
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import yake
from wordcloud import WordCloud

# Titre et description
st.markdown("# Une approche hybride pour l’analyse des documents médicaux")
st.info("L'objectif de cette approche est d'extraire des mots-clés à partir de deux jeux de données : les échantillons de transcription médicale et les avis sur les médicaments")
st.title('Données Médicales : Extraction des mots clés')

# Chargement des données
data = pd.read_csv('mtsamples.csv')
data.rename(columns={"Unnamed: 0": "ID"}, inplace=True)
colonnes_a_visualiser = ['ID', 'description', 'medical_specialty', 'sample_name', 'transcription', 'keywords']
data_selection = data[colonnes_a_visualiser]

# Sélection des colonnes à afficher
colonnes_selectionnees = st.multiselect('Sélectionner les colonnes à visualiser', colonnes_a_visualiser, default=colonnes_a_visualiser)

# Filtrer les données en fonction des colonnes sélectionnées
data_affichee = data_selection[colonnes_selectionnees]

# Afficher les données sélectionnées
st.write(data_affichee)

# Compter le nombre de valeurs manquantes pour chaque colonne sélectionnée
missing_values = data_selection.isnull().sum()

# Créer un graphique à barres pour visualiser les valeurs manquantes
fig, ax = plt.subplots(figsize=(10, 6))
missing_values.plot(kind='bar', color='skyblue', ax=ax)
ax.set_title('Nombre de valeurs manquantes par colonne')
ax.set_xlabel('Colonnes')
ax.set_ylabel('Nombre de valeurs manquantes')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()

# Afficher le graphique avec Streamlit
st.pyplot(fig)

# Supprimer les lignes avec des valeurs manquantes dans les colonnes 'transcription' et 'keywords'
data = data.dropna(subset=['transcription', 'keywords'])

# Afficher la nouvelle forme du DataFrame
st.write("Shape après suppression des valeurs manquantes :", data.shape)

# Créer un histogramme pour visualiser la longueur des transcriptions
fig, ax = plt.subplots(figsize=(10, 6))
data['transcription'].apply(len).plot(kind='hist', bins=20, color='skyblue', ax=ax)
ax.set_title('Distribution de la longueur des transcriptions')
ax.set_xlabel('Longueur des transcriptions')
ax.set_ylabel('Fréquence')
ax.grid(True)
plt.tight_layout()

# Afficher le graphique avec Streamlit
st.pyplot(fig)

# Créer une chaîne de caractères contenant tous les mots-clés
keywords_text = ' '.join(data['keywords'].dropna())

# Créer un nuage de mots à partir des mots-clés
wordcloud = WordCloud(width=800, height=400, background_color='black').generate(keywords_text)

# Afficher le nuage de mots avec Streamlit
st.title('Nuage de mots des mots-clés du dataset')
st.image(wordcloud.to_array(), use_column_width=True)

# Filtrer les données pour exclure les valeurs manquantes dans la spécialité médicale
df_filtered = data.dropna(subset=['medical_specialty'])

# Limiter le nombre de spécialités médicales à afficher dans le graphique
top_specialites = df_filtered['medical_specialty'].value_counts().nlargest(5)

# Créer un dictionnaire de couleurs personnalisées pour les spécialités médicales
colors = plt.cm.tab10.colors[:len(top_specialites)]

# Créer un graphique en camembert pour visualiser la répartition des spécialités médicales
fig, ax = plt.subplots(figsize=(10, 6))
top_specialites.plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=colors, ax=ax)
ax.set_title('Répartition des spécialités médicales (top 5)')
ax.axis('equal')  # Assurer que le diagramme est un cercle

# Ajouter une légende
ax.legend(labels=top_specialites.index, loc='best')
plt.tight_layout()

# Afficher le graphique avec Streamlit
st.pyplot(fig)

# Sélection du document
document_selectionne = st.selectbox('Sélectionner un document:', data['transcription'])

# Extraction des mots-clés avec KeyBERT
model_keybert = KeyBERT('distilbert-base-nli-mean-tokens')
keywords_keybert = model_keybert.extract_keywords(document_selectionne, keyphrase_ngram_range=(1, 1), stop_words='english')

# Extraire uniquement les mots-clés
keywords_keybert_list = [keyword for keyword, score in keywords_keybert]

st.title('Évaluation du modèle KeyBERT')

st.write('Mots-clés (KeyBERT) :', ', '.join(keywords_keybert_list))

# Fonction pour évaluer les mots-clés extraits
def evaluate_keywords(row, extracted_keywords):
    # Vérifier si la valeur de la colonne 'keywords' est une chaîne de caractères
    if isinstance(row['keywords'], str):
        actual_keywords = row['keywords'].split(', ')
    else:
        actual_keywords = []
    # Calculer la précision en comparant les mots-clés extraits avec les mots-clés réels
    precision = len(set(actual_keywords) & set(extracted_keywords)) / len(set(extracted_keywords)) if extracted_keywords else 0
    recall = len(set(actual_keywords) & set(extracted_keywords)) / len(set(actual_keywords)) if actual_keywords else 0
    return precision, recall

# Appliquer la fonction sur chaque ligne du DataFrame
data['precision'], data['recall'] = zip(*data.apply(lambda row: evaluate_keywords(row, keywords_keybert_list), axis=1))

# Calculer le F-score
data['f_score'] = 2 * (data['precision'] * data['recall']) / (data['precision'] + data['recall'])

# Afficher la précision, le rappel et le F-score pour le document sélectionné
st.write("Précision :", data['precision'].iloc[0])
st.write("Rappel :", data['recall'].iloc[0])
st.write("F-score :", data['f_score'].iloc[0])





