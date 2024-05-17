

import streamlit as st
import pandas as pd
from keybert import KeyBERT
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import yake
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

# Afficher les données sélectionnées
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
# Supprimer les lignes avec des valeurs manquantes dans les colonnes 'transcription' et 'keywords'
data = data.dropna(subset=['transcription', 'keywords'])

# Afficher la nouvelle forme du DataFrame
st.write("Shape après suppression des valeurs manquantes :", data.shape)

# Créer un histogramme pour visualiser la longueur des transcriptions
plt.figure(figsize=(10, 6))
data['transcription'].apply(len).plot(kind='hist', bins=20, color='skyblue')
plt.title('Distribution de la longueur des transcriptions')
plt.xlabel('Longueur des transcriptions')
plt.ylabel('Fréquence')
plt.grid(True)
plt.tight_layout()

# Afficher le graphique avec Streamlit
st.pyplot()
from wordcloud import WordCloud

# Charger les données
data = pd.read_csv('mtsamples.csv')

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

st.title('Evaluation du modèle KeyBert')
# Affichage des mots-clés
from googletrans import Translator
# Initialisation du traducteur
translator = Translator()

# Traduction des mots-clés en français
keywords_translated = [translator.translate(keyword, src='en', dest='fr').text for keyword in keywords_keybert_list]

st.title('Évaluation du modèle KeyBERT')
# Affichage des mots-clés traduits
st.write('Mots-clés (KeyBERT) en français :', ', '.join(keywords_translated))
st.write('Mots-clés (KeyBERT) :', ', '.join(keywords_keybert_list))

# Fonction pour évaluer les mots-clés extraits
def evaluate_keywords(row):
    # Vérifier si la valeur de la colonne 'keywords' est une chaîne de caractères
    if isinstance(row['keywords'], str):
        actual_keywords = row['keywords'].split(', ')
    else:
        actual_keywords = []
    # Utiliser la liste keywords_tfidf que vous avez déjà extraite
    extracted_keywords = keywords_keybert_list
    # Calculer la précision en comparant les mots-clés extraits avec les mots-clés réels
    precision = len(set(actual_keywords) & set(extracted_keywords)) / len(set(extracted_keywords)) if extracted_keywords else 0
    recall = len(set(actual_keywords) & set(extracted_keywords)) / len(set(actual_keywords)) if actual_keywords else 0
    return precision, recall
data['precision'], data['recall'] = zip(*data.apply(evaluate_keywords, axis=1))

# Calculer le F-score
data['f_score'] = 2 * (data['precision'] * data['recall']) / (data['precision'] + data['recall'])





# Afficher les métriques moyennes
average_precision = data['precision'].mean()
average_recall = data['recall'].mean()
average_f_score = data['f_score'].mean()



st.write("Précision moyenne des mots-clés extraits :", average_precision)
st.write("Rappel moyen des mots-clés extraits :", average_recall)
st.write("F-score moyen des mots-clés extraits :", average_f_score)



# Extraction des mots-clés avec TF-IDF
data.dropna(subset=['transcription'], inplace=True)
# Extraction des mots-clés avec TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(data['transcription'])
document_index = data[data['transcription'] == document_selectionne].index[0]
tfidf_scores = tfidf_matrix[document_index].toarray()[0]
top_tfidf_indices = tfidf_scores.argsort()[-5:][::-1]
feature_names = tfidf_vectorizer.get_feature_names_out()  # Utiliser get_feature_names_out()
keywords_tfidf = [feature_names[index] for index in top_tfidf_indices]

# Affichage des mots-clés
st.title('Evaluation de méthode TF_IDF')
st.write('Mots-clés (TF-IDF) :', ', '.join(keywords_tfidf))

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Fonction pour évaluer les mots-clés extraits
def evaluate_keywords(row):
    # Vérifier si la valeur de la colonne 'keywords' est une chaîne de caractères
    if isinstance(row['keywords'], str):
        actual_keywords = row['keywords'].split(', ')
    else:
        actual_keywords = []
    # Utiliser la liste keywords_tfidf que vous avez déjà extraite
    extracted_keywords = keywords_tfidf
    # Calculer la précision en comparant les mots-clés extraits avec les mots-clés réels
    precision = len(set(actual_keywords) & set(extracted_keywords)) / len(set(extracted_keywords)) if extracted_keywords else 0
    # Calculer le rappel
    recall = len(set(actual_keywords) & set(extracted_keywords)) / len(set(actual_keywords)) if actual_keywords else 0
    return precision, recall

# Appliquer la fonction d'évaluation à chaque ligne
data['precision'], data['recall'] = zip(*data.apply(evaluate_keywords, axis=1))

# Calculer le F-score
data['f_score'] = 2 * (data['precision'] * data['recall']) / (data['precision'] + data['recall'])





# Afficher les métriques moyennes
average_precision = data['precision'].mean()
average_recall = data['recall'].mean()
average_f_score = data['f_score'].mean()



st.write("Précision moyenne des mots-clés extraits :", average_precision)
st.write("Rappel moyen des mots-clés extraits :", average_recall)
st.write("F-score moyen des mots-clés extraits :", average_f_score)





# Extraction des mots-clés avec TextRank (YAKE)
language = "en"
max_ngram_size = 1
deduplication_threshold = 0.9
num_keywords = 5

custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=num_keywords, features=None)
keywords_yake = custom_kw_extractor.extract_keywords(document_selectionne)

# Extraire uniquement les mots-clés
keywords_yake_list = [keyword for keyword, score in keywords_yake]

# Affichage des mots-clés avec TextRank (YAKE)
st.title('Evaluation de méthode TextRank - YAKE')
st.write('Mots-clés (TextRank - YAKE) :', ', '.join(keywords_yake_list))
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Fonction pour évaluer les mots-clés extraits
def evaluate_keywords(row):
    # Vérifier si la valeur de la colonne 'keywords' est une chaîne de caractères
    if isinstance(row['keywords'], str):
        actual_keywords = row['keywords'].split(', ')
    else:
        actual_keywords = []
    # Utiliser la liste keywords_tfidf que vous avez déjà extraite
    extracted_keywords = keywords_yake_list
    # Calculer la précision en comparant les mots-clés extraits avec les mots-clés réels
    precision = len(set(actual_keywords) & set(extracted_keywords)) / len(set(extracted_keywords)) if extracted_keywords else 0
    # Calculer le rappel
    recall = len(set(actual_keywords) & set(extracted_keywords)) / len(set(actual_keywords)) if actual_keywords else 0
    return precision, recall

# Appliquer la fonction d'évaluation à chaque ligne
data['precision'], data['recall'] = zip(*data.apply(evaluate_keywords, axis=1))

# Calculer le F-score
data['f_score'] = 2 * (data['precision'] * data['recall']) / (data['precision'] + data['recall'])





# Afficher les métriques moyennes
average_precision = data['precision'].mean()
average_recall = data['recall'].mean()
average_f_score = data['f_score'].mean()



st.write("Précision moyenne des mots-clés extraits :", average_precision)
st.write("Rappel moyen des mots-clés extraits :", average_recall)
st.write("F-score moyen des mots-clés extraits :", average_f_score)







# Charger les données
df = pd.read_csv('drugsComTrain_raw.csv')
test = pd.read_csv('drugsComTest_raw.csv')
data1 = pd.concat([df, test])
# Afficher les premières lignes du DataFrame
st.title(" Dataset Drug Reviews")
st.write(data1.head())

from wordcloud import WordCloud
from wordcloud import STOPWORDS

# Définir les stopwords
stopwords = set(STOPWORDS)

# Créer le nuage de mots
wordcloud = WordCloud(background_color='orange', stopwords=stopwords, width=1200, height=800).generate(str(data1['drugName']))

# Définir la taille de la figure
plt.rcParams['figure.figsize'] = (15, 15)

# Afficher le nuage de mots avec Streamlit
st.title('Nuage de mots des noms de médicaments')
st.image(wordcloud.to_array(), use_column_width=True)










# Définir les sentiments en fonction de la notation
data1.loc[(data1['rating'] >= 5), 'Review_Sentiment'] = 1
data1.loc[(data1['rating'] < 5), 'Review_Sentiment'] = 0

# Compter les sentiments
sentiment_counts = data1['Review_Sentiment'].value_counts()

# Créer un graphique en camembert pour représenter les sentiments des patients
fig, ax = plt.subplots()
ax.pie(sentiment_counts, labels=['Positive Sentiment', 'Negative Sentiment'], autopct='%.2f%%', colors=['lightblue', 'navy'], explode=[0, 0.1])
ax.set_title('Représentation en pie chart des sentiments (rating)', fontsize=25)
ax.axis('equal')  # Assurer que le diagramme est un cercle

# Afficher le graphique avec Streamlit
st.pyplot(fig)

document_selectionne1 = st.selectbox('Sélectionner un review: ', data1['review'])

# Extraction des mots-clés avec KeyBERT
model_keybert = KeyBERT('distilbert-base-nli-mean-tokens')
keywords_keybert = model_keybert.extract_keywords(document_selectionne1, keyphrase_ngram_range=(1, 1), stop_words='english')
# Extraire uniquement les mots-clés
keywords_keybert_list = [keyword for keyword, score in keywords_keybert]

# Affichage des mots-clés
st.write('Mots-clés (KeyBERT) :', ', '.join(keywords_keybert_list))

import seaborn as sns

# Création d'un dataframe pour les mots-clés et leur score
df_keywords = pd.DataFrame(keywords_keybert, columns=['Keyword', 'Score'])

# Graphique à barres pour les mots-clés et leur score
plt.figure(figsize=(10, 6))
sns.barplot(x='Score', y='Keyword', data=df_keywords, palette='viridis')
plt.title('Extraction des  Keywords avec KeyBERT')
plt.xlabel('Score')
plt.ylabel('Keyword')
plt.xticks(rotation=45)
# Afficher le graphique dans Streamlit
st.pyplot()
