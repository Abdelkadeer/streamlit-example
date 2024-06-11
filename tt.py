import streamlit as st
import pandas as pd
from keybert import KeyBERT
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from summa import keywords as summa_keywords
from sklearn import svm
import nltk
import spacy
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Télécharger les ressources nécessaires pour nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Charger spacy modèle anglais
nlp = spacy.load('en_core_web_sm')

# Fonction de préparation des données
def preprocess_text(text):
    # Convertir en minuscules
    text = text.lower()
    
    # Supprimer la ponctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenizer
    words = nltk.word_tokenize(text)
    
    # Supprimer les stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Lemmatisation
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    return ' '.join(words)

# Interface utilisateur
st.title('Extraction et Classification des Mots-clés Médicaux')

# Zones de texte pour l'article médical et les mots-clés manuels
article_text = st.text_area("Copiez votre article médical ici", height=200)
manual_keywords = st.text_area("Entrez les mots-clés extraits manuellement ici (séparés par des virgules)", height=100)

if st.button('Extraire et Classifier les Mots-clés'):
    if article_text:
        document_pretraite = preprocess_text(article_text)
        
        # Extraction des mots-clés avec KeyBERT
        kw_model = KeyBERT()
        keywords_bert = kw_model.extract_keywords(document_pretraite, top_n=10)
        keywords_bert = [kw[0] for kw in keywords_bert]

        # Extraction des mots-clés avec TF-IDF
        tfidf_vectorizer = TfidfVectorizer(max_features=10)
        tfidf_matrix = tfidf_vectorizer.fit_transform([document_pretraite])
        keywords_tfidf = tfidf_vectorizer.get_feature_names_out()

        # Extraction des mots-clés avec TextRank (Summa)
        keywords_textrank = summa_keywords.keywords(document_pretraite, words=10).split('\n')

        # Fusionner tous les mots-clés en un seul ensemble
        all_keywords = list(set(keywords_bert + list(keywords_tfidf) + keywords_textrank))

        # Si des mots-clés manuels sont fournis, utiliser SVM pour la classification
        if manual_keywords:
            st.write("Filtrage des mots avec SVM")
            manual_keywords_list = [kw.strip() for kw in manual_keywords.split(',')]
            labels = [1 if kw in manual_keywords_list else 0 for kw in all_keywords]
            
            # Préparation des vecteurs
            vectorizer = TfidfVectorizer().fit_transform(all_keywords)
            vectors = vectorizer.toarray()
            
            # Entraînement du modèle SVM
            clf = svm.SVC(kernel='linear')
            clf.fit(vectors, labels)
            
            # Prédictions
            predicted_labels = clf.predict(vectors)
            
            # Séparation des mots-clés pertinents et non pertinents
            relevant_keywords = [all_keywords[i] for i in range(len(all_keywords)) if predicted_labels[i] == 1]
            non_relevant_keywords = [all_keywords[i] for i in range(len(all_keywords)) if predicted_labels[i] == 0]
        else:
            st.write("Filtrage des mots avec K-means")
            # Clustering des mots-clés avec K-means
            vectorizer = CountVectorizer().fit_transform(all_keywords)
            vectors = vectorizer.toarray()
            kmeans = KMeans(n_clusters=2, random_state=0).fit(vectors)
            labels = kmeans.labels_

            # Séparer les mots-clés en deux clusters
            cluster_1 = [all_keywords[i] for i in range(len(all_keywords)) if labels[i] == 0]
            cluster_2 = [all_keywords[i] for i in range(len(all_keywords)) if labels[i] == 1]

            # Identifier les clusters pertinents et non pertinents
            relevant_keywords = cluster_1 if len(cluster_1) > len(cluster_2) else cluster_2
            non_relevant_keywords = cluster_2 if len(cluster_1) > len(cluster_2) else cluster_1

        # Afficher les résultats
        st.markdown("## Résultats des Mots-clés")
        st.write("### Mots-clés pertinents")
        st.write(relevant_keywords)
    else:
        st.error("Veuillez entrer un article médical.")
