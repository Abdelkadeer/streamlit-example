import streamlit as st
import pandas as pd
from keybert import KeyBERT
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
import nltk
import spacy
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

# Chemin local vers les ressources NLTK dans votre dépôt GitHub
nltk_data_path = "ntlk.py"

# Ajouter le chemin local à nltk.data.path
nltk.data.path.append(nltk_data_path)
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

# Fonction pour extraire les mots-clés avec TextRank
def extract_textrank_keywords(text, top_n=10):
    words = nltk.word_tokenize(text)
    unique_words = list(set(words))
    vocab = {word: i for i, word in enumerate(unique_words)}
    reverse_vocab = {i: word for word, i in vocab.items()}
    
    # Créer une matrice de similarité
    word_vectors = TfidfVectorizer().fit_transform(unique_words)
    similarity_matrix = cosine_similarity(word_vectors)
    
    # Construire le graphe
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)
    
    # Extraire les mots-clés
    ranked_words = sorted(scores, key=scores.get, reverse=True)
    top_keywords = [reverse_vocab[i] for i in ranked_words[:top_n]]
    
    return top_keywords

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

        # Extraction des mots-clés avec TextRank (utilisant networkx)
        keywords_textrank = extract_textrank_keywords(document_pretraite, top_n=10)

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
