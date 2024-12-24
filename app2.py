import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import nltk
import pandas as pd
import numpy as np

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize NLTK components
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Streamlit interface
def main():
    st.title("ChatBot de traitement du langage naturel avec NLTK pour la classification de texte")
    st.write("Cette application illustre un chatbot pour la classification de texte en utilisant des techniques de traitement du langage naturel (NLP).")

    # File Upload
    uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.dataframe(data.head())

        text_column = st.selectbox("Select the text column (e.g., User Query)", data.columns)
        label_column = st.selectbox("Select the label column (e.g., Intent)", data.columns)

        if st.button("Train Chatbot"):
            # Preprocessing
            st.subheader("NLP Techniques")
            
            data['tokenized'] = data[text_column].apply(lambda x: word_tokenize(str(x).lower()))
            data['no_stopwords'] = data['tokenized'].apply(lambda x: [word for word in x if word not in stop_words])
            data['lemmatized'] = data['no_stopwords'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
            data['stemmed'] = data['lemmatized'].apply(lambda x: [stemmer.stem(word) for word in x])
            data['processed'] = data['stemmed'].apply(lambda x: ' '.join(x))

            st.write("Processed Dataset:")
            st.dataframe(data[[text_column, 'processed']].head())

            # Vectorization
            vectorizer = CountVectorizer()
            X = vectorizer.fit_transform(data['processed'])
            y = data[label_column]

            # Train-Test Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Hyperparameter Tuning for Decision Tree
            dt_params = {'criterion': ['gini', 'entropy'], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}
            dt_grid = GridSearchCV(DecisionTreeClassifier(random_state=42), dt_params, cv=3, scoring='accuracy')
            dt_grid.fit(X_train, y_train)

            # Best Decision Tree Model
            best_dt = dt_grid.best_estimator_
            dt_predictions = best_dt.predict(X_test)

            # Hyperparameter Tuning for Naive Bayes
            nb_params = {'alpha': [0.1, 0.5, 1.0, 5.0, 10.0]}
            nb_grid = GridSearchCV(MultinomialNB(), nb_params, cv=3, scoring='accuracy')
            nb_grid.fit(X_train, y_train)

            # Best Naive Bayes Model
            best_nb = nb_grid.best_estimator_
            nb_predictions = best_nb.predict(X_test)

            # Display Performance
            st.subheader("Model Performance")

            st.write("### Decision Tree Classifier (Hyperparameter Tuned):")
            st.write("Best Parameters:", dt_grid.best_params_)
            st.write("Accuracy:", accuracy_score(y_test, dt_predictions))
            st.text(classification_report(y_test, dt_predictions))

            st.write("### Naive Bayes Classifier (Hyperparameter Tuned):")
            st.write("Best Parameters:", nb_grid.best_params_)
            st.write("Accuracy:", accuracy_score(y_test, nb_predictions))
            st.text(classification_report(y_test, nb_predictions))

            # Chatbot Interaction
            st.subheader("Chatbot Interaction")
            st.write("Entrer une demande de clissification de texte pour le chatbot.")

            user_input = st.text_input("Your Query:")

            if user_input:
                processed_input = ' '.join([stemmer.stem(lemmatizer.lemmatize(word.lower())) for word in word_tokenize(user_input) if word.lower() not in stop_words])
                input_vector = vectorizer.transform([processed_input])

                dt_prediction = best_dt.predict(input_vector)[0]
                nb_prediction = best_nb.predict(input_vector)[0]

                st.write("### Predicted Intent:")
                st.write(f"Decision Tree: {dt_prediction}")
                st.write(f"Naive Bayes: {nb_prediction}")

            st.write("### Explanation of NLP and Chatbot Functionality:")
            st.write(
                "Le traitement du langage naturel (NLP) consiste à traiter des données linguistiques humaines pour permettre aux machines de les comprendre et de les analyser.\n"
                "Les techniques de NLP, telles que la tokenisation, la suppression des mots vides, la lemmatisation et la racinisation, aident à préparer le texte pour des tâches de classification.\n"
                "Ce chatbot utilise ces techniques, combinées à des classificateurs de machine learning, pour prédire les intentions des utilisateurs à partir de requêtes liées aux voyages."
            )

if __name__ == "__main__":
    main()
