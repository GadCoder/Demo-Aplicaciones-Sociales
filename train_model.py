import nltk
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from nltk.corpus import stopwords

nltk.download('stopwords')

# Sample dataset
data = [
    ("Qué tal enseña el profesor Cordero", "profesor"),
    ("Qué tal enseña el profesor Arredondo", "profesor"),
    ("Cómo es el profesor Gamarra", "profesor"),
    ("Cuál es la especialidad del profesor Luna?", "profesor"),
    ("El profesor Vargas da buenos exámenes?", "profesor"),
    ("Cuántos créditos como máximo puedo llevar en un ciclo?", "matricula"),
    ("Qué pasa si jalo un curso 2 o 3 veces?", "matricula"),
    ("Cuándo empieza el siguiente proceso de matrícula?", "matricula"),
    ("Cuándo es la siguiente rectificación?", "matricula"),
    ("Qué documentos necesito para la matrícula?", "matricula"),
    ("Cómo me inscribo en un curso electivo?", "matricula"),
    ("Cuándo es el siguiente proceso del carnet universitario?", "carnet"),
    ("Cómo puedo renovar mi carnet universitario?", "carnet"),
    ("Dónde recojo mi carnet universitario?", "carnet"),
    ("Qué cursos se llevan en el 6to ciclo de ingeniería de software?", "carrera"),
    ("Qué cursos se llevan en el 3er ciclo de ingeniería de sistemas?", "carrera"),
    ("Qué cursos electivos puedo tomar en ingeniería de software?", "carrera"),
    ("Qué especializaciones ofrece la carrera de ingeniería de sistemas?", "carrera"),
    ("Qué se ve en el curso de Taller de aplicaciones sociales en ingeniería de software?", "curso"),
    ("Qué temas se ven en el curso de Arquitectura de software en ingeniería de software?", "curso"),
    ("Qué se ve en el curso de Ecuaciones diferenciales en ingeniería de sistemas?", "curso"),
    ("Cuál es el enfoque del curso de Inteligencia Artificial?", "curso"),
    ("Qué libros se utilizan en el curso de Programación Orientada a Objetos?", "curso"),
]


# Separate data into features (X) and labels (y)
X, y = zip(*data)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline for text vectorization and classification
model = make_pipeline(TfidfVectorizer(stop_words=stopwords.words('spanish')), MultinomialNB())

# Train the model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))

# Test with a new sentence
def predict_intent(text):
    return model.predict([text])[0]


joblib.dump(model, 'intent_classifier_model.pkl')
