import pandas as pd
import unicodedata
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import spacy
import joblib

# Cargar el modelo de spaCy para español (Instalar: python -m spacy download es_core_news_sm)
nlp = spacy.load("es_core_news_sm")

# Transformador para el perfilamiento
class PerfilamientoTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Suponemos que X es un DataFrame
        df = X.copy()
        # Eliminar "ID" y "Fecha" solo si existen
        df = df.drop(columns=["ID", "Fecha"], errors="ignore")
        # Eliminamos filas con valores nulos
        df = df.dropna()
        # Eliminamos duplicados en la columna 'Titulo'
        df = df.drop_duplicates(subset=['Titulo'])
        return df

# Transformador para la limpieza de texto
class LimpiarTextoTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X.copy()
        df['Titulo'] = df['Titulo'].apply(self.limpiar_texto_aux)
        df['Descripcion'] = df['Descripcion'].apply(self.limpiar_texto_aux)
        return df

    @staticmethod
    def limpiar_texto_aux(texto):
        texto = texto.lower()
        tokens = word_tokenize(texto, language='spanish')
        stop_words = set(stopwords.words('spanish'))
        palabras_filtradas = [palabra for palabra in tokens if palabra not in stop_words]
        return ' '.join(palabras_filtradas)

# Transformador para la lematización
class LematizarTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X.copy()
        # Procesamos la columna 'Titulo'
        df['Titulo'] = self.lemmatize_series(df['Titulo'])
        # Procesamos la columna 'Descripcion'
        df['Descripcion'] = self.lemmatize_series(df['Descripcion'])
        return df

    @staticmethod
    def lemmatize_series(text_series):
        docs = nlp.pipe(text_series, batch_size=500)
        textos_lematizados = [
            " ".join([token.lemma_ for token in doc if not token.is_punct])
            for doc in docs
        ]
        # Normalizamos los textos para eliminar acentos
        textos_normalizados = [
            unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode('utf-8')
            for texto in textos_lematizados
        ]
        return textos_normalizados

# Transformador para la vectorización con HashingVectorizer
class VectorizarHashingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_features=1000):  # Reducir el número de características
        self.n_features = n_features
        self.vectorizer = HashingVectorizer(
            token_pattern=r"(?u)\b[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]{2,}\b",
            n_features=self.n_features,
            alternate_sign=False,
            norm='l2'
        )

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X.copy()
        textos = df["Titulo"] + " " + df["Descripcion"]
        return self.vectorizer.transform(textos)

# Cargar datos y aplicar perfilamiento
db_location = 'fake_news_spanish.csv'
df = pd.read_csv(db_location, sep=';', encoding="utf-8")

perfilador = PerfilamientoTransformer()
df_clean = perfilador.transform(df)

# Preparar datos para entrenamiento y prueba
X = df_clean[["Titulo", "Descripcion"]]
y = df_clean["Label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clases = df_clean["Label"].unique()

# Pipeline
pipeline = Pipeline([
    ('limpiar_texto', LimpiarTextoTransformer()),
    ('lematizar', LematizarTransformer()),
    ('vectorizar', VectorizarHashingTransformer(n_features=5000)),
    ('modeloClf', MultinomialNB())
])

# Transformar los datos hasta la etapa de vectorización
# Aquí transformamos los datos usando todas las etapas excepto el modelo
X_train_trans = pipeline[:-1].transform(X_train)
X_test_trans = pipeline[:-1].transform(X_test)

# Entrenar el modelo de forma incremental usando partial_fit
modelo = pipeline.named_steps['modeloClf']
modelo.partial_fit(X_train_trans, y_train, classes=clases)

# Realizar predicciones
predicciones = modelo.predict(X_test_trans)

# Persistencia del pipeline completo con joblib
joblib.dump(pipeline, "assets/modelo.joblib")
print("Pipeline guardado exitosamente.")
