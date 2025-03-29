import joblib
import pandas as pd
from fastapi import HTTPException

# Cargar el pipeline persistido
def load_pipeline(pipeline_path: str = "assets/modelo.joblib"):
    try:
        pipeline = joblib.load(pipeline_path)
        return pipeline
    except Exception as e:
        raise Exception(f"Error al cargar el pipeline: {e}")

# Función de predicción: transforma la entrada y usa el modelo para predecir
def predict(pipeline, input_df: pd.DataFrame):
    try:
        # Transformar los datos usando todas las etapas del pipeline excepto el modelo
        X_trans = pipeline[:-1].transform(input_df)
        # Obtener predicciones desde el modelo incremental
        predictions = pipeline.named_steps["modeloClf"].predict(X_trans)
        return predictions
    except Exception as e:
        raise Exception(f"Error en la predicción: {e}")

# Función de reentrenamiento: transforma los nuevos datos, actualiza el modelo con partial_fit y guarda el pipeline actualizado
def retrain(pipeline, new_data_df: pd.DataFrame, new_labels, pipeline_path: str = "assets/modelo.joblib"):
    try:
        # Transformar los nuevos datos usando las etapas de preprocesamiento del pipeline
        X_new_trans = pipeline[:-1].transform(new_data_df)
        # Actualizar el modelo incremental con partial_fit
        pipeline.named_steps["modeloClf"].partial_fit(X_new_trans, new_labels)
        # Persistir el pipeline actualizado
        joblib.dump(pipeline, pipeline_path)
        return pipeline
    except Exception as e:
        raise Exception(f"Error en el reentrenamiento: {e}")