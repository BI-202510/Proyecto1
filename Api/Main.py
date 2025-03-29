import io
import logging
from typing import List
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.exceptions import NotFittedError
from sklearn.metrics import precision_score, recall_score, f1_score
from EntrenarModelo import LimpiarTextoTransformer

app = FastAPI()

# Permite todas las solicitudes de cualquier origen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = None
version = 1

@app.get("/")
def read_root():
    global version
    return {"API": f"En funcionamiento. Version de pipeline: {version}"}

# Función para cargar el pipeline de manera asíncrona
def load_pipeline():
    global pipeline
    try:
        pipeline = joblib.load("assets/modelo.joblib")
    except Exception as e:
        print(f"Error al cargar el pipeline: {e}")

load_pipeline()

class InputData(BaseModel):
    Titulo: str
    Descripcion: str

# Endpoint de predicción vía JSON
@app.post("/predict")
def predict_endpoint_json(items: List[InputData]):
    try:
        # Verificar si la lista de datos no está vacía
        if not items:
            raise HTTPException(status_code=400, detail="No se proporcionaron datos para la predicción.")

        # Convertir la lista de objetos JSON a DataFrame
        input_df = pd.DataFrame([item.dict() for item in items])

        # Verificar si las columnas necesarias están presentes
        required_columns = {"Titulo", "Descripcion"}
        missing_columns = required_columns - set(input_df.columns)
        if missing_columns:
            raise HTTPException(status_code=400, detail=f"Faltan las siguientes columnas: {', '.join(missing_columns)}")

        # Verificar si el pipeline está entrenado
        try:
            # Transformar los datos con las etapas previas del pipeline
            X_trans = pipeline[:-1].transform(input_df)
        except NotFittedError:
            raise HTTPException(status_code=500, detail="El modelo no ha sido entrenado correctamente.")
        
        # Obtener las predicciones y las probabilidades de cada clase
        y_pred = pipeline.named_steps["modeloClf"].predict(X_trans)
        y_proba = pipeline.named_steps["modeloClf"].predict_proba(X_trans)

        # Extraer la probabilidad asociada a la clase predicha para cada instancia
        prob_pred = []
        classes_list = pipeline.named_steps["modeloClf"].classes_.tolist()
        for i, pred in enumerate(y_pred):
            idx = classes_list.index(pred)
            prob_pred.append(y_proba[i][idx])
        
        return {
            "predictions": y_pred.tolist(),
            "probabilities": prob_pred
        }
    except HTTPException as e:
        raise e  # Rethrow HTTP exceptions to keep the error message intact
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error inesperado: {str(e)}")

# Endpoint de predicción vía CSV
@app.post("/predict_csv")
async def predict_endpoint_csv(file: UploadFile = File(...)):
    try:
        # Verificar si el archivo está presente
        if not file:
            raise HTTPException(status_code=400, detail="No se ha proporcionado ningún archivo.")

        # Leer el archivo CSV
        contents = await file.read()
        try:
            input_df = pd.read_csv(io.StringIO(contents.decode('utf-8')), sep=';')
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error al leer el archivo CSV: {str(e)}")

        # Verificar si las columnas necesarias están presentes
        required_columns = {"Titulo", "Descripcion"}
        missing_columns = required_columns - set(input_df.columns)
        if missing_columns:
            raise HTTPException(status_code=400, detail=f"El archivo CSV debe contener las columnas: {', '.join(missing_columns)}")

        # Verificar si el pipeline está entrenado
        try:
            # Transformar los datos con las etapas previas del pipeline
            X_trans = pipeline[:-1].transform(input_df)
        except NotFittedError:
            raise HTTPException(status_code=500, detail="El modelo no ha sido entrenado correctamente.")
        
        # Obtener las predicciones y las probabilidades de cada clase
        y_pred = pipeline.named_steps["modeloClf"].predict(X_trans)
        y_proba = pipeline.named_steps["modeloClf"].predict_proba(X_trans)

        # Extraer la probabilidad asociada a la clase predicha para cada instancia
        prob_pred = []
        classes_list = pipeline.named_steps["modeloClf"].classes_.tolist()
        for i, pred in enumerate(y_pred):
            idx = classes_list.index(pred)
            prob_pred.append(y_proba[i][idx])
        
        return {
            "predictions": y_pred.tolist(),
            "probabilities": prob_pred
        }
    except HTTPException as e:
        raise e  # Rethrow HTTP exceptions to keep the error message intact
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error inesperado: {str(e)}")

@app.post("/retrain")
async def retrain_endpoint_csv(file: UploadFile = File(...)):
    try:
        # Leer el archivo CSV
        contents = await file.read()
        logging.info("Archivo CSV recibido, comenzando procesamiento.")
        df_new = pd.read_csv(io.StringIO(contents.decode('utf-8')), sep=';')

        required_columns = {"Titulo", "Descripcion", "Label"}
        if not required_columns.issubset(df_new.columns):
            logging.error(f"El CSV no tiene las columnas requeridas: {required_columns}")
            raise HTTPException(status_code=400, detail="El CSV debe contener las columnas 'Titulo', 'Descripcion' y 'Label'.")

        # Seleccionar solo las columnas necesarias
        new_data_df = df_new[["Titulo", "Descripcion"]]
        new_labels = df_new["Label"].tolist()

        logging.info("Datos cargados correctamente. Comenzando transformación.")

        # Transformar los nuevos datos con el pipeline (hasta la etapa de vectorización)
        X_new_trans = pipeline[:-1].transform(new_data_df)
        modelo = pipeline.named_steps["modeloClf"]

        # Actualizar el modelo incrementalmente con partial_fit
        logging.info("Reentrenando el modelo.")
        modelo.partial_fit(X_new_trans, new_labels)

        # Calcular métricas
        y_new_pred = modelo.predict(X_new_trans)
        precision = precision_score(new_labels, y_new_pred, average='macro')
        recall = recall_score(new_labels, y_new_pred, average='macro')
        f1 = f1_score(new_labels, y_new_pred, average='macro')

        # Guardar el pipeline actualizado
        joblib.dump(pipeline, "assets/modelo.joblib")
        logging.info("Pipeline reentrenado y guardado exitosamente.")
        
        global version
        version += 1  # Incrementar la versión del pipeline
        logging.info(f"Pipeline versión {version} guardado.")

        return {
            "detail": "Pipeline reentrenado y guardado exitosamente.",
            "metrics": {
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }
        }

    except Exception as e:
        logging.error(f"Error en el endpoint de reentrenamiento: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))