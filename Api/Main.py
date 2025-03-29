from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List
import pandas as pd
import io

from prediction_model import load_pipeline, predict, retrain

app = FastAPI()

# Modelo de datos para entrada JSON (para predicción)
class InputData(BaseModel):
    Titulo: str
    Descripcion: str

# Cargar el pipeline al iniciar la aplicación
pipeline = load_pipeline()

@app.post("/predict")
def predict_endpoint_json(items: List[InputData]):
    """
    Endpoint para predicción mediante JSON. Puede recibir una lista de ejemplos.
    """
    try:
        # Convertir la lista de entrada en un DataFrame
        input_df = pd.DataFrame([item.dict() for item in items])
        preds = predict(pipeline, input_df)
        return {"predictions": preds.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_csv")
async def predict_endpoint_csv(file: UploadFile = File(...)):
    """
    Endpoint para predicción mediante CSV. El archivo debe contener columnas "Titulo" y "Descripcion".
    """
    try:
        contents = await file.read()
        # Suponiendo que el CSV usa ';' como separador (ajusta según tu caso)
        input_df = pd.read_csv(io.StringIO(contents.decode('utf-8')), sep=';')
        preds = predict(pipeline, input_df)
        return {"predictions": preds.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain_csv")
async def retrain_endpoint_csv(file: UploadFile = File(...)):
    """
    Endpoint para reentrenamiento mediante CSV. El CSV debe incluir las columnas "Titulo", "Descripcion" y "Label".
    """
    try:
        contents = await file.read()
        # Leer el CSV (ajusta el separador si es necesario)
        df_new = pd.read_csv(io.StringIO(contents.decode('utf-8')), sep=';')
        # Verificar que las columnas necesarias existan
        if not {"Titulo", "Descripcion", "Label"}.issubset(df_new.columns):
            raise HTTPException(status_code=400, detail="El CSV debe contener las columnas 'Titulo', 'Descripcion' y 'Label'.")
        new_data_df = df_new[["Titulo", "Descripcion"]]
        new_labels = df_new["Label"].tolist()
        # Actualizar el pipeline usando los nuevos datos
        retrain(pipeline, new_data_df, new_labels)
        return {"detail": "Pipeline reentrenado y guardado exitosamente."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))