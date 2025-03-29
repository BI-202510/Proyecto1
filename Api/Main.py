from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List
import pandas as pd
import io
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score

app = FastAPI()

class InputData(BaseModel):
    Titulo: str
    Descripcion: str

# Función para cargar el pipeline persistido
def load_pipeline(pipeline_path: str = "assets/modelo.joblib"):
    try:
        pipeline = joblib.load(pipeline_path)
        return pipeline
    except Exception as e:
        raise Exception(f"Error al cargar el pipeline: {e}")

# Cargar el pipeline al iniciar la aplicación
pipeline = load_pipeline()

# Endpoint de predicción vía JSON
@app.post("/predict")
def predict_endpoint_json(items: List[InputData]):
    try:
        # Convertir la lista de objetos JSON a DataFrame
        input_df = pd.DataFrame([item.dict() for item in items])
        # Transformar los datos con las etapas previas del pipeline
        X_trans = pipeline[:-1].transform(input_df)
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint de predicción vía CSV
@app.post("/predict_csv")
async def predict_endpoint_csv(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        # Convertir el contenido CSV a DataFrame (ajusta el separador si es necesario)
        input_df = pd.read_csv(io.StringIO(contents.decode('utf-8')), sep=';')
        # Transformar los datos
        X_trans = pipeline[:-1].transform(input_df)
        y_pred = pipeline.named_steps["modeloClf"].predict(X_trans)
        y_proba = pipeline.named_steps["modeloClf"].predict_proba(X_trans)
        
        prob_pred = []
        classes_list = pipeline.named_steps["modeloClf"].classes_.tolist()
        for i, pred in enumerate(y_pred):
            idx = classes_list.index(pred)
            prob_pred.append(y_proba[i][idx])
        
        return {
            "predictions": y_pred.tolist(),
            "probabilities": prob_pred
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint de reentrenamiento vía CSV
@app.post("/retrain")
async def retrain_endpoint_csv(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        # Leer el CSV; se espera que tenga las columnas: "Titulo", "Descripcion" y "Label"
        df_new = pd.read_csv(io.StringIO(contents.decode('utf-8')), sep=';')
        required_columns = {"Titulo", "Descripcion", "Label"}
        if not required_columns.issubset(df_new.columns):
            raise HTTPException(
                status_code=400, 
                detail="El CSV debe contener las columnas 'Titulo', 'Descripcion' y 'Label'."
            )
        new_data_df = df_new[["Titulo", "Descripcion"]]
        new_labels = df_new["Label"].tolist()
        
        # Transformar los nuevos datos con el pipeline (hasta la etapa de vectorización)
        X_new_trans = pipeline[:-1].transform(new_data_df)
        modelo = pipeline.named_steps["modeloClf"]
        
        # Actualizar el modelo incremental con partial_fit
        modelo.partial_fit(X_new_trans, new_labels)
        
        # Calcular métricas sobre el nuevo lote
        y_new_pred = modelo.predict(X_new_trans)
        precision = precision_score(new_labels, y_new_pred, average='macro')
        recall = recall_score(new_labels, y_new_pred, average='macro')
        f1 = f1_score(new_labels, y_new_pred, average='macro')
        
        # Guardar el pipeline actualizado
        joblib.dump(pipeline, "assets/modelo.joblib")
        
        return {
            "detail": "Pipeline reentrenado y guardado exitosamente.",
            "metrics": {
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))