from pydantic import BaseModel

class InputDataPred(BaseModel):
    Titulo: str
    Descripcion: str

    def columns(self):
        return ["Titulo", "Descripcion"]

class InputDataRetrain(BaseModel):
    Titulo: str
    Descripcion: str
    Label: int

    def columns(self):
        return ["Titulo", "Descripcion", "Label"]