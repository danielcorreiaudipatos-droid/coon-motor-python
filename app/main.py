from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import numpy as np
from scipy import stats
from .regressao import calcular_regressao

app = FastAPI(title="CO.ON Motor Estatístico", version="1.0.0")

class Amostra(BaseModel):
    id: str
    ordem: int
    area_total: Optional[float] = None
    area_priv: Optional[float] = None
    valor: float
    fonte: Optional[str] = None

class Laudo(BaseModel):
    id: str
    endereco: str
    cidade: str
    uf: str
    finalidade: str

class RegressaoRequest(BaseModel):
    laudo: Laudo
    amostras: list[Amostra]

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/regressao")
def regressao(req: RegressaoRequest):
    if len(req.amostras) < 5:
        raise HTTPException(400, "Mínimo de 5 amostras para regressão (NBR 14653-2)")

    try:
        resultado = calcular_regressao(req.amostras)
        return resultado
    except Exception as e:
        raise HTTPException(422, str(e))
