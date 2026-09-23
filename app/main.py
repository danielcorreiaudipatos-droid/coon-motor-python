from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import numpy as np
from scipy import stats
import math, json
from .regressao import calcular_regressao

app = FastAPI(title="CO.ON Motor Estatístico", version="1.0.0")

def _sanitize(obj):
    """Converte tipos numpy e substitui inf/nan — json padrão falha com esses valores."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if (math.isnan(obj) or math.isinf(obj)) else float(obj)
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    return obj

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
        return JSONResponse(content=_sanitize(resultado))
    except Exception as e:
        raise HTTPException(422, str(e))
