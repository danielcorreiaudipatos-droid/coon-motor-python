from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Any
import numpy as np
import math, json, os, base64, io
import httpx
from .regressao import calcular_regressao

app = FastAPI(title="CO.ON Motor Estatístico", version="2.0.0")

CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY", "")
CLAUDE_MODEL   = "claude-haiku-4-5-20251001"   # rápido e barato para extração


# ── helpers ──────────────────────────────────────────────────────────────────

def _sanitize(obj):
    if isinstance(obj, dict):  return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [_sanitize(v) for v in obj]
    if isinstance(obj, np.bool_):    return bool(obj)
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, np.floating): return None if (math.isnan(obj) or math.isinf(obj)) else float(obj)
    if isinstance(obj, float):       return None if (math.isnan(obj) or math.isinf(obj)) else obj
    return obj

def _claude(messages: list, system: str = "") -> str:
    """Chama a API do Claude e retorna o texto da resposta."""
    if not CLAUDE_API_KEY:
        raise RuntimeError("CLAUDE_API_KEY não configurada no Motor")
    headers = {
        "x-api-key": CLAUDE_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {"model": CLAUDE_MODEL, "max_tokens": 2048, "messages": messages}
    if system:
        body["system"] = system
    resp = httpx.post("https://api.anthropic.com/v1/messages", headers=headers, json=body, timeout=60)
    resp.raise_for_status()
    return resp.json()["content"][0]["text"]

def _baixar_arquivo(url: str) -> bytes:
    r = httpx.get(url, timeout=30, follow_redirects=True)
    r.raise_for_status()
    return r.content

def _extrair_texto_pdf(conteudo: bytes) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(conteudo))
        partes = [p.extract_text() or "" for p in reader.pages]
        return "\n".join(partes)[:12000]   # limite de contexto
    except Exception as e:
        return f"[erro ao ler PDF: {e}]"

def _montar_conteudo_doc(doc: dict) -> list:
    """Retorna lista de blocos de conteúdo para a mensagem Claude."""
    url  = doc.get("url", "")
    tipo = doc.get("tipo", "outro")
    nome = doc.get("nome", "documento")
    try:
        dados = _baixar_arquivo(url)
    except Exception as e:
        return [{"type": "text", "text": f"[Arquivo {nome} indisponível: {e}]"}]

    mime = url.split("?")[0].lower()
    if mime.endswith(".pdf") or "pdf" in doc.get("nome", "").lower():
        texto = _extrair_texto_pdf(dados)
        return [{"type": "text", "text": f"=== {tipo.upper()}: {nome} ===\n{texto}"}]
    else:
        # imagem — envia como vision
        b64 = base64.standard_b64encode(dados).decode()
        ext = nome.rsplit(".", 1)[-1].lower() if "." in nome else "jpeg"
        mt  = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
               "webp": "image/webp", "gif": "image/gif"}.get(ext, "image/jpeg")
        return [
            {"type": "text",  "text": f"=== {tipo.upper()}: {nome} ==="},
            {"type": "image", "source": {"type": "base64", "media_type": mt, "data": b64}},
        ]


# ── modelos ───────────────────────────────────────────────────────────────────

class Amostra(BaseModel):
    id: str
    ordem: int
    area_total:  Optional[float] = None
    area_priv:   Optional[float] = None
    valor:       float
    endereco:    Optional[str]   = None
    fonte:       Optional[str]   = None
    data_ref:    Optional[str]   = None

class Laudo(BaseModel):
    id: str
    endereco: str
    cidade:   str
    uf:       str
    finalidade: str
    valor_final: Optional[float] = None
    grau_fund:   Optional[str]   = None
    grau_prec:   Optional[str]   = None

class RegressaoRequest(BaseModel):
    laudo:    Laudo
    amostras: list[Amostra]

class Documento(BaseModel):
    id:       str
    tipo:     str = "outro"
    nome:     Optional[str] = None
    url:      str
    extraido: int = 0

class ExtrairRequest(BaseModel):
    laudo:      Laudo
    documentos: list[Documento]

class Propriedade(BaseModel):
    endereco:    Optional[str]   = None
    cidade:      Optional[str]   = None
    uf:          Optional[str]   = None
    area_total:  Optional[float] = None
    area_priv:   Optional[float] = None
    quartos:     Optional[int]   = None
    vagas:       Optional[int]   = None
    finalidade:  Optional[str]   = None

class InferirRequest(BaseModel):
    laudo_id:    str
    propriedade: Propriedade

class ConferirRequest(BaseModel):
    laudo:      Laudo
    amostras:   list[Amostra]
    resultado:  Optional[dict[str, Any]] = None


# ── rotas ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"ok": True, "versao": "2.0.0"}


@app.post("/regressao")
def regressao(req: RegressaoRequest):
    if len(req.amostras) < 5:
        raise HTTPException(400, "Mínimo de 5 amostras para regressão (NBR 14653-2)")
    try:
        resultado = calcular_regressao(req.amostras)
        return JSONResponse(content=_sanitize(resultado))
    except Exception as e:
        raise HTTPException(422, str(e))


@app.post("/extrair")
def extrair(req: ExtrairRequest):
    """
    Baixa cada documento, extrai texto/imagem e pede ao Claude os campos do imóvel.
    Retorna dict com os campos encontrados.
    """
    if not req.documentos:
        raise HTTPException(400, "Nenhum documento enviado")

    # monta blocos de conteúdo com todos os documentos
    blocos: list = []
    for doc in req.documentos:
        blocos.extend(_montar_conteudo_doc(doc.model_dump()))

    blocos.append({"type": "text", "text": (
        "\n\nExtraia do(s) documento(s) acima os seguintes campos do imóvel avaliado. "
        "Responda SOMENTE com JSON válido, sem texto extra, com as chaves:\n"
        "endereco, numero, bairro, cep, cidade, uf, "
        "area_total, area_priv, area_terreno, "
        "quartos, suites, banheiros, vagas, andares, andar_unidade, "
        "matricula, inscricao_municipal, valor_venal, "
        "proprietario, cpf_cnpj, finalidade.\n"
        "Use null para campos não encontrados. Números como número, não string."
    )})

    system = (
        "Você é um extrator de dados imobiliários. Analisa documentos de avaliação "
        "(OS Caixa, matrícula, escritura, IPTU, alvará) e retorna JSON estruturado. "
        "Nunca inventa dados — se não encontrar, retorna null."
    )

    try:
        resposta = _claude([{"role": "user", "content": blocos}], system=system)
        # extrai o JSON da resposta
        inicio = resposta.find("{")
        fim    = resposta.rfind("}") + 1
        if inicio == -1:
            raise ValueError("Claude não retornou JSON")
        campos = json.loads(resposta[inicio:fim])
    except Exception as e:
        raise HTTPException(502, f"Erro na extração Claude: {e}")

    return JSONResponse(content=_sanitize(campos))


@app.post("/inferir")
def inferir(req: InferirRequest):
    """
    Recebe características do imóvel e retorna amostras de mercado sugeridas.
    """
    p = req.propriedade
    desc = (
        f"Imóvel em {p.endereco or 'endereço não informado'}, "
        f"{p.cidade or '?'}/{p.uf or '?'}. "
        f"Área total: {p.area_total or '?'} m², área privativa: {p.area_priv or '?'} m². "
        f"Quartos: {p.quartos or '?'}, vagas: {p.vagas or '?'}. "
        f"Finalidade: {p.finalidade or 'garantia'}."
    )

    prompt = (
        f"{desc}\n\n"
        "Sugira 6 amostras de mercado comparáveis para embasar a avaliação pelo método comparativo "
        "(NBR 14653-2). As amostras devem ser de imóveis similares na mesma cidade ou região, "
        "com valores de mercado realistas para o Brasil em 2025-2026.\n\n"
        "Responda SOMENTE com JSON válido:\n"
        '{"amostras": [{"endereco": "...", "area_total": 0.0, "area_priv": 0.0, '
        '"valor": 0.0, "fonte": "ZAP/VivaReal/OLX", "data_ref": "2025-MM-DD"}, ...]}'
    )

    system = (
        "Você é um avaliador imobiliário sênior. Sugere amostras de mercado plausíveis "
        "com base na localização e características do imóvel. Os valores devem refletir "
        "o mercado imobiliário brasileiro atual. Nunca retorna amostras com valor zero."
    )

    try:
        resposta = _claude([{"role": "user", "content": prompt}], system=system)
        inicio = resposta.find("{")
        fim    = resposta.rfind("}") + 1
        if inicio == -1:
            raise ValueError("Claude não retornou JSON")
        dados = json.loads(resposta[inicio:fim])
        amostras = dados.get("amostras", [])
    except Exception as e:
        raise HTTPException(502, f"Erro na inferência Claude: {e}")

    return JSONResponse(content=_sanitize({"amostras": amostras, "sugestoes": amostras}))


@app.post("/conferir")
def conferir(req: ConferirRequest):
    """
    Confere o laudo antes do envio ao SIMIL: verifica coerência das amostras,
    campos obrigatórios, qualidade da regressão e emite alertas.
    """
    laudo = req.laudo
    amostras = req.amostras
    resultado = req.resultado or {}

    # verificações determinísticas (sem Claude)
    alertas_auto: list[str] = []
    if len(amostras) < 5:
        alertas_auto.append(f"Apenas {len(amostras)} amostra(s) — NBR 14653-2 exige mínimo 5 para regressão III grau.")
    if not laudo.valor_final:
        alertas_auto.append("Valor final não calculado — rode a regressão antes de enviar.")
    r2 = resultado.get("r2")
    if r2 is not None and r2 < 0.64:
        alertas_auto.append(f"R² = {r2:.2%} está abaixo de 64% — grau de precisão será III (aceitável mas revise as amostras).")
    if resultado.get("amplitude_ic") and resultado["amplitude_ic"] > 50:
        alertas_auto.append(f"Amplitude do IC = {resultado['amplitude_ic']:.1f}% — intervalo muito largo, possíveis outliers.")

    # valores unitários para detectar outliers simples
    valores_unit: list[float] = []
    for a in amostras:
        area = a.area_priv or a.area_total
        if area and a.valor:
            valores_unit.append(a.valor / area)
    if len(valores_unit) >= 3:
        import statistics
        med = statistics.median(valores_unit)
        for i, vu in enumerate(valores_unit):
            if vu < med * 0.4 or vu > med * 2.5:
                alertas_auto.append(f"Amostra {i+1} tem valor/m² muito discrepante ({vu:,.0f} vs mediana {med:,.0f}) — verifique.")

    # resumo para o Claude analisar
    amostras_txt = "\n".join(
        f"  {i+1}. {a.endereco or 'sem endereço'} | {a.area_total or '?'} m² | R$ {a.valor:,.0f} | {a.fonte or '—'}"
        for i, a in enumerate(amostras)
    )
    resultado_txt = (
        f"Valor unitário: R$ {resultado.get('valor_unit', '?'):,}/m², "
        f"Valor total: R$ {resultado.get('valor_total', '?'):,}, "
        f"R²={resultado.get('r2', '?')}, Grau Fund.={resultado.get('grau_fund', '?')}, "
        f"Grau Prec.={resultado.get('grau_prec', '?')}"
    ) if resultado else "Regressão não calculada."

    prompt = (
        f"Laudo de avaliação imobiliária — {laudo.finalidade} — {laudo.endereco}, {laudo.cidade}/{laudo.uf}\n\n"
        f"Amostras ({len(amostras)}):\n{amostras_txt}\n\n"
        f"Resultado da regressão: {resultado_txt}\n\n"
        "Como perito avaliador, faça uma análise crítica rápida deste laudo antes do envio ao SIMIL Caixa. "
        "Aponte problemas relevantes: amostras suspeitas, valor final fora de mercado, inconsistências. "
        "Responda SOMENTE com JSON:\n"
        '{"nivel": "ok"|"aviso"|"critico", "alertas": ["..."], "sugestoes": ["..."]}'
    )

    system = (
        "Você é um perito avaliador sênior revisando laudos para a Caixa Econômica Federal. "
        "Seja direto e objetivo. Não invente problemas onde não existem. "
        "nivel=ok se tudo parece razoável, aviso se há pontos de atenção, critico se há erro grave."
    )

    try:
        resposta = _claude([{"role": "user", "content": prompt}], system=system)
        inicio = resposta.find("{")
        fim    = resposta.rfind("}") + 1
        if inicio == -1:
            raise ValueError("Claude não retornou JSON")
        analise = json.loads(resposta[inicio:fim])
    except Exception:
        # Claude indisponível — retorna só os alertas automáticos
        analise = {"nivel": "aviso" if alertas_auto else "ok", "alertas": [], "sugestoes": []}

    # mescla alertas automáticos com os do Claude
    todos_alertas = alertas_auto + (analise.get("alertas") or [])
    nivel = analise.get("nivel", "ok")
    if alertas_auto and nivel == "ok":
        nivel = "aviso"

    return JSONResponse(content=_sanitize({
        "nivel":     nivel,
        "alertas":   todos_alertas,
        "sugestoes": analise.get("sugestoes") or [],
    }))
