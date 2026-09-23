"""
Motor de Regressão Linear Múltipla — NBR 14653-2
Método dos Mínimos Quadrados Ordinários (MQO)
"""
import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Optional


@dataclass
class ResultadoRegressao:
    # Coeficientes
    coeficientes: dict
    intercepto: float

    # Qualidade do ajuste
    r2: float
    r2_ajustado: float
    erro_padrao: float

    # Teste F (significância global)
    f_calculado: float
    f_critico: float
    significativo: bool

    # Estimativa central
    valor_unit: float
    valor_total: Optional[float]

    # Intervalo de confiança 80% (NBR 14653)
    ic_inf: float
    ic_sup: float
    amplitude_ic: float

    # Graus NBR 14653-2
    grau_fund: str   # I, II ou III
    grau_prec: str   # I, II ou III

    # Diagnóstico
    n_amostras: int
    n_variaveis: int
    pressupostos: dict


def calcular_regressao(amostras) -> dict:
    n = len(amostras)

    # Variáveis independentes disponíveis
    tem_area = all(a.area_priv or a.area_total for a in amostras)

    variaveis = []
    X_cols = []

    if tem_area:
        areas = np.array([float(a.area_priv or a.area_total) for a in amostras])
        X_cols.append(areas)
        variaveis.append("area")

    # Variável dependente: valor unitário (R$/m²) se tiver área, senão valor total
    if tem_area:
        y = np.array([a.valor / float(a.area_priv or a.area_total) for a in amostras])
    else:
        y = np.array([a.valor for a in amostras])

    k = len(variaveis)  # número de variáveis independentes

    if k == 0:
        # Sem variáveis — apenas média com IC
        media = float(np.mean(y))
        dp = float(np.std(y, ddof=1))
        t_crit = float(stats.t.ppf(0.90, df=n - 1))  # 80% bilateral = 10% cada cauda
        ic_inf = media - t_crit * dp / np.sqrt(n)
        ic_sup = media + t_crit * dp / np.sqrt(n)

        return _formatar(
            coef={}, intercepto=media, r2=0.0, r2_adj=0.0,
            ep=dp, f_calc=0.0, f_crit=0.0, sig=False,
            val_unit=media, val_total=None,
            ic_inf=ic_inf, ic_sup=ic_sup,
            n=n, k=0, y=y, y_pred=np.full(n, media),
            variaveis=variaveis, amostras=amostras
        )

    # Matriz X com intercepto
    X = np.column_stack([np.ones(n)] + X_cols)

    # MQO: β = (X'X)^-1 X'y
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise ValueError(f"Sistema singular — verifique colinearidade: {e}")

    intercepto = float(beta[0])
    coef = {variaveis[i]: float(beta[i + 1]) for i in range(k)}

    # Valores ajustados e resíduos
    y_pred = X @ beta
    residuos = y - y_pred
    SQR = float(np.sum(residuos ** 2))        # soma quadrados resíduos
    SQT = float(np.sum((y - np.mean(y)) ** 2))  # soma quadrados total
    SQE = SQT - SQR

    r2 = 1 - SQR / SQT if SQT > 0 else 0.0
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - k - 1) if n > k + 1 else 0.0
    ep = float(np.sqrt(SQR / (n - k - 1))) if n > k + 1 else 0.0

    # Teste F global (proteção contra divisão por zero quando R²≈1)
    denom_f = SQR / (n - k - 1) if n > k + 1 else 0.0
    f_calc = float((SQE / k) / denom_f) if denom_f > 1e-10 else 9999.9999
    f_crit = float(stats.f.ppf(0.90, dfn=k, dfd=n - k - 1))  # α=10%
    sig = f_calc > f_crit

    # Estimativa no ponto médio das variáveis
    x_med = np.array([1.0] + [float(np.mean(col)) for col in X_cols])
    val_unit = float(x_med @ beta)
    val_total = None
    if tem_area:
        area_med = float(np.mean(areas))
        val_total = val_unit * area_med

    # IC 80% para previsão (NBR 14653 exige 80%)
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
        var_pred = ep ** 2 * (1 + x_med @ XtX_inv @ x_med)
        t_crit = float(stats.t.ppf(0.90, df=n - k - 1))
        ic_inf = val_unit - t_crit * np.sqrt(var_pred)
        ic_sup = val_unit + t_crit * np.sqrt(var_pred)
    except Exception:
        t_crit = float(stats.t.ppf(0.90, df=max(n - k - 1, 1)))
        ic_inf = val_unit - t_crit * ep
        ic_sup = val_unit + t_crit * ep

    return _formatar(
        coef=coef, intercepto=intercepto, r2=r2, r2_adj=r2_adj,
        ep=ep, f_calc=f_calc, f_crit=f_crit, sig=sig,
        val_unit=val_unit, val_total=val_total,
        ic_inf=ic_inf, ic_sup=ic_sup,
        n=n, k=k, y=y, y_pred=y_pred,
        variaveis=variaveis, amostras=amostras
    )


def _grau_fund(n: int, k: int) -> str:
    """Grau de fundamentação NBR 14653-2 Tabela 2"""
    if n >= 6 * (k + 1) and k >= 3:
        return "III"
    elif n >= 4 * (k + 1) and k >= 2:
        return "II"
    else:
        return "I"


def _grau_prec(amplitude_pct: float) -> str:
    """Grau de precisão NBR 14653-2 Tabela 3"""
    if amplitude_pct <= 30:
        return "III"
    elif amplitude_pct <= 50:
        return "II"
    else:
        return "I"


def _formatar(*, coef, intercepto, r2, r2_adj, ep, f_calc, f_crit, sig,
              val_unit, val_total, ic_inf, ic_sup, n, k, y, y_pred,
              variaveis, amostras) -> dict:

    amplitude_pct = ((ic_sup - ic_inf) / val_unit * 100) if val_unit else 100
    grau_fund = _grau_fund(n, k)
    grau_prec = _grau_prec(amplitude_pct)

    # Pressupostos básicos
    residuos = y - y_pred
    _, p_norm = stats.shapiro(residuos) if len(residuos) >= 3 else (0, 1.0)
    _, p_homo = stats.levene(residuos[:len(residuos)//2], residuos[len(residuos)//2:]) \
        if len(residuos) >= 4 else (0, 1.0)

    return {
        "coeficientes":   coef,
        "intercepto":     round(intercepto, 4),
        "r2":             round(r2, 4),
        "r2_ajustado":    round(r2_adj, 4),
        "erro_padrao":    round(ep, 4),
        "f_calculado":    round(f_calc, 4),
        "f_critico":      round(f_crit, 4),
        "significativo":  sig,
        "valor_unit":     round(val_unit, 2),
        "valor_total":    round(val_total, 2) if val_total else None,
        "ic_inf":         round(ic_inf, 2),
        "ic_sup":         round(ic_sup, 2),
        "amplitude_ic_pct": round(amplitude_pct, 1),
        "grau_fund":      grau_fund,
        "grau_prec":      grau_prec,
        "n_amostras":     n,
        "n_variaveis":    k,
        "pressupostos": {
            "normalidade_p":    round(float(p_norm), 4),
            "normalidade_ok":   p_norm > 0.05,
            "homogeneidade_p":  round(float(p_homo), 4),
            "homogeneidade_ok": p_homo > 0.05,
        }
    }
