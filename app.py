import base64
import json
from pathlib import Path

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================

st.set_page_config(
    page_title="Visor de precios y abastecimiento agroalimentario",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR        = Path("datos")
RUTA_LINEAS     = BASE_DIR / "lineas_abastecimiento.parquet"
RUTA_MUNICIPIOS = BASE_DIR / "municipios_ligeros.parquet"
RUTA_LOGO       = BASE_DIR / "MDS-245-ES.jpg"
RUTA_LINEAS_SQL = RUTA_LINEAS.as_posix()

DEPTOS_RAPE = {
    "BOGOTÁ", "BOGOTÁ, D.C.", "BOGOTA", "BOGOTA D.C.", "BOGOTÁ D.C.",
    "CUNDINAMARCA", "META", "BOYACÁ", "BOYACA", "TOLIMA"
}

MAX_FILAS_TABLA_DEFAULT = 300
MAX_LINEAS_MAPA_DEFAULT = 600
MAX_LINEAS_MAPA_MAX     = 1500

# =========================================================
# KEEP-ALIVE — evita que Streamlit Cloud duerma el app
# Hace un ping silencioso cada 45 minutos
# =========================================================

st.markdown("""
<script>
(function keepAlive() {
    setInterval(function() {
        fetch(window.location.href, {method: 'GET', cache: 'no-store'})
            .catch(function() {});
    }, 45 * 60 * 1000);
})();
</script>
""", unsafe_allow_html=True)

# =========================================================
# ESTILO
# =========================================================

st.markdown("""
<style>
    .stApp { background-color: #0F1116; color: #E8EDF5; }
    [data-testid="stHeader"] { background: rgba(0,0,0,0); }
    .block-container {
        max-width: 100%;
        padding-top: 0.8rem; padding-bottom: 1rem;
        padding-left: 1.1rem; padding-right: 1.1rem;
    }
    h1, h2, h3, h4 { color: #E8EDF5 !important; margin-bottom: 0.2rem; }
    .panel {
        background: #171A21; border: 1px solid #2B3240;
        border-radius: 12px; padding: 0.85rem 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.18);
    }
    .panel-title { color: #F2F5FA; font-size: 1rem; font-weight: 600; margin-bottom: 0.65rem; }
    .metric-card {
        background: #171A21; border: 1px solid #2B3240;
        border-radius: 12px; padding: 0.8rem 1rem;
        text-align: center; margin-bottom: 0.8rem;
    }
    .metric-label { color: #9EABC0; font-size: 0.82rem; margin-bottom: 0.35rem; }
    .metric-value { color: #FFFFFF; font-size: 2rem; font-weight: 700; line-height: 1.05; }
    .metric-small { color: #C7D0DD; font-size: 0.8rem; margin-top: 0.25rem; }
    .legend-item { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; font-size: 0.9rem; color: #D8E0EA; }
    .legend-box { width: 14px; height: 14px; border-radius: 3px; border: 1px solid rgba(255,255,255,0.15); }
    .small-note { color: #99A7BC; font-size: 0.82rem; line-height: 1.45; }
    .method-note {
        background: #171A21; border: 1px solid #2B3240;
        border-left: 4px solid #4DA3FF; border-radius: 10px;
        padding: 0.8rem 1rem; color: #C7D0DD;
        font-size: 0.86rem; line-height: 1.55; margin-top: 0.8rem;
    }
    .filter-wrap {
        background: #171A21; border: 1px solid #2B3240;
        border-radius: 12px; padding: 0.65rem 0.9rem 0.15rem 0.9rem;
        margin-bottom: 0.85rem;
    }
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background-color: #12161D !important; border-color: #2B3240 !important;
    }
    div[data-baseweb="tag"] { background-color: #243042 !important; }
    [data-testid="stDateInputField"] { background-color: #12161D !important; }
    [data-testid="stPlotlyChart"], [data-testid="stDeckGlJsonChart"] {
        background: #171A21; border: 1px solid #2B3240;
        border-radius: 12px; padding: 0.45rem;
    }
    .stDataFrame, div[data-testid="stTable"] {
        background: #171A21; border-radius: 12px;
        border: 1px solid #2B3240; padding: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# FUNCIONES AUXILIARES
# =========================================================

def normalizar_codigo_5(valor):
    if pd.isna(valor):
        return np.nan
    s = str(valor).strip().replace("'", "")
    if s.endswith(".0"):
        s = s[:-2]
    if s == "":
        return np.nan
    return s.zfill(5) if s.isdigit() else s

def normalizar_texto(valor):
    if pd.isna(valor):
        return ""
    return str(valor).strip()

def formatear_cop(valor):
    if pd.isna(valor):
        return "Sin dato"
    return f"$ {valor:,.0f}"

def formatear_ton(valor):
    if pd.isna(valor):
        return "Sin dato"
    return f"{valor:,.1f}"

def norm_serie(s):
    if len(s) == 0:
        return s
    s = s.fillna(0)
    if s.max() == s.min():
        return pd.Series(np.ones(len(s)), index=s.index)
    return (s - s.min()) / (s.max() - s.min())

def obtener_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return 0.0

def construir_sankey(sankey_top):
    nodos_origen  = sankey_top["municipio_origen"].astype(str).tolist()
    nodos_destino = sankey_top["central_mayorista"].astype(str).tolist()
    nodos = list(dict.fromkeys(nodos_origen + nodos_destino))
    idx   = {n: i for i, n in enumerate(nodos)}
    valores      = sankey_top["toneladas_total"].astype(float).tolist()
    total_sankey = sum(valores) if valores else 0
    porcentajes  = [(v / total_sankey) * 100 if total_sankey > 0 else 0 for v in valores]
    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(pad=12, thickness=16, line=dict(color="gray", width=0.5), label=nodos),
        link=dict(
            source=[idx[s] for s in sankey_top["municipio_origen"].astype(str)],
            target=[idx[t] for t in sankey_top["central_mayorista"].astype(str)],
            value=valores, customdata=porcentajes,
            hovertemplate=(
                "Origen: %{source.label}<br>Central mayorista: %{target.label}<br>"
                "Toneladas: %{value:,.1f}<br>Participación: %{customdata:.1f}%<extra></extra>"
            )
        )
    )])
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#171A21", plot_bgcolor="#171A21",
        font=dict(color="#E8EDF5", size=11),
        margin=dict(l=10, r=10, t=10, b=10), height=430
    )
    return fig

# =========================================================
# CARGA DE MUNICIPIOS — una sola vez
# =========================================================

@st.cache_resource(show_spinner=False)
def cargar_municipios(mtime_mun):
    mun = gpd.read_parquet(RUTA_MUNICIPIOS)
    mun = mun.copy()
    for col in ["MpCodigo","CODIGO_MUNICIPIO"]:
        if col in mun.columns:
            mun["codigo_origen"] = mun[col].apply(normalizar_codigo_5)
            break
    for col in ["nombre_municipio","MpNombre","MUNICIPIO","Nombre"]:
        if col in mun.columns:
            mun["nombre_municipio"] = mun[col].apply(normalizar_texto)
            break
    else:
        mun["nombre_municipio"] = ""
    for col in ["departamento","Depto","DEPARTAMENTO"]:
        if col in mun.columns:
            mun["departamento"] = mun[col].apply(normalizar_texto)
            break
    else:
        mun["departamento"] = ""
    return mun

# =========================================================
# CONEXIÓN DUCKDB — lee lineas_abastecimiento.parquet
# El nuevo parquet tiene: fecha_mes, grupo, rubro,
# central_mayorista, lon_central, lat_central,
# departamento_origen, cod_depto, municipio_origen,
# cod_municipio, lon_mun, lat_mun, departamento_geo,
# toneladas, dias_con_datos
# =========================================================

@st.cache_resource(show_spinner=False)
def get_duckdb_connection(mtime_lineas):
    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=4")
    con.execute(f"""
        CREATE OR REPLACE VIEW lineas AS
        SELECT
            CAST(fecha_mes AS DATE)                          AS fecha_mes,
            CAST(MONTH(CAST(fecha_mes AS DATE)) AS INTEGER)  AS mes,
            CASE WHEN MONTH(CAST(fecha_mes AS DATE)) BETWEEN 1 AND 6
                 THEN 'Primer semestre' ELSE 'Segundo semestre' END AS semestre,
            STRFTIME(CAST(fecha_mes AS DATE), '%Y-%m')       AS etiqueta_mes,
            TRIM(CAST(grupo AS VARCHAR))                     AS grupo,
            TRIM(CAST(rubro AS VARCHAR))                     AS rubro,
            TRIM(CAST(central_mayorista AS VARCHAR))         AS central_mayorista,
            CAST(lon_central AS DOUBLE)                      AS lon_central,
            CAST(lat_central AS DOUBLE)                      AS lat_central,
            UPPER(TRIM(CAST(departamento_origen AS VARCHAR))) AS departamento_origen,
            TRIM(CAST(cod_depto AS VARCHAR))                 AS cod_depto,
            UPPER(TRIM(CAST(municipio_origen AS VARCHAR)))   AS municipio_origen,
            TRIM(CAST(cod_municipio AS VARCHAR))             AS cod_municipio,
            CAST(lon_mun AS DOUBLE)                          AS lon_mun,
            CAST(lat_mun AS DOUBLE)                          AS lat_mun,
            CAST(toneladas AS DOUBLE)                        AS toneladas,
            CAST(dias_con_datos AS DOUBLE)                   AS dias_con_datos
        FROM read_parquet('{RUTA_LINEAS_SQL}')
    """)
    return con

@st.cache_data(show_spinner=False)
def consultar_catalogos(mtime_lineas):
    con = get_duckdb_connection(mtime_lineas)
    grupos    = con.execute("SELECT DISTINCT grupo FROM lineas WHERE grupo IS NOT NULL ORDER BY grupo").df()["grupo"].astype(str).tolist()
    rubros    = con.execute("SELECT DISTINCT rubro FROM lineas WHERE rubro IS NOT NULL ORDER BY rubro").df()["rubro"].astype(str).tolist()
    centrales = con.execute("SELECT DISTINCT central_mayorista FROM lineas WHERE central_mayorista IS NOT NULL ORDER BY central_mayorista").df()["central_mayorista"].astype(str).tolist()
    deptos    = con.execute("SELECT DISTINCT departamento_origen FROM lineas WHERE departamento_origen IS NOT NULL ORDER BY departamento_origen").df()["departamento_origen"].astype(str).tolist()
    rango     = con.execute("SELECT MIN(fecha_mes) AS fmin, MAX(fecha_mes) AS fmax FROM lineas").df()
    fecha_min = pd.to_datetime(rango.loc[0,"fmin"]).date()
    fecha_max = pd.to_datetime(rango.loc[0,"fmax"]).date()
    return grupos, rubros, centrales, deptos, fecha_min, fecha_max

# =========================================================
# CONSULTA PRINCIPAL — un solo escaneo DuckDB
# Soporta 3 niveles: global, por grupo, por rubro
# =========================================================

def construir_where(fecha_ini, fecha_fin, semestre_sel,
                    grupo_sel, rubro_sel, centrales_sel, deptos_sel):
    clauses = ["fecha_mes BETWEEN ? AND ?"]
    params  = [fecha_ini.isoformat(), fecha_fin.isoformat()]
    if semestre_sel == "Primer semestre":
        clauses.append("mes BETWEEN 1 AND 6")
    elif semestre_sel == "Segundo semestre":
        clauses.append("mes BETWEEN 7 AND 12")
    # Jerarquía: rubro > grupo > global
    if rubro_sel and rubro_sel != "Todos":
        clauses.append("rubro = ?"); params.append(rubro_sel)
    elif grupo_sel and grupo_sel != "Todos":
        clauses.append("grupo = ?"); params.append(grupo_sel)
    if centrales_sel:
        clauses.append(f"central_mayorista IN ({','.join(['?']*len(centrales_sel))})")
        params.extend(list(centrales_sel))
    if deptos_sel:
        clauses.append(f"departamento_origen IN ({','.join(['?']*len(deptos_sel))})")
        params.extend(list(deptos_sel))
    return " AND ".join(clauses), params

@st.cache_data(show_spinner=False)
def consultar_todo(fecha_ini, fecha_fin, semestre_sel,
                   grupo_sel, rubro_sel, centrales_sel, deptos_sel,
                   mtime_lineas):
    con = get_duckdb_connection(mtime_lineas)
    where_sql, params = construir_where(
        fecha_ini, fecha_fin, semestre_sel,
        grupo_sel, rubro_sel, centrales_sel, deptos_sel
    )
    where_total_sql, params_total = construir_where(
        fecha_ini, fecha_fin, semestre_sel,
        grupo_sel, rubro_sel, (), ()
    )
    deptos_rape       = list(DEPTOS_RAPE)
    placeholders_rape = ",".join(["?"]*len(deptos_rape))

    query = f"""
        WITH base AS (
            SELECT * FROM lineas WHERE {where_sql}
        ),
        base_total AS (
            SELECT * FROM lineas WHERE {where_total_sql}
        ),
        metricas AS (
            SELECT
                SUM(toneladas)                  AS volumen_total_filtro,
                COUNT(DISTINCT cod_municipio)   AS municipios_activos,
                COUNT(DISTINCT central_mayorista) AS centrales_activas
            FROM base
        ),
        metricas_total AS (
            SELECT SUM(toneladas) AS volumen_total_total FROM base_total
        ),
        metricas_rape AS (
            SELECT SUM(toneladas) AS volumen_total_rape
            FROM base_total
            WHERE departamento_origen IN ({placeholders_rape})
        ),
        ranking AS (
            SELECT
                CASE
                    WHEN UPPER(municipio_origen) = 'UNE'     THEN '25845'
                    WHEN UPPER(municipio_origen) = 'FÓMEQUE' THEN '25279'
                    WHEN UPPER(municipio_origen) = 'FOMEQUE' THEN '25279'
                    WHEN UPPER(municipio_origen) = 'CERRITO' THEN '68162'
                    ELSE cod_municipio
                END AS cod_municipio,
                MAX(municipio_origen)            AS municipio_origen,
                MAX(departamento_origen)         AS departamento_origen,
                SUM(toneladas)                   AS toneladas_total,
                COUNT(DISTINCT etiqueta_mes)     AS meses_participacion,
                SUM(dias_con_datos)              AS dias_con_datos
            FROM base
            GROUP BY 1
        ),
        serie AS (
            SELECT etiqueta_mes,
                SUM(toneladas) AS toneladas_total
            FROM base GROUP BY 1 ORDER BY 1
        ),
        flujos AS (
            SELECT
                CASE
                    WHEN UPPER(municipio_origen) = 'UNE'     THEN '25845'
                    WHEN UPPER(municipio_origen) = 'FÓMEQUE' THEN '25279'
                    WHEN UPPER(municipio_origen) = 'FOMEQUE' THEN '25279'
                    WHEN UPPER(municipio_origen) = 'CERRITO' THEN '68162'
                    ELSE cod_municipio
                END AS cod_municipio,
                municipio_origen,
                departamento_origen,
                central_mayorista,
                AVG(lon_mun)     AS lon_orig,
                AVG(lat_mun)     AS lat_orig,
                AVG(lon_central) AS lon_dest,
                AVG(lat_central) AS lat_dest,
                SUM(toneladas)   AS toneladas_total
            FROM base
            WHERE lon_mun IS NOT NULL AND lat_mun IS NOT NULL
              AND lon_central IS NOT NULL AND lat_central IS NOT NULL
            GROUP BY 1,2,3,4
            ORDER BY toneladas_total DESC
        ),
        sankey AS (
            SELECT municipio_origen, central_mayorista,
                SUM(toneladas) AS toneladas_total
            FROM base GROUP BY 1,2 ORDER BY toneladas_total DESC
        )
        SELECT 'metricas' AS _t, TO_JSON(metricas)       AS _j FROM metricas
        UNION ALL SELECT 'total',   TO_JSON(metricas_total) FROM metricas_total
        UNION ALL SELECT 'rape',    TO_JSON(metricas_rape)  FROM metricas_rape
        UNION ALL SELECT 'ranking', TO_JSON(ranking)        FROM ranking
        UNION ALL SELECT 'serie',   TO_JSON(serie)          FROM serie
        UNION ALL SELECT 'flujos',  TO_JSON(flujos)         FROM flujos
        UNION ALL SELECT 'sankey',  TO_JSON(sankey)         FROM sankey
    """

    resultado = con.execute(query, params + params_total + deptos_rape).df()

    def ex(nombre):
        filas = resultado[resultado["_t"] == nombre]["_j"].tolist()
        if not filas:
            return pd.DataFrame()
        return pd.DataFrame([json.loads(f) for f in filas])

    return ex("metricas"), ex("total"), ex("rape"), ex("ranking"), ex("serie"), ex("flujos"), ex("sankey")

# =========================================================
# CARGA INICIAL
# =========================================================

mtime_lineas = obtener_mtime(RUTA_LINEAS)
mtime_mun    = obtener_mtime(RUTA_MUNICIPIOS)

municipios = cargar_municipios(mtime_mun)
grupos, rubros, centrales, deptos, fecha_min_global, fecha_max_global = consultar_catalogos(mtime_lineas)

codigos_validos = set(municipios["codigo_origen"].dropna().astype(str).unique())

# =========================================================
# ENCABEZADO
# =========================================================

with open(RUTA_LOGO, "rb") as _f:
    _logo_b64 = base64.b64encode(_f.read()).decode()

st.markdown(f"""
<div style="background:#FFFFFF;border-bottom:2px solid #2B3240;padding:10px 20px;
    margin-bottom:0.85rem;margin-left:-1.1rem;margin-right:-1.1rem;margin-top:-0.8rem;
    display:flex;align-items:center;gap:18px;">
    <img src="data:image/jpeg;base64,{_logo_b64}" style="height:64px;width:auto;flex-shrink:0;"/>
    <div>
        <div style="font-size:1.85rem;font-weight:700;color:#111827;letter-spacing:-0.01em;line-height:1.1;">
            Visor de precios y abastecimiento agroalimentario
        </div>
        <div style="font-size:0.92rem;color:#6B7280;margin-top:4px;">
            Lectura territorial de flujos y eficiencia relativa de municipios de origen
            por producto y central mayorista · SIPSA–DANE 2020–2026
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# FILTROS — jerarquía: grupo → rubro → demás
# =========================================================

st.markdown('<div class="filter-wrap">', unsafe_allow_html=True)
f1, f2, f3, f4, f5, f6 = st.columns([1.1, 1.3, 1.5, 1.0, 1.0, 1.4])

with f1:
    grupo_sel = st.selectbox("Grupo", ["Todos"] + grupos, index=0)

with f2:
    # Filtrar rubros según grupo seleccionado
    if grupo_sel != "Todos":
        con_tmp = get_duckdb_connection(mtime_lineas)
        rubros_filtrados = con_tmp.execute(
            "SELECT DISTINCT rubro FROM lineas WHERE grupo = ? AND rubro IS NOT NULL ORDER BY rubro",
            [grupo_sel]
        ).df()["rubro"].astype(str).tolist()
    else:
        rubros_filtrados = rubros
    rubro_sel = st.selectbox("Rubro", ["Todos"] + rubros_filtrados, index=0)

with f3:
    centrales_sel = st.multiselect("Central mayorista", options=centrales, default=[])

with f4:
    semestre_sel = st.selectbox("Periodo", ["Todos","Primer semestre","Segundo semestre"], index=0)

with f5:
    deptos_sel = st.multiselect("Depto. origen", deptos, default=[])

with f6:
    rango = st.date_input("Fechas", value=(fecha_min_global, fecha_max_global),
                          min_value=fecha_min_global, max_value=fecha_max_global)

st.markdown("</div>", unsafe_allow_html=True)

if isinstance(rango, tuple) and len(rango) == 2:
    fecha_ini, fecha_fin = rango
else:
    fecha_ini, fecha_fin = fecha_min_global, fecha_max_global

centrales_tuple = tuple(centrales_sel)
deptos_tuple    = tuple(deptos_sel)
nivel_sel       = rubro_sel if rubro_sel != "Todos" else (grupo_sel if grupo_sel != "Todos" else "Todos los productos")

# =========================================================
# CONSULTA PRINCIPAL
# =========================================================

(
    metricas_df, total_df, rape_df,
    ranking_base, serie_df, flujos_df, sankey_df
) = consultar_todo(
    fecha_ini, fecha_fin, semestre_sel,
    grupo_sel, rubro_sel,
    centrales_tuple, deptos_tuple,
    mtime_lineas
)

# =========================================================
# ESCALARES
# =========================================================

def _sf(df, col):
    try: v = df.loc[0, col]; return float(v) if pd.notna(v) else 0.0
    except: return 0.0

def _si(df, col):
    try: v = df.loc[0, col]; return int(v) if pd.notna(v) else 0
    except: return 0

volumen_filtro = _sf(metricas_df, "volumen_total_filtro")
mun_activos    = _si(metricas_df, "municipios_activos")
cent_activas   = _si(metricas_df, "centrales_activas")
volumen_total  = _sf(total_df,    "volumen_total_total")
volumen_rape   = _sf(rape_df,     "volumen_total_rape")

# =========================================================
# RANKING E ÍNDICE DE EFICIENCIA
# =========================================================

if not ranking_base.empty:
    ranking = ranking_base.copy()
    total_meses = max(serie_df["etiqueta_mes"].nunique(), 1) if not serie_df.empty else 1

    ranking["part_filtro_pct"] = np.where(volumen_filtro > 0, (ranking["toneladas_total"] / volumen_filtro) * 100, 0)
    ranking["part_total_pct"]  = np.where(volumen_total  > 0, (ranking["toneladas_total"] / volumen_total)  * 100, 0)
    ranking["part_rape_pct"]   = np.where(volumen_rape   > 0, (ranking["toneladas_total"] / volumen_rape)   * 100, 0)
    ranking["frec_relativa"]   = ranking["meses_participacion"] / total_meses

    ranking["score_volumen"]   = norm_serie(ranking["toneladas_total"])
    ranking["score_actividad"] = norm_serie(ranking["meses_participacion"])
    ranking["indice_eficiencia"] = (
        ranking["score_volumen"] * 0.60 +
        ranking["score_actividad"] * 0.40
    )
    ranking = ranking.sort_values(
        ["indice_eficiencia","toneladas_total"], ascending=False
    ).reset_index(drop=True)
    ranking["ranking"] = ranking.index + 1

    top30_codigos = set(
        ranking.sort_values("toneladas_total", ascending=False)
        .head(30)["cod_municipio"].astype(str).tolist()
    )

    # Flujos
    if not flujos_df.empty:
        flujos_df = flujos_df.sort_values("toneladas_total", ascending=False).head(MAX_LINEAS_MAPA_DEFAULT).copy()
        vmin = flujos_df["toneladas_total"].min()
        vmax = flujos_df["toneladas_total"].max()
        flujos_df["ancho_linea"] = 2 + 10 * ((flujos_df["toneladas_total"] - vmin) / (vmax - vmin + 1e-9))
        flujos_df["ton_fmt"]  = flujos_df["toneladas_total"].map(formatear_ton)

    # Sankey — top 12 municipios
    if not sankey_df.empty:
        top_mun = (
            sankey_df.groupby("municipio_origen", as_index=False)["toneladas_total"].sum()
            .sort_values("toneladas_total", ascending=False).head(12)["municipio_origen"].tolist()
        )
        sankey_top = sankey_df[sankey_df["municipio_origen"].isin(top_mun)].copy()
    else:
        sankey_top = pd.DataFrame(columns=["municipio_origen","central_mayorista","toneladas_total"])
else:
    ranking    = pd.DataFrame()
    top30_codigos = set()
    flujos_df  = pd.DataFrame()
    sankey_top = pd.DataFrame(columns=["municipio_origen","central_mayorista","toneladas_total"])

# =========================================================
# MAPA — polígonos con top30 en morado
# =========================================================

top30_list = municipios["codigo_origen"].astype(str).isin(top30_codigos).tolist()
mun_web    = municipios[["nombre_municipio","departamento","codigo_origen","geometry"]].copy()
mun_web["fill_color"] = [[110,68,255,150] if t else [40,48,62,18] for t in top30_list]
mun_web["line_color"] = [[170,130,255,240] if t else [100,110,125,60] for t in top30_list]
mun_web["tipo"]       = "Municipio"
mun_web["d1"] = "Nombre: "       + mun_web["nombre_municipio"].fillna("Sin nombre").astype(str)
mun_web["d2"] = "Departamento: " + mun_web["departamento"].fillna("Sin dato").astype(str)
mun_web["d3"] = "Código: "       + mun_web["codigo_origen"].fillna("").astype(str)
mun_web["d4"] = ""

geojson_mun = json.loads(
    mun_web[["fill_color","line_color","tipo","d1","d2","d3","d4","geometry"]].to_json()
)

# Centrales — puntos destino únicos del resultado
if not flujos_df.empty:
    centrales_pts = (
        flujos_df.groupby("central_mayorista", as_index=False)
        .agg(lon=("lon_dest","first"), lat=("lat_dest","first"))
    )
    centrales_pts = centrales_pts[centrales_pts["lon"].notna()].copy()
else:
    centrales_pts = pd.DataFrame(columns=["central_mayorista","lon","lat"])

# =========================================================
# LAYOUT PRINCIPAL
# =========================================================

@st.fragment
def render_principal(
    volumen_filtro, mun_activos, cent_activas,
    geojson_mun, flujos_df, centrales_pts, serie_df,
    sankey_top, nivel_sel
):
    left_col, center_col, right_col = st.columns([1.05, 3.8, 1.45], gap="small")

    # ── Métricas + leyenda ──
    with left_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Indicadores principales</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Toneladas abastecidas</div>
            <div class="metric-value">{volumen_filtro:,.0f}</div>
            <div class="metric-small">Periodo filtrado</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Municipios origen activos</div>
            <div class="metric-value">{mun_activos:,}</div>
            <div class="metric-small">Con flujo válido</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Centrales activas</div>
            <div class="metric-value">{cent_activas}</div>
            <div class="metric-small">Bajo filtros actuales</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="panel-title">Leyenda</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="legend-item"><span class="legend-box" style="background:#6E44FF;"></span>Top 30 abastecedores</div>
        <div class="legend-item"><span class="legend-box" style="background:#F5B041;"></span>Arcos de flujo OD</div>
        <div class="legend-item"><span class="legend-box" style="background:#00D2FF;"></span>Central mayorista</div>
        <div class="small-note" style="margin-top:0.75rem;">
            Sin filtro de rubro los arcos muestran el flujo agregado por municipio-central.
            Al seleccionar un rubro o grupo se filtra automáticamente.
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Mapa + serie ──
    with center_col:
        st.markdown('<div class="panel-title">Mapa de flujos de abastecimiento</div>', unsafe_allow_html=True)

        layers = [pdk.Layer(
            "GeoJsonLayer", data=geojson_mun,
            stroked=True, filled=True, extruded=False,
            get_fill_color="properties.fill_color",
            get_line_color="properties.line_color",
            line_width_min_pixels=1.0, pickable=True, auto_highlight=True
        )]

        if not flujos_df.empty:
            layers.append(pdk.Layer(
                "ArcLayer", data=flujos_df,
                get_source_position=["lon_orig","lat_orig"],
                get_target_position=["lon_dest","lat_dest"],
                get_source_color=[245,176,65,190], get_target_color=[0,210,255,190],
                get_width="ancho_linea", width_scale=1, width_min_pixels=1,
                pickable=True, auto_highlight=True
            ))

        if not centrales_pts.empty:
            layers.append(pdk.Layer(
                "ScatterplotLayer", data=centrales_pts,
                get_position="[lon, lat]", get_radius=13500,
                get_fill_color=[0,210,255,190], get_line_color=[170,245,255,255],
                line_width_min_pixels=2, pickable=True
            ))

        deck = pdk.Deck(
            layers=layers,
            initial_view_state=pdk.ViewState(latitude=4.5, longitude=-74.1, zoom=4.6, pitch=0),
            tooltip={
                "html": "<b>{tipo}</b><br/>{d1}<br/>{d2}<br/>{d3}<br/>{d4}",
                "style": {"backgroundColor":"rgba(18,22,29,0.95)","color":"#F5F7FA","fontSize":"12px"}
            },
            map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
        )
        st.pydeck_chart(deck, use_container_width=True)

        # Serie mensual de toneladas
        st.markdown('<div class="panel-title" style="margin-top:0.65rem;">Serie mensual de toneladas abastecidas</div>', unsafe_allow_html=True)
        if not serie_df.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=serie_df["etiqueta_mes"], y=serie_df["toneladas_total"],
                name="Toneladas", marker_color="#4DA3FF",
                hovertemplate="Mes: %{x}<br>Toneladas: %{y:,.1f}<extra></extra>"
            ))
            fig.update_layout(
                template="plotly_dark", paper_bgcolor="#171A21", plot_bgcolor="#171A21",
                margin=dict(l=15, r=15, t=10, b=10), height=280,
                xaxis=dict(showgrid=False),
                yaxis=dict(title="Toneladas", gridcolor="#2B3240"),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay datos de serie mensual con los filtros actuales.")

    # ── Sankey ──
    with right_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown(f'<div class="panel-title">Flujos hacia centrales mayoristas<br><span style="font-weight:400;font-size:0.82rem;color:#AEB9C9;">{nivel_sel}</span></div>', unsafe_allow_html=True)
        if not sankey_top.empty:
            st.plotly_chart(construir_sankey(sankey_top), use_container_width=True)
        else:
            st.info("No hay datos suficientes para mostrar.")
        st.markdown('</div>', unsafe_allow_html=True)


render_principal(
    volumen_filtro=volumen_filtro,
    mun_activos=mun_activos,
    cent_activas=cent_activas,
    geojson_mun=geojson_mun,
    flujos_df=flujos_df,
    centrales_pts=centrales_pts,
    serie_df=serie_df,
    sankey_top=sankey_top,
    nivel_sel=nivel_sel,
)

# =========================================================
# TABLA — fragmento independiente
# =========================================================

@st.fragment
def render_tabla(ranking, volumen_filtro, volumen_total, volumen_rape, max_filas):
    st.markdown('<div class="panel-title" style="margin-top:0.8rem;">Tabla consolidada de análisis</div>', unsafe_allow_html=True)

    if not ranking.empty:
        tabla = ranking[[
            "ranking","municipio_origen","departamento_origen",
            "toneladas_total","meses_participacion",
            "part_filtro_pct","part_total_pct","part_rape_pct","indice_eficiencia"
        ]].copy()

        tabla.columns = [
            "Ranking","Municipio origen","Departamento origen",
            "Toneladas acumuladas","Meses activos",
            "Participación en filtro","Participación total",
            "Participación RAPE","Índice de eficiencia"
        ]

        cols_ord = [
            "Ranking","Toneladas acumuladas","Meses activos",
            "Participación en filtro","Participación total","Índice de eficiencia"
        ]
        col_s, col_d = st.columns([2, 1])
        with col_s:
            col_orden = st.selectbox("Ordenar por", cols_ord, index=0,
                                     key="tabla_col", label_visibility="collapsed")
        with col_d:
            dir_orden = st.radio("Dirección", ["↓ Mayor a menor","↑ Menor a mayor"],
                                 index=0, horizontal=True, key="tabla_dir",
                                 label_visibility="collapsed")

        asc = dir_orden == "↑ Menor a mayor"
        tabla = tabla.sort_values(col_orden, ascending=asc).reset_index(drop=True)
        if col_orden != "Ranking":
            tabla["Ranking"] = range(1, len(tabla) + 1)

        tabla_fmt = tabla.copy()
        tabla_fmt["Toneladas acumuladas"]    = tabla_fmt["Toneladas acumuladas"].map("{:,.1f}".format)
        tabla_fmt["Participación en filtro"] = tabla_fmt["Participación en filtro"].map("{:.1f}%".format)
        tabla_fmt["Participación total"]     = tabla_fmt["Participación total"].map("{:.1f}%".format)
        tabla_fmt["Participación RAPE"]      = tabla_fmt["Participación RAPE"].map("{:.1f}%".format)
        tabla_fmt["Índice de eficiencia"]    = tabla_fmt["Índice de eficiencia"].map("{:.2f}".format)

        st.dataframe(tabla_fmt.head(max_filas), use_container_width=True,
                     hide_index=True, height=420)
    else:
        st.info("No hay información disponible para la tabla consolidada.")

    st.markdown("""
    <div class="method-note">
        <b>Cómo se calcula el índice de eficiencia:</b><br>
        El índice compara municipios de origen dentro del subconjunto filtrado.
        Combina dos dimensiones normalizadas: volumen acumulado abastecido (60%)
        y frecuencia de participación mensual (40%). No incluye precio porque
        al trabajar con múltiples rubros el precio no es comparable entre productos.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:0.8rem;padding-top:0.6rem;border-top:1px solid #2B3240;
        color:#8FA0B7;font-size:0.8rem;text-align:center;">
        Fuente: Sistema de Información de Precios y Abastecimiento del Sector Agropecuario (SIPSA) · DANE · 2020–2026
    </div>
    """, unsafe_allow_html=True)


render_tabla(
    ranking=ranking,
    volumen_filtro=volumen_filtro,
    volumen_total=volumen_total,
    volumen_rape=volumen_rape,
    max_filas=MAX_FILAS_TABLA_DEFAULT,
)
