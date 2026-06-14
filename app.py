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

BASE_DIR         = Path("Datos")
RUTA_LINEAS      = BASE_DIR / "lineas_abastecimiento.parquet"
RUTA_MUNICIPIOS  = BASE_DIR / "municipios_ligeros.parquet"
RUTA_PRECIOS     = BASE_DIR / "precios_rubros.parquet"
RUTA_VARIEDADES       = BASE_DIR / "lineas_variedades.parquet"
RUTA_INTERNACIONALES  = BASE_DIR / "lineas_internacionales.parquet"
RUTA_LOGO             = BASE_DIR / "MDS-245-ES.jpg"
RUTA_LINEAS_SQL       = RUTA_LINEAS.as_posix()
RUTA_PRECIOS_SQL      = RUTA_PRECIOS.as_posix()
RUTA_VAR_SQL          = RUTA_VARIEDADES.as_posix()
RUTA_INTL_SQL         = RUTA_INTERNACIONALES.as_posix()

DEPTOS_RAPE = {
    "BOGOTÁ", "BOGOTÁ, D.C.", "BOGOTA", "BOGOTA D.C.", "BOGOTÁ D.C.",
    "CUNDINAMARCA", "META", "BOYACÁ", "BOYACA", "TOLIMA"
}

MAX_FILAS_TABLA = 300
MAX_LINEAS_MAPA = 600

# ── Etiquetas legibles para rubros ────────────────────────
RUBROS_LABEL = {
    "Acelga_espinaca":"Acelga y espinaca","Aguacate":"Aguacate","Ahuyama":"Ahuyama",
    "Ajo":"Ajo","Apio_perejil":"Apio y perejil","Arracacha":"Arracacha","Arroz":"Arroz",
    "Arveja":"Arveja","Banano":"Banano","Berenjena":"Berenjena",
    "Brocoli_coliflor":"Brócoli y coliflor","Calabacin_calabaza":"Calabacín y calabaza",
    "Carne_cerdo":"Carne de cerdo","Carne_pollo":"Carne de pollo","Carne_res":"Carne de res",
    "Carnes_grupo":"Carnes (grupo)","Cebolla_cabezona":"Cebolla cabezona",
    "Cebolla_larga":"Cebolla larga","Cerdo_en_pie":"Cerdo en pie","Chocolo":"Chócolo",
    "Cilantro":"Cilantro","Ciruela":"Ciruela","Curuba":"Curuba","Durazno":"Durazno",
    "Fresa":"Fresa","Frijol":"Fríjol","Frutas_otras_frescas":"Frutas frescas otras",
    "Gallina_en_pie":"Gallina en pie","Granadilla":"Granadilla",
    "Granos_cereales_grupo":"Granos y cereales (grupo)","Guanabana":"Guanábana",
    "Guayaba":"Guayaba","Gulupa_pitahaya":"Gulupa y pitahaya","Habichuela":"Habichuela",
    "Huevos":"Huevos","Kiwi":"Kiwi","Lacteos_huevos_grupo":"Lácteos y huevos (grupo)",
    "Leche":"Leche","Lechuga":"Lechuga","Lenteja_garbanzo":"Lenteja y garbanzo",
    "Limon":"Limón","Lulo":"Lulo","Maiz":"Maíz","Mandarina":"Mandarina","Mango":"Mango",
    "Manzana":"Manzana","Maracuya":"Maracuyá","Mariscos":"Mariscos",
    "Melon_patilla":"Melón y patilla","Mora":"Mora","Name":"Ñame","Naranja":"Naranja",
    "Panela":"Panela","Papa":"Papa","Papaya":"Papaya","Pepino":"Pepino","Pera":"Pera",
    "Pescado":"Pescado","Pimenton":"Pimentón","Pina":"Piña","Platano":"Plátano",
    "Procesados_grupo":"Procesados (grupo)","Quesos_cuajadas":"Quesos y cuajadas",
    "Remolacha":"Remolacha","Repollo":"Repollo","Res_en_pie":"Res en pie",
    "Tangelo":"Tangelo","Tomate":"Tomate","Tomate_de_arbol":"Tomate de árbol",
    "Tuberculos_otros":"Tubérculos otros","Uchuva":"Uchuva","Ulluco":"Ulluco",
    "Uva":"Uva","Verduras_otras_frescas":"Verduras frescas otras","Yuca":"Yuca",
    "Zanahoria":"Zanahoria",
}
LABEL_RUBROS = {v: k for k, v in RUBROS_LABEL.items()}
def label_rubro(c): return RUBROS_LABEL.get(c, c)
def codigo_rubro(l): return LABEL_RUBROS.get(l, l)

# ── Rubros priorizados SARA (37) ──────────────────────────
RUBROS_PRIORIZADOS_SARA = {
    "Aguacate","Ahuyama","Arracacha","Arroz","Arveja","Banano",
    "Calabacin_calabaza","Cebolla_cabezona","Cebolla_larga","Frijol",
    "Guayaba","Habichuela","Lechuga","Limon","Lulo","Mandarina","Mango",
    "Maracuya","Mora","Name","Naranja","Papa","Papaya","Pina","Platano","Tomate",
    "Tomate_de_arbol","Yuca","Zanahoria","Carne_cerdo","Carne_pollo",
    "Carne_res","Pescado","Huevos","Leche","Quesos_cuajadas","Panela","Maiz"
}

# ── Municipios priorizados SARA ────────────────────────────
MUNS_OFERTA = {
    '11001','15001','15047','15238','15407','15638','15646','15696',
    '15759','15763','15814','25151','25178','25181','25183','25214',
    '25260','25269','25279','25286','25290','25312','25322','25377',
    '25386','25430','25473','25535','25649','25743','25754','25769',
    '25793','25817','25843','25845','25873','25899','41001','41298',
    '41551','50001','50251','50287','50313','50400','50568','50573',
    '50590','50689','73001','73124','73268','73443'
}
MUNS_DEMANDA = {
    '11001','25126','25151','25175','25183','25269','25286','25290',
    '25297','25307','25320','25377','25386','25430','25438','25473',
    '25513','25662','25754','25843','25875','25899'
}
MUNS_AMBOS = MUNS_OFERTA & MUNS_DEMANDA
TERRITORIOS_FUNC = {
    "Bogotá, D.C.":     ['11001'],
    "Norte":            ['25126','25154','25175','25178','25181','25258','25293',
                         '25307','25312','25317','25320','25326','25372','25438',
                         '25513','25662','25875'],
    "Noroccidental":    ['25148','25214','25260','25269','25279','25335','25402',
                         '25407','25430','25473','25483','25535','25572','25599',
                         '25612','25754','25769','25779','25785','25793','25797',
                         '25815','25817','25823','25839','25841','25843','25845',
                         '25851','25857','25867','25871','25873','25875','25877',
                         '25885','25899'],
    "Occidental":       ['25019','25040','25086','25095','25099','25151','25168',
                         '25183','25200','25245','25258','25281','25290','25297',
                         '25322','25377','25386','25430','25438','25513','25649',
                         '25662','25743','25754','25843','25899'],
    "Oriente - Llanos": ['25151','25178','25281','25524','25599','25649'],
    "Oriente Guavio":   ['25181','25279','25293','25312','25318','25326','25430',
                         '25592','25612','25743','25769','25779','25793','25797',
                         '25823','25839','25841','25851','25857','25867','25871',
                         '25873','25877','25885'],
    "Suroccidental":    ['25001','25035','25053','25126','25154','25175','25245',
                         '25258','25281','25286','25290','25297','25307','25320',
                         '25322','25377','25386','25430','25438','25473','25513',
                         '25535','25572','25599','25612','25649','25662','25743',
                         '25754','25769','25779','25785','25793','25817','25823',
                         '25839','25843','25845','25851','25857','25867','25873',
                         '25875','25877','25885','25899'],
}

# ── Países de origen internacional ────────────────────────
PAISES_COORDS = {
    "CHILE":(-71.5430,-35.6751),"ECUADOR":(-78.1834,-1.8312),
    "ESTADOS UNIDOS DE AMÉRICA":(-95.7129,37.0902),
    "ESTADOS UNIDOS DE AMERICA":(-95.7129,37.0902),
    "CANADÁ":(-96.8165,56.1304),"CANADA":(-96.8165,56.1304),
    "PERÚ":(-75.0152,-9.1900),"PERU":(-75.0152,-9.1900),
    "CHINA":(104.1954,35.8617),"VIETNAM":(108.2772,14.0583),
    "ARGENTINA":(-63.6167,-38.4161),"BRASIL":(-51.9253,-14.2350),
    "VENEZUELA":(-66.5897,6.4238),"COSTA RICA":(-83.7534,9.7489),
    "MEXICO":(-102.5528,23.6345),"ESPAÑA":(-3.7492,40.4637),
    "SUDAFRICA":(22.9375,-30.5595),
}

# =========================================================
# KEEP-ALIVE
# =========================================================

st.markdown("""
<script>
(function keepAlive() {
    setInterval(function() {
        fetch(window.location.href, {method:'GET', cache:'no-store'}).catch(function(){});
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
        max-width: 100%; padding-top: 0.8rem; padding-bottom: 1rem;
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
# HELPERS
# =========================================================

def normalizar_codigo_5(valor):
    if pd.isna(valor): return np.nan
    s = str(valor).strip().replace("'", "")
    if s.endswith(".0"): s = s[:-2]
    if s == "": return np.nan
    return s.zfill(5) if s.isdigit() else s

def normalizar_texto(valor):
    if pd.isna(valor): return ""
    return str(valor).strip()

def formatear_cop(valor):
    if pd.isna(valor): return "Sin dato"
    return f"$ {valor:,.0f}"

def formatear_ton(valor):
    if pd.isna(valor): return "Sin dato"
    return f"{valor:,.1f}"

def norm_serie(s):
    s = s.fillna(0)
    if s.max() == s.min():
        return pd.Series(np.ones(len(s)), index=s.index)
    return (s - s.min()) / (s.max() - s.min())

def obtener_mtime(path):
    try: return path.stat().st_mtime
    except: return 0.0

def construir_sankey(df_sk):
    nodos = list(dict.fromkeys(df_sk["municipio_origen"].tolist() + df_sk["central_mayorista"].tolist()))
    idx   = {n: i for i, n in enumerate(nodos)}
    vals  = df_sk["toneladas_total"].astype(float).tolist()
    tot   = sum(vals) or 1
    pcts  = [v / tot * 100 for v in vals]
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(pad=12, thickness=16, line=dict(color="gray", width=0.5), label=nodos),
        link=dict(
            source=[idx[r] for r in df_sk["municipio_origen"]],
            target=[idx[r] for r in df_sk["central_mayorista"]],
            value=vals, customdata=pcts,
            hovertemplate="Origen: %{source.label}<br>Central: %{target.label}<br>Tons: %{value:,.1f}<br>Part.: %{customdata:.1f}%<extra></extra>"
        )
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#171A21", plot_bgcolor="#171A21",
        font=dict(color="#E8EDF5", size=11),
        margin=dict(l=10, r=10, t=10, b=10), height=430
    )
    return fig

# =========================================================
# CARGA DE MUNICIPIOS
# =========================================================

@st.cache_resource(show_spinner=False)
def cargar_municipios(mtime):
    mun = gpd.read_parquet(RUTA_MUNICIPIOS)
    for col in ["MpCodigo","CODIGO_MUNICIPIO"]:
        if col in mun.columns:
            mun["codigo_origen"] = mun[col].apply(normalizar_codigo_5); break
    for col in ["nombre_municipio","MpNombre","MUNICIPIO","Nombre"]:
        if col in mun.columns:
            mun["nombre_municipio"] = mun[col].apply(normalizar_texto); break
    else: mun["nombre_municipio"] = ""
    for col in ["departamento","Depto","DEPARTAMENTO"]:
        if col in mun.columns:
            mun["departamento"] = mun[col].apply(normalizar_texto); break
    else: mun["departamento"] = ""
    return mun

# =========================================================
# CONEXIÓN DUCKDB — abastecimiento
# =========================================================

@st.cache_resource(show_spinner=False)
def get_con_abast(mtime):
    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=4")
    con.execute(f"""
        CREATE OR REPLACE VIEW lineas AS
        SELECT
            CAST(fecha_mes AS DATE)                         AS fecha_mes,
            MONTH(CAST(fecha_mes AS DATE))                  AS mes,
            CASE WHEN MONTH(CAST(fecha_mes AS DATE)) BETWEEN 1 AND 6
                 THEN 'Primer semestre' ELSE 'Segundo semestre' END AS semestre,
            STRFTIME(CAST(fecha_mes AS DATE),'%Y-%m')       AS etiqueta_mes,
            YEAR(CAST(fecha_mes AS DATE))                   AS anio,
            TRIM(CAST(grupo AS VARCHAR))                    AS grupo,
            TRIM(CAST(rubro AS VARCHAR))                    AS rubro,
            TRIM(CAST(central_mayorista AS VARCHAR))        AS central_mayorista,
            CAST(lon_central AS DOUBLE)                     AS lon_central,
            CAST(lat_central AS DOUBLE)                     AS lat_central,
            UPPER(TRIM(CAST(departamento_origen AS VARCHAR))) AS departamento_origen,
            TRIM(CAST(cod_depto AS VARCHAR))                AS cod_depto,
            UPPER(TRIM(CAST(municipio_origen AS VARCHAR)))  AS municipio_origen,
            TRIM(CAST(cod_municipio AS VARCHAR))            AS cod_municipio,
            CAST(lon_mun AS DOUBLE)                         AS lon_mun,
            CAST(lat_mun AS DOUBLE)                         AS lat_mun,
            CAST(toneladas AS DOUBLE)                       AS toneladas,
            CAST(dias_con_datos AS DOUBLE)                  AS dias_con_datos
        FROM read_parquet('{RUTA_LINEAS_SQL}')
    """)
    return con

# =========================================================
# CONEXIÓN DUCKDB — precios
# =========================================================

@st.cache_resource(show_spinner=False)
def get_con_precios(mtime):
    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=4")
    con.execute(f"""
        CREATE OR REPLACE VIEW precios AS
        SELECT
            CAST(fecha AS DATE)                             AS fecha,
            MONTH(CAST(fecha AS DATE))                      AS mes,
            YEAR(CAST(fecha AS DATE))                       AS anio,
            STRFTIME(CAST(fecha AS DATE),'%Y-%m')           AS etiqueta_mes,
            CASE WHEN MONTH(CAST(fecha AS DATE)) BETWEEN 1 AND 6
                 THEN 'Primer semestre' ELSE 'Segundo semestre' END AS semestre,
            TRIM(CAST(grupo AS VARCHAR))                    AS grupo,
            TRIM(CAST(rubro AS VARCHAR))                    AS rubro,
            TRIM(CAST(producto AS VARCHAR))                 AS producto,
            TRIM(CAST(central_mayorista AS VARCHAR))        AS central_mayorista,
            CAST(precio AS DOUBLE)                          AS precio
        FROM read_parquet('{RUTA_PRECIOS_SQL}')
        WHERE precio IS NOT NULL AND precio > 0
          AND LOWER(TRIM(CAST(central_mayorista AS VARCHAR))) != 'popayán, plaza de mercado del barrio bolívar'
        UNION ALL
        SELECT
            CAST(fecha AS DATE) AS fecha,
            MONTH(CAST(fecha AS DATE)) AS mes,
            YEAR(CAST(fecha AS DATE)) AS anio,
            STRFTIME(CAST(fecha AS DATE),'%Y-%m') AS etiqueta_mes,
            CASE WHEN MONTH(CAST(fecha AS DATE)) BETWEEN 1 AND 6
                 THEN 'Primer semestre' ELSE 'Segundo semestre' END AS semestre,
            TRIM(CAST(grupo AS VARCHAR)) AS grupo,
            TRIM(CAST(rubro AS VARCHAR)) AS rubro,
            TRIM(CAST(producto AS VARCHAR)) AS producto,
            'Popayán, Plaza de mercado del barrio Bolívar' AS central_mayorista,
            CAST(precio AS DOUBLE) AS precio
        FROM read_parquet('{RUTA_PRECIOS_SQL}')
        WHERE precio IS NOT NULL AND precio > 0
          AND LOWER(TRIM(CAST(central_mayorista AS VARCHAR))) = 'popayán, plaza de mercado del barrio bolívar'
    """)
    return con

# =========================================================
# CATALOGOS
# =========================================================

@st.cache_data(show_spinner=False)
def consultar_catalogos(mtime_ab, mtime_pr):
    ca = get_con_abast(mtime_ab)
    grupos    = ca.execute("SELECT DISTINCT grupo FROM lineas WHERE grupo IS NOT NULL ORDER BY grupo").df()["grupo"].tolist()
    rubros    = ca.execute("SELECT DISTINCT rubro FROM lineas WHERE rubro IS NOT NULL ORDER BY rubro").df()["rubro"].tolist()
    centrales = ca.execute("SELECT DISTINCT central_mayorista FROM lineas WHERE central_mayorista IS NOT NULL ORDER BY central_mayorista").df()["central_mayorista"].tolist()
    deptos_raw = ca.execute("""
        SELECT DISTINCT
            CASE
                WHEN UPPER(TRIM(departamento_origen)) IN ('BOGOTÁ, D. C.','BOGOTA, D.C.','BOGOTÁ D.C.','BOGOTA D.C.','BOGOTA','BOGOTÁ DC','BOGOTA DC')
                THEN 'BOGOTÁ, D.C.'
                ELSE departamento_origen
            END AS departamento_origen
        FROM lineas WHERE departamento_origen IS NOT NULL
        ORDER BY 1
    """).df()["departamento_origen"].tolist()
    deptos = list(dict.fromkeys(deptos_raw))  # dedup manteniendo orden
    rango     = ca.execute("SELECT MIN(fecha_mes) AS fmin, MAX(fecha_mes) AS fmax FROM lineas WHERE fecha_mes <= '2026-03-01'").df()
    fecha_min = pd.to_datetime(rango.loc[0,"fmin"]).date()
    fecha_max = pd.to_datetime(rango.loc[0,"fmax"]).date()
    return grupos, rubros, centrales, deptos, fecha_min, fecha_max

# =========================================================
# WHERE BUILDER
# =========================================================

BOGOTA_VARIANTES = {
    'BOGOTÁ, D. C.','BOGOTA, D.C.','BOGOTÁ D.C.','BOGOTA D.C.',
    'BOGOTA','BOGOTÁ DC','BOGOTA DC','BOGOTÁ, D.C.','BOGOTÁ'
}

def build_where_ab(fecha_ini, fecha_fin, semestre, grupo, rubros, centrales, deptos, muns_prio=None):
    c = ["fecha_mes BETWEEN ? AND ?", "fecha_mes <= '2026-03-01'"]
    p = [fecha_ini.isoformat(), fecha_fin.isoformat()]
    if semestre == "Primer semestre":    c.append("mes BETWEEN 1 AND 6")
    elif semestre == "Segundo semestre": c.append("mes BETWEEN 7 AND 12")
    if rubros:
        c.append(f"rubro IN ({','.join(['?']*len(rubros))})"); p.extend(list(rubros))
    elif grupo and grupo != "Todos":
        c.append("grupo = ?"); p.append(grupo)
    if centrales:
        c.append(f"central_mayorista IN ({','.join(['?']*len(centrales))})"); p.extend(list(centrales))
    if deptos:
        deptos_norm = []
        for d in deptos:
            if 'BOGOT' in d.upper(): deptos_norm.extend(list(BOGOTA_VARIANTES))
            else: deptos_norm.append(d)
        deptos_norm = list(set(deptos_norm))
        c.append(f"departamento_origen IN ({','.join(['?']*len(deptos_norm))})"); p.extend(deptos_norm)
    if muns_prio:
        ml = list(muns_prio)
        c.append(f"cod_municipio IN ({','.join(['?']*len(ml))})"); p.extend(ml)
    return " AND ".join(c), p

def build_where_pr(fecha_ini, fecha_fin, semestre, grupo, rubros, centrales):
    c = ["fecha BETWEEN ? AND ?", "fecha <= '2026-03-01'"]
    p = [fecha_ini.isoformat(), fecha_fin.isoformat()]
    if semestre == "Primer semestre":    c.append("mes BETWEEN 1 AND 6")
    elif semestre == "Segundo semestre": c.append("mes BETWEEN 7 AND 12")
    if rubros:
        c.append(f"rubro IN ({','.join(['?']*len(rubros))})"); p.extend(list(rubros))
    elif grupo and grupo != "Todos":
        c.append("grupo = ?"); p.append(grupo)
    if centrales:
        c.append(f"central_mayorista IN ({','.join(['?']*len(centrales))})"); p.extend(list(centrales))
    return " AND ".join(c), p

# =========================================================
# CONSULTA ABASTECIMIENTO
# =========================================================

@st.cache_data(show_spinner=False)
def consultar_abast(fecha_ini, fecha_fin, semestre, grupo, rubro,
                    centrales_t, deptos_t, muns_prio_t, mtime):
    con  = get_con_abast(mtime)
    w, p = build_where_ab(fecha_ini, fecha_fin, semestre, grupo, rubro,
                          centrales_t, deptos_t, set(muns_prio_t) if muns_prio_t else None)
    wt, pt = build_where_ab(fecha_ini, fecha_fin, semestre, grupo, rubro, (), ())
    dr   = list(DEPTOS_RAPE)
    ph_r = ",".join(["?"]*len(dr))

    q = f"""
        WITH base AS (SELECT * FROM lineas WHERE {w}),
        base_tot AS (SELECT * FROM lineas WHERE {wt}),
        met AS (
            SELECT SUM(toneladas) AS vol_filtro,
                   COUNT(DISTINCT cod_municipio) AS mun_activos,
                   COUNT(DISTINCT central_mayorista) AS cent_activas
            FROM base
        ),
        met_tot AS (SELECT SUM(toneladas) AS vol_total FROM base_tot),
        met_rape AS (
            SELECT SUM(toneladas) AS vol_rape FROM base_tot
            WHERE departamento_origen IN ({ph_r})
        ),
        ranking AS (
            SELECT
                CASE
                    WHEN UPPER(municipio_origen)='UNE'     THEN '25845'
                    WHEN UPPER(municipio_origen)='FÓMEQUE' THEN '25279'
                    WHEN UPPER(municipio_origen)='FOMEQUE' THEN '25279'
                    WHEN UPPER(municipio_origen)='CERRITO' THEN '68162'
                    ELSE cod_municipio
                END AS cod_municipio,
                MAX(municipio_origen)        AS municipio_origen,
                MAX(departamento_origen)     AS departamento_origen,
                SUM(toneladas)               AS toneladas_total,
                COUNT(DISTINCT etiqueta_mes) AS meses_participacion,
                SUM(dias_con_datos)          AS dias_con_datos
            FROM base GROUP BY 1
        ),
        serie AS (
            SELECT etiqueta_mes, SUM(toneladas) AS toneladas_total
            FROM base GROUP BY 1 ORDER BY 1
        ),
        flujos AS (
            SELECT
                CASE
                    WHEN UPPER(municipio_origen)='UNE'     THEN '25845'
                    WHEN UPPER(municipio_origen)='FÓMEQUE' THEN '25279'
                    WHEN UPPER(municipio_origen)='FOMEQUE' THEN '25279'
                    WHEN UPPER(municipio_origen)='CERRITO' THEN '68162'
                    ELSE cod_municipio
                END AS cod_municipio,
                municipio_origen, departamento_origen, central_mayorista,
                AVG(lon_mun) AS lon_orig, AVG(lat_mun) AS lat_orig,
                AVG(lon_central) AS lon_dest, AVG(lat_central) AS lat_dest,
                SUM(toneladas) AS toneladas_total
            FROM base
            WHERE lon_mun IS NOT NULL AND lat_mun IS NOT NULL
              AND lon_central IS NOT NULL AND lat_central IS NOT NULL
            GROUP BY 1,2,3,4 ORDER BY toneladas_total DESC
        ),
        sankey AS (
            SELECT municipio_origen, central_mayorista,
                SUM(toneladas) AS toneladas_total
            FROM base GROUP BY 1,2 ORDER BY toneladas_total DESC
        )
        SELECT 'met'     AS _t, TO_JSON(met)     AS _j FROM met
        UNION ALL SELECT 'tot',    TO_JSON(met_tot)  FROM met_tot
        UNION ALL SELECT 'rape',   TO_JSON(met_rape) FROM met_rape
        UNION ALL SELECT 'rank',   TO_JSON(ranking)  FROM ranking
        UNION ALL SELECT 'serie',  TO_JSON(serie)    FROM serie
        UNION ALL SELECT 'flujos', TO_JSON(flujos)   FROM flujos
        UNION ALL SELECT 'sankey', TO_JSON(sankey)   FROM sankey
    """
    res = con.execute(q, p + pt + dr).df()

    def ex(n):
        rows = res[res["_t"] == n]["_j"].tolist()
        if not rows: return pd.DataFrame()
        return pd.DataFrame([json.loads(r) for r in rows])

    return ex("met"), ex("tot"), ex("rape"), ex("rank"), ex("serie"), ex("flujos"), ex("sankey")

# =========================================================
# CONSULTA PRECIOS
# =========================================================

@st.cache_data(show_spinner=False)
@st.cache_data(show_spinner=False)
def consultar_precios(fecha_ini, fecha_fin, semestre, grupo, rubro,
                      centrales_t, mtime_pr):
    """Un solo escaneo de precios que devuelve serie mensual y precio por central."""
    con  = get_con_precios(mtime_pr)
    w, p = build_where_pr(fecha_ini, fecha_fin, semestre, grupo, rubro, centrales_t)
    resultado = con.execute(f"""
        WITH base AS (SELECT * FROM precios WHERE {w})
        SELECT 'serie'   AS _t, TO_JSON(s) AS _j
        FROM (
            SELECT etiqueta_mes, AVG(precio) AS precio_promedio
            FROM base GROUP BY 1 ORDER BY 1
        ) s
        UNION ALL
        SELECT 'central' AS _t, TO_JSON(c) AS _j
        FROM (
            SELECT central_mayorista,
                   AVG(precio) AS precio_central,
                   COUNT(*)    AS n_registros
            FROM base GROUP BY 1
        ) c
    """, p).df()

    def ex(nombre):
        rows = resultado[resultado["_t"] == nombre]["_j"].tolist()
        if not rows: return pd.DataFrame()
        return pd.DataFrame([json.loads(r) for r in rows])

    return ex("serie"), ex("central")


# =========================================================
# CARGA INICIAL
# =========================================================

mtime_ab  = obtener_mtime(RUTA_LINEAS)
mtime_pr  = obtener_mtime(RUTA_PRECIOS)
mtime_mun = obtener_mtime(RUTA_MUNICIPIOS)

municipios = cargar_municipios(mtime_mun)
grupos, rubros, centrales, deptos, fecha_min_g, fecha_max_g = consultar_catalogos(mtime_ab, mtime_pr)

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
            Lectura territorial de flujos, precios y eficiencia relativa de municipios de origen
            por producto y central mayorista · SIPSA–DANE 2020–2026
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# FILTROS
# =========================================================

st.markdown('<div class="filter-wrap">', unsafe_allow_html=True)
f1, f2, f3, f4, f5, f6 = st.columns([1.1, 1.3, 1.5, 1.0, 1.0, 1.4])

with f1:
    grupo_sel = st.selectbox("Grupo", ["Todos"] + grupos, index=0)
with f2:
    if grupo_sel != "Todos":
        con_tmp = get_con_abast(mtime_ab)
        rubros_f_codigos = con_tmp.execute(
            "SELECT DISTINCT rubro FROM lineas WHERE grupo=? AND rubro IS NOT NULL ORDER BY rubro",
            [grupo_sel]
        ).df()["rubro"].tolist()
    else:
        rubros_f_codigos = sorted(rubros, key=lambda r: label_rubro(r))
    rubros_f_labels = [label_rubro(r) for r in rubros_f_codigos]
    rubros_sel_labels = st.multiselect("Rubro", options=rubros_f_labels,
                                       default=[], placeholder="Todos los rubros")
    rubros_sel = [codigo_rubro(l) for l in rubros_sel_labels]
with f3:
    centrales_sel = st.multiselect("Central mayorista", options=centrales, default=[])
with f4:
    semestre_sel = st.selectbox("Periodo", ["Todos","Primer semestre","Segundo semestre"], index=0)
with f5:
    deptos_sel = st.multiselect("Depto. origen", deptos, default=[])
with f6:
    rango = st.date_input("Fechas", value=(fecha_min_g, fecha_max_g),
                          min_value=fecha_min_g, max_value=fecha_max_g)

st.markdown("</div>", unsafe_allow_html=True)

# ── Sección SARA ──────────────────────────────────────────
with st.container():
    st.markdown("""
    <div style="background:#1A2133;border:1px solid #3D4F6A;border-radius:10px;
        padding:0.6rem 0.9rem 0.5rem 0.9rem;margin-bottom:0.85rem;">
        <div style="font-size:0.78rem;color:#7A9CC0;font-weight:600;
            letter-spacing:0.06em;margin-bottom:0.5rem;">
            FILTROS DEL PROYECTO SARA
        </div>
    </div>
    """, unsafe_allow_html=True)

    fs0, fs1, fs2, fs3 = st.columns([1.2, 1, 1.5, 2.3])
    with fs0:
        solo_priorizados = st.checkbox(
            "Solo rubros priorizados SARA",
            value=False,
            help="Filtra el selector de Rubro para mostrar únicamente los 37 rubros priorizados en el marco analítico del proyecto SARA"
        )
    with fs1:
        prio_tipo = st.selectbox(
            "Municipios priorizados",
            ["Todos","Oferta","Demanda","Oferta y demanda"],
            index=0
        )
    with fs2:
        territorios_opciones = sorted(TERRITORIOS_FUNC.keys())
        territorio_sel = st.multiselect(
            "Territorio funcional",
            options=territorios_opciones,
            default=[],
            placeholder="Todos los territorios"
        )
    with fs3:
        st.markdown('<div style="font-size:0.82rem;color:#9EABC0;padding-top:1.8rem;">'
                    '🔀 Máx. flujos en mapa</div>', unsafe_allow_html=True)
        max_flujos = st.slider(
            "Máx. flujos", min_value=100, max_value=2000,
            value=MAX_LINEAS_MAPA, step=100, label_visibility="collapsed"
        )

# Aplicar filtro de priorizados SARA al selector de rubros si está activo
if solo_priorizados and not rubros_sel:
    rubros_sel = list(RUBROS_PRIORIZADOS_SARA)

# ── Resolver filtros SARA ─────────────────────────────────
if prio_tipo == "Oferta":
    muns_prio = MUNS_OFERTA
elif prio_tipo == "Demanda":
    muns_prio = MUNS_DEMANDA
elif prio_tipo == "Oferta y demanda":
    muns_prio = MUNS_AMBOS
else:
    muns_prio = None

if territorio_sel:
    muns_territorio = set()
    for t in territorio_sel:
        muns_territorio.update(TERRITORIOS_FUNC.get(t, []))
    muns_prio = (muns_prio & muns_territorio) if muns_prio else muns_territorio

if isinstance(rango, tuple) and len(rango) == 2:
    fecha_ini, fecha_fin = rango
else:
    fecha_ini, fecha_fin = fecha_min_g, fecha_max_g

centrales_t   = tuple(centrales_sel)
deptos_t      = tuple(deptos_sel)
rubros_t      = tuple(rubros_sel)
muns_prio_t   = tuple(sorted(muns_prio)) if muns_prio else ()

rubro_unico       = rubros_sel[0] if len(rubros_sel) == 1 else None
tiene_rubro_unico = len(rubros_sel) == 1

if len(rubros_sel) == 1:
    nivel_sel = label_rubro(rubros_sel[0])
elif len(rubros_sel) > 1:
    nivel_sel = f"{len(rubros_sel)} rubros seleccionados"
elif grupo_sel != "Todos":
    nivel_sel = grupo_sel
else:
    nivel_sel = "Todos los productos"

# =========================================================
# CONSULTAS
# =========================================================

met_df, tot_df, rape_df, rank_df, serie_ab_df, flujos_df, sankey_df = consultar_abast(
    fecha_ini, fecha_fin, semestre_sel, grupo_sel, rubros_t,
    centrales_t, deptos_t, muns_prio_t, mtime_ab
)

serie_pr_df, precios_central_df = consultar_precios(
    fecha_ini, fecha_fin, semestre_sel, grupo_sel, rubros_t,
    centrales_t, mtime_pr
)

# ── Datos internacionales ─────────────────────────────────
@st.cache_data(show_spinner=False)
def consultar_internacionales(fecha_ini, fecha_fin, semestre, grupo, rubros,
                               centrales_t, mtime):
    con = get_con_abast(mtime)
    c = ["fecha_mes BETWEEN ? AND ?","fecha_mes <= '2026-03-01'"]
    p = [fecha_ini.isoformat(), fecha_fin.isoformat()]
    if semestre == "Primer semestre":    c.append("mes BETWEEN 1 AND 6")
    elif semestre == "Segundo semestre": c.append("mes BETWEEN 7 AND 12")
    if rubros:
        c.append(f"rubro IN ({','.join(['?']*len(rubros))})"); p.extend(list(rubros))
    elif grupo and grupo != "Todos":
        c.append("grupo = ?"); p.append(grupo)
    if centrales_t:
        c.append(f"central_mayorista IN ({','.join(['?']*len(centrales_t))})"); p.extend(list(centrales_t))
    # Solo registros sin código DIVIPOLA válido (internacionales)
    c.append("(LENGTH(TRIM(cod_municipio)) != 5 OR TRY_CAST(cod_municipio AS INTEGER) IS NULL)")
    w = " AND ".join(c)
    try:
        df = con.execute(f"""
            SELECT municipio_origen AS pais_origen, central_mayorista,
                   AVG(lon_central) AS lon_dest, AVG(lat_central) AS lat_dest,
                   SUM(toneladas) AS toneladas_total
            FROM lineas WHERE {w}
            GROUP BY 1,2 ORDER BY toneladas_total DESC
        """, p).df()
    except Exception:
        df = pd.DataFrame()
    return df

intl_raw = consultar_internacionales(
    fecha_ini, fecha_fin, semestre_sel, grupo_sel, rubros_t, centrales_t, mtime_ab
)
if not intl_raw.empty:
    intl_raw["lon_orig"] = intl_raw["pais_origen"].str.upper().map(
        {k: v[0] for k, v in PAISES_COORDS.items()})
    intl_raw["lat_orig"] = intl_raw["pais_origen"].str.upper().map(
        {k: v[1] for k, v in PAISES_COORDS.items()})
    intl_df = intl_raw.dropna(subset=["lon_orig","lat_orig","lon_dest","lat_dest"]).copy()
    if not intl_df.empty:
        vi = intl_df["toneladas_total"].sum()
        intl_df["pct_intl"] = intl_df["toneladas_total"] / vi * 100
        vi_min, vi_max = intl_df["toneladas_total"].min(), intl_df["toneladas_total"].max()
        intl_df["ancho"] = 2 + 8*((intl_df["toneladas_total"]-vi_min)/(vi_max-vi_min+1e-9))
        intl_df["tipo_elemento"] = "Origen internacional"
        intl_df["detalle_1"] = "Pais: "      + intl_df["pais_origen"].fillna("")
        intl_df["detalle_2"] = "Central: "   + intl_df["central_mayorista"].fillna("")
        intl_df["detalle_3"] = "Toneladas: " + intl_df["toneladas_total"].map(formatear_ton)
        intl_df["detalle_4"] = "% vol. intl: " + intl_df["pct_intl"].map("{:.1f}%".format)
else:
    intl_df = pd.DataFrame()

# =========================================================
# ESCALARES
# =========================================================

def _sf(df, col):
    try: v = df.loc[0,col]; return float(v) if pd.notna(v) else 0.0
    except: return 0.0
def _si(df, col):
    try: v = df.loc[0,col]; return int(v) if pd.notna(v) else 0
    except: return 0

vol_filtro = _sf(met_df, "vol_filtro")
mun_act    = _si(met_df, "mun_activos")
cent_act   = _si(met_df, "cent_activas")
vol_total  = _sf(tot_df, "vol_total")
vol_rape   = _sf(rape_df, "vol_rape")

# Precio promedio general — solo válido con un único rubro seleccionado
precio_prom_general = (
    serie_pr_df["precio_promedio"].mean()
    if not serie_pr_df.empty and tiene_rubro_unico
    else None
)

# =========================================================
# RANKING
# =========================================================

if not rank_df.empty:
    rk = rank_df.copy()
    total_meses = max(serie_ab_df["etiqueta_mes"].nunique(), 1) if not serie_ab_df.empty else 1
    rk["part_filtro"] = np.where(vol_filtro > 0, rk["toneladas_total"] / vol_filtro * 100, 0)
    rk["part_total"]  = np.where(vol_total  > 0, rk["toneladas_total"] / vol_total  * 100, 0)
    rk["part_rape"]   = np.where(vol_rape   > 0, rk["toneladas_total"] / vol_rape   * 100, 0)
    rk["frec"]        = rk["meses_participacion"] / total_meses
    rk["score_vol"]   = norm_serie(rk["toneladas_total"])
    rk["score_act"]   = norm_serie(rk["meses_participacion"])
    top30 = set(rk.sort_values("toneladas_total", ascending=False).head(30)["cod_municipio"].astype(str))

    # ── Cruzar precio por central con flujos ──────────────────
    if not flujos_df.empty and not precios_central_df.empty:
        flujos_df = flujos_df.merge(
            precios_central_df[["central_mayorista","precio_central"]],
            on="central_mayorista", how="left"
        )
    elif not flujos_df.empty:
        flujos_df["precio_central"] = np.nan

    # Precio por municipio — Opción A:
    # Se usa SOLO el precio de las centrales que aparecen en los flujos filtrados.
    # Si el usuario filtra por Corabastos, el precio de cada municipio es
    # el de Corabastos únicamente, sin mezclar otras centrales sin precio.
    # Esto evita NaN cuando el municipio abastece a otras centrales sin dato de precio.
    if not flujos_df.empty and "precio_central" in flujos_df.columns:
        # Solo usar flujos que SÍ tienen precio (centrales filtradas con dato)
        sub = flujos_df.dropna(subset=["precio_central"])
        if not sub.empty:
            # Precio ponderado por toneladas — solo sobre centrales con precio disponible
            pm = (
                sub.groupby("cod_municipio")
                .apply(lambda g: np.average(g["precio_central"], weights=g["toneladas_total"])
                       if g["toneladas_total"].sum() > 0 else np.nan)
                .reset_index()
            )
            pm.columns = ["cod_municipio", "precio_municipio"]
            rk = rk.merge(pm, on="cod_municipio", how="left")
        else:
            rk["precio_municipio"] = np.nan
    else:
        rk["precio_municipio"] = np.nan

    # ── Índice con precio si hay rubro seleccionado ───────────
    precio_ref_global = rk["precio_municipio"].median() if rk["precio_municipio"].notna().any() else 0
    tiene_precio = rk["precio_municipio"].notna().any() and tiene_rubro_unico

    if tiene_precio and precio_ref_global > 0:
        rk["ventaja_precio"] = (precio_ref_global - rk["precio_municipio"]) / precio_ref_global * 100
        rk["score_precio"]   = norm_serie(rk["ventaja_precio"].clip(lower=0))
        rk["indice"] = rk["score_vol"] * 0.40 + rk["score_act"] * 0.20 + rk["score_precio"] * 0.40
    else:
        rk["ventaja_precio"] = np.nan
        rk["indice"]         = rk["score_vol"] * 0.60 + rk["score_act"] * 0.40

    rk = rk.sort_values(["indice","toneladas_total"], ascending=False).reset_index(drop=True)
    rk["ranking"] = rk.index + 1

    # ── Preparar flujos para mapa ─────────────────────────────
    if not flujos_df.empty:
        total_ton_flujos = flujos_df["toneladas_total"].sum()
        flujos_df = flujos_df.head(max_flujos).copy()
        ton_visible   = flujos_df["toneladas_total"].sum()
        pct_cobertura = (ton_visible / total_ton_flujos * 100) if total_ton_flujos > 0 else 0
        vmin, vmax    = flujos_df["toneladas_total"].min(), flujos_df["toneladas_total"].max()
        flujos_df["ancho"]      = 2 + 10*((flujos_df["toneladas_total"]-vmin)/(vmax-vmin+1e-9))
        flujos_df["ton_fmt"]    = flujos_df["toneladas_total"].map(formatear_ton)
        flujos_df["precio_fmt"] = flujos_df["precio_central"].map(formatear_cop) \
                                  if "precio_central" in flujos_df.columns \
                                  else "Sin dato"
    else:
        pct_cobertura = 0

    if not sankey_df.empty:
        top_mun = sankey_df.groupby("municipio_origen")["toneladas_total"].sum().nlargest(12).index.tolist()
        sk_top  = sankey_df[sankey_df["municipio_origen"].isin(top_mun)].copy()
    else:
        sk_top = pd.DataFrame(columns=["municipio_origen","central_mayorista","toneladas_total"])
else:
    rk        = pd.DataFrame()
    top30     = set()
    flujos_df = pd.DataFrame()
    sk_top    = pd.DataFrame(columns=["municipio_origen","central_mayorista","toneladas_total"])
    pct_cobertura = 0

# =========================================================
# MAPA — polígonos con colores por estado
# deptos_sel resalta municipios del departamento en amarillo
# top30 en morado, resto gris
# =========================================================

deptos_sel_upper = {d.upper() for d in deptos_sel}

def color_fill(codigo, depto):
    if str(codigo) in top30:
        return [110, 68, 255, 160]
    if deptos_sel_upper and str(depto).upper() in deptos_sel_upper:
        return [180, 190, 205, 60]   # gris azulado muy suave
    return [40, 48, 62, 18]

def color_line(codigo, depto):
    if str(codigo) in top30:
        return [170, 130, 255, 240]
    if deptos_sel_upper and str(depto).upper() in deptos_sel_upper:
        return [200, 210, 225, 180]  # gris claro para el borde
    return [100, 110, 125, 60]

mun_web = municipios[["nombre_municipio","departamento","codigo_origen","geometry"]].copy()
mun_web["fill_color"] = [
    color_fill(c, d)
    for c, d in zip(mun_web["codigo_origen"].astype(str), mun_web["departamento"])
]
mun_web["line_color"] = [
    color_line(c, d)
    for c, d in zip(mun_web["codigo_origen"].astype(str), mun_web["departamento"])
]
mun_web["tipo_elemento"] = "Municipio"
mun_web["detalle_1"] = "Nombre: "       + mun_web["nombre_municipio"].fillna("Sin nombre").astype(str)
mun_web["detalle_2"] = "Departamento: " + mun_web["departamento"].fillna("Sin dato").astype(str)
mun_web["detalle_3"] = "Código: "       + mun_web["codigo_origen"].fillna("").astype(str)
mun_web["detalle_4"] = ""

geojson_mun = json.loads(
    mun_web[["fill_color","line_color","tipo_elemento",
             "detalle_1","detalle_2","detalle_3","detalle_4","geometry"]].to_json()
)

# Centrales como puntos
if not flujos_df.empty:
    cent_pts = (
        flujos_df.groupby("central_mayorista", as_index=False)
        .agg(lon=("lon_dest","first"), lat=("lat_dest","first"),
             nombre=("central_mayorista","first"))
    ).dropna(subset=["lon","lat"])
    cent_pts["tipo_elemento"] = "Central mayorista"
    cent_pts["detalle_1"]     = "Central: " + cent_pts["nombre"].fillna("")
    cent_pts["detalle_2"]     = ""
    cent_pts["detalle_3"]     = ""
    cent_pts["detalle_4"]     = ""
else:
    cent_pts = pd.DataFrame(columns=["central_mayorista","lon","lat","nombre",
                                      "tipo_elemento","detalle_1","detalle_2","detalle_3","detalle_4"])

# Arcos tooltip
if not flujos_df.empty:
    flujos_df["tipo_elemento"] = "Flujo OD"
    flujos_df["detalle_1"]     = "Origen: "        + flujos_df["municipio_origen"].fillna("")
    flujos_df["detalle_2"]     = "Central: "        + flujos_df["central_mayorista"].fillna("")
    flujos_df["detalle_3"]     = "Toneladas: "      + flujos_df["ton_fmt"].fillna("")
    flujos_df["detalle_4"]     = "Precio central: " + flujos_df["precio_fmt"].fillna("Sin dato")

# =========================================================
# LAYOUT PRINCIPAL
# =========================================================

@st.fragment
def render_principal(vol_filtro, mun_act, cent_act, precio_prom_general,
                     geojson_mun, flujos_df, cent_pts, intl_df,
                     serie_ab_df, serie_pr_df, sk_top, nivel_sel,
                     deptos_sel, tiene_rubro_unico, max_flujos, pct_cobertura):
    left_col, center_col, right_col = st.columns([1.05, 3.8, 1.45], gap="small")

    with left_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Indicadores principales</div>', unsafe_allow_html=True)

        # Precio promedio solo si hay rubro seleccionado
        if tiene_rubro_unico and precio_prom_general is not None:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Precio promedio</div>
                <div class="metric-value" style="font-size:1.65rem;">$ {precio_prom_general:,.0f}</div>
                <div class="metric-small">Mercado filtrado ($/kg)</div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Toneladas abastecidas</div>
            <div class="metric-value">{vol_filtro:,.0f}</div>
            <div class="metric-small">Periodo filtrado</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Municipios origen activos</div>
            <div class="metric-value">{mun_act:,}</div>
            <div class="metric-small">Con flujo válido</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Centrales activas</div>
            <div class="metric-value">{cent_act}</div>
            <div class="metric-small">Bajo filtros actuales</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="panel-title">Leyenda</div>', unsafe_allow_html=True)
        leyenda_depto = ""
        if deptos_sel:
            leyenda_depto = '<div class="legend-item"><span class="legend-box" style="background:#B0B8C8;border:1px solid #888;"></span>Municipios depto. filtrado</div>'
        leyenda_intl = '<div class="legend-item"><span class="legend-box" style="background:#00C878;"></span>Flujos internacionales</div>'
        st.markdown(f"""
        <div class="legend-item"><span class="legend-box" style="background:#6E44FF;"></span>Top 30 abastecedores</div>
        {leyenda_depto}
        <div class="legend-item"><span class="legend-box" style="background:#F5A020;border-radius:50%;"></span>Municipio de origen activo</div>
        <div class="legend-item"><span class="legend-box" style="background:#F5B041;"></span>Arcos de flujo nacional</div>
        {leyenda_intl}
        <div class="legend-item"><span class="legend-box" style="background:#00D2FF;"></span>Central mayorista</div>
        <div class="small-note" style="margin-top:0.75rem;">
            <b>Flujos visibles:</b> {max_flujos:,} principales flujos por volumen
            ({pct_cobertura:.0f}% del total bajo filtros activos).
            Los arcos internacionales son adicionales a este limite.
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

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
                get_width="ancho", width_scale=1, width_min_pixels=1,
                pickable=True, auto_highlight=True
            ))
            orig_pts = flujos_df.groupby(
                ["municipio_origen","departamento_origen","cod_municipio"],as_index=False
            ).agg(lon=("lon_orig","first"),lat=("lat_orig","first"),
                  toneladas_total=("toneladas_total","sum"))
            orig_pts = orig_pts.dropna(subset=["lon","lat"])
            orig_pts["tipo_elemento"] = "Municipio de origen"
            orig_pts["detalle_1"] = "Municipio: "    + orig_pts["municipio_origen"].fillna("")
            orig_pts["detalle_2"] = "Departamento: " + orig_pts["departamento_origen"].fillna("")
            orig_pts["detalle_3"] = "Toneladas: "    + orig_pts["toneladas_total"].map(formatear_ton)
            orig_pts["detalle_4"] = ""
            layers.append(pdk.Layer(
                "ScatterplotLayer", data=orig_pts,
                get_position="[lon, lat]", get_radius=4200,
                get_fill_color=[245,160,32,180], get_line_color=[255,210,100,220],
                line_width_min_pixels=1, pickable=True, auto_highlight=True
            ))
        if not intl_df.empty:
            layers.append(pdk.Layer(
                "ArcLayer", data=intl_df,
                get_source_position=["lon_orig","lat_orig"],
                get_target_position=["lon_dest","lat_dest"],
                get_source_color=[0,200,120,210], get_target_color=[0,240,160,210],
                get_width="ancho", width_scale=1, width_min_pixels=1,
                pickable=True, auto_highlight=True
            ))

        if not cent_pts.empty:
            layers.append(pdk.Layer(
                "ScatterplotLayer", data=cent_pts,
                get_position="[lon, lat]", get_radius=13500,
                get_fill_color=[0,210,255,190], get_line_color=[170,245,255,255],
                line_width_min_pixels=2, pickable=True
            ))

        deck = pdk.Deck(
            layers=layers,
            initial_view_state=pdk.ViewState(latitude=4.5, longitude=-74.1, zoom=4.6, pitch=0),
            tooltip={
                "html": "<b>{tipo_elemento}</b><br/>{detalle_1}<br/>{detalle_2}<br/>{detalle_3}<br/>{detalle_4}",
                "style": {"backgroundColor":"rgba(18,22,29,0.95)","color":"#F5F7FA","fontSize":"12px"}
            },
            map_style="dark",
        )
        st.pydeck_chart(deck, use_container_width=True)

        # Serie mensual precio + toneladas
        st.markdown('<div class="panel-title" style="margin-top:0.65rem;">Serie mensual — precio y toneladas abastecidas</div>', unsafe_allow_html=True)

        if not serie_ab_df.empty or not serie_pr_df.empty:
            fig = go.Figure()

            if not serie_pr_df.empty:
                fig.add_trace(go.Bar(
                    x=serie_pr_df["etiqueta_mes"], y=serie_pr_df["precio_promedio"],
                    name="Precio promedio ($/kg)", marker_color="#4DA3FF", yaxis="y1",
                    hovertemplate="Mes: %{x}<br>Precio: $%{y:,.0f}<extra></extra>"
                ))

            if not serie_ab_df.empty:
                fig.add_trace(go.Scatter(
                    x=serie_ab_df["etiqueta_mes"], y=serie_ab_df["toneladas_total"],
                    name="Toneladas abastecidas", mode="lines+markers",
                    line=dict(color="#F5B041", width=2.5), marker=dict(size=6, color="#F5B041"),
                    yaxis="y2",
                    hovertemplate="Mes: %{x}<br>Toneladas: %{y:,.1f}<extra></extra>"
                ))

            fig.update_layout(
                template="plotly_dark", paper_bgcolor="#171A21", plot_bgcolor="#171A21",
                margin=dict(l=15, r=15, t=10, b=10), height=300,
                legend=dict(orientation="h", y=1.08, x=0),
                xaxis=dict(showgrid=False),
                yaxis=dict(title="Precio ($/kg)", gridcolor="#2B3240"),
                yaxis2=dict(title="Toneladas", overlaying="y", side="right", showgrid=False)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay datos de serie con los filtros actuales.")

    with right_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown(f'<div class="panel-title">Flujos hacia centrales mayoristas<br><span style="font-weight:400;font-size:0.82rem;color:#AEB9C9;">{nivel_sel}</span></div>', unsafe_allow_html=True)
        if not sk_top.empty:
            st.plotly_chart(construir_sankey(sk_top), use_container_width=True)
        else:
            st.info("No hay datos suficientes.")
        st.markdown('</div>', unsafe_allow_html=True)


render_principal(
    vol_filtro=vol_filtro, mun_act=mun_act, cent_act=cent_act,
    precio_prom_general=precio_prom_general,
    geojson_mun=geojson_mun, flujos_df=flujos_df, cent_pts=cent_pts,
    intl_df=intl_df,
    serie_ab_df=serie_ab_df, serie_pr_df=serie_pr_df,
    sk_top=sk_top, nivel_sel=nivel_sel, deptos_sel=deptos_sel,
    tiene_rubro_unico=tiene_rubro_unico,
    max_flujos=max_flujos, pct_cobertura=pct_cobertura
)

# =========================================================
# TABLA
# =========================================================

@st.fragment
def render_tabla(rk, vol_filtro, vol_total, vol_rape, max_filas, tiene_rubro_unico):
    st.markdown('<div class="panel-title" style="margin-top:0.8rem;">Tabla consolidada de análisis</div>', unsafe_allow_html=True)

    if not rk.empty:
        cols_base    = ["ranking","municipio_origen","departamento_origen",
                        "toneladas_total","meses_participacion",
                        "part_filtro","part_total","part_rape"]
        nombres_base = ["Ranking","Municipio origen","Departamento origen",
                        "Toneladas acumuladas","Meses activos",
                        "Participación en filtro","Participación total",
                        "Participación RAPE"]
        tabla = rk[cols_base].copy()
        tabla.columns = nombres_base

        tiene_precio = (tiene_rubro_unico
                        and "precio_municipio" in rk.columns
                        and rk["precio_municipio"].notna().any())
        if tiene_precio:
            tabla.insert(4, "Precio prom. ($/kg)",
                         rk["precio_municipio"].map(
                             lambda x: formatear_cop(x) if pd.notna(x) else "Sin dato"))
            if "ventaja_precio" in rk.columns:
                tabla.insert(5, "Ventaja precio (%)",
                             rk["ventaja_precio"].map(
                                 lambda x: f"{x:+.1f}%" if pd.notna(x) else "Sin dato"))

        # Opciones de ordenamiento — todos los campos numéricos
        if tiene_precio:
            cols_ord = ["Toneladas acumuladas","Precio prom. ($/kg)","Ventaja precio (%)",
                        "Meses activos","Participación en filtro","Participación total",
                        "Participación RAPE","Ranking"]
        else:
            cols_ord = ["Toneladas acumuladas","Meses activos",
                        "Participación en filtro","Participación total",
                        "Participación RAPE","Ranking"]

        c1, c2 = st.columns([2,1])
        with c1:
            col_ord = st.selectbox("Ordenar por", cols_ord, index=0,
                                   key="t_col", label_visibility="collapsed")
        with c2:
            dir_ord = st.radio("Dir", ["↓ Mayor","↑ Menor"], index=0,
                               horizontal=True, key="t_dir", label_visibility="collapsed")

        asc = dir_ord == "↑ Menor"
        if col_ord == "Precio prom. ($/kg)" and tiene_precio:
            tabla["_s"] = rk["precio_municipio"].values
            tabla = tabla.sort_values("_s", ascending=asc).drop(columns=["_s"])
        elif col_ord == "Ventaja precio (%)" and tiene_precio:
            tabla["_s"] = rk["ventaja_precio"].values
            tabla = tabla.sort_values("_s", ascending=asc).drop(columns=["_s"])
        else:
            tabla = tabla.sort_values(col_ord, ascending=asc)
        tabla = tabla.reset_index(drop=True)
        if col_ord != "Ranking":
            tabla["Ranking"] = range(1, len(tabla)+1)

        tf = tabla.copy()
        tf["Toneladas acumuladas"]    = tf["Toneladas acumuladas"].map("{:,.1f}".format)
        tf["Participación en filtro"] = tf["Participación en filtro"].map("{:.1f}%".format)
        tf["Participación total"]     = tf["Participación total"].map("{:.1f}%".format)
        tf["Participación RAPE"]      = tf["Participación RAPE"].map("{:.1f}%".format)

        st.dataframe(tf.head(max_filas), use_container_width=True, hide_index=True, height=420)
    else:
        st.info("No hay información disponible.")

    st.markdown("""
    <div class="method-note">
        <b>Nota sobre el precio:</b> El dato de precio solo es interpretable cuando se selecciona
        un <b>unico rubro</b>. Con multiples rubros o sin rubro seleccionado no se muestra precio.
        Si una central no reporta precio para el rubro seleccionado en SIPSA, aparece "Sin dato".
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:0.8rem;padding-top:0.6rem;border-top:1px solid #2B3240;
        color:#8FA0B7;font-size:0.8rem;text-align:center;">
        Fuente: SIPSA · DANE · 2020–2026
    </div>
    """, unsafe_allow_html=True)


render_tabla(rk=rk, vol_filtro=vol_filtro, vol_total=vol_total,
             vol_rape=vol_rape, max_filas=MAX_FILAS_TABLA,
             tiene_rubro_unico=tiene_rubro_unico)
