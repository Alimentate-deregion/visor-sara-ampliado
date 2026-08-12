import base64
import json
import unicodedata
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
RUTA_INTERNACIONALES  = BASE_DIR / "lineas_internacionales.parquet"
RUTA_LOGO             = BASE_DIR / "MDS-245-ES.jpg"
RUTA_LINEAS_SQL       = RUTA_LINEAS.as_posix()
RUTA_PRECIOS_SQL      = RUTA_PRECIOS.as_posix()
RUTA_INTL_SQL         = RUTA_INTERNACIONALES.as_posix()

DEPTOS_RAPE = {
    "BOGOTÁ", "BOGOTÁ, D.C.", "BOGOTA", "BOGOTA D.C.", "BOGOTÁ D.C.",
    "CUNDINAMARCA", "META", "BOYACÁ", "BOYACA", "TOLIMA", "HUILA"
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
    "Carne_res","Pescado","Huevos","Leche","Quesos_cuajadas","Panela"
}

# ── Clasificaciones territoriales SARA ─────────────────────
# Fuente: Municipios_Nacional_Base_Completa.xlsx (hoja "Municipios").
# Se mantienen las categorías exactamente como aparecen en la base suministrada.

DEPTOS_REGION_CENTRAL = {
    'BOYACÁ','BOYACA','CUNDINAMARCA','HUILA','META','TOLIMA',
    'BOGOTÁ','BOGOTÁ, D.C.','BOGOTA','BOGOTA D.C.','BOGOTÁ D.C.'
}
DEPTOS_REGION_CENTRAL_EXTERNA = {'BOYACÁ','BOYACA','HUILA','META','TOLIMA'}
DEPTOS_RMBC = {
    'CUNDINAMARCA','BOGOTÁ','BOGOTÁ, D.C.','BOGOTA','BOGOTA D.C.','BOGOTÁ D.C.'
}

TERRITORIOS_FUNC = {
    'Bogotá, D.C.': {
        '11001',
},
    'Noroccidental': {
        '25148', '25214', '25260', '25320', '25394', '25398', '25402', '25489',
        '25491', '25572', '25592', '25658', '25769', '25777', '25799', '25851',
        '25862', '25875', '25885',
},
    'Norte': {
        '25126', '25154', '25175', '25183', '25200', '25224', '25258', '25288',
        '25295', '25317', '25407', '25426', '25436', '25486', '25513', '25518',
        '25653', '25736', '25745', '25758', '25772', '25779', '25781', '25785',
        '25793', '25807', '25817', '25823', '25843', '25871', '25873', '25899',
},
    'Occidental': {
        '25019', '25040', '25086', '25095', '25099', '25123', '25168', '25269',
        '25286', '25328', '25430', '25473', '25580', '25596', '25662', '25718',
        '25867', '25898',
},
    'Oriente - Llanos': {
        '25151', '25178', '25281', '25335', '25339', '25438', '25530', '25594',
        '25845',
},
    'Oriente Guavio': {
        '25181', '25279', '25293', '25297', '25299', '25322', '25326', '25372',
        '25377', '25839', '25841',
},
    'Suroccidental': {
        '25001', '25035', '25053', '25120', '25245', '25290', '25307', '25312',
        '25324', '25368', '25386', '25483', '25488', '25506', '25524', '25535',
        '25599', '25612', '25645', '25649', '25740', '25743', '25754', '25797',
        '25805', '25815', '25878',
},
}

MUNS_PRIORI_ROL = {
    'Oferta': {
        '15001', '15047', '15238', '15407', '15638', '15646', '15696', '15759',
        '15763', '15814', '25178', '25181', '25214', '25260', '25279', '25312',
        '25322', '25535', '25649', '25743', '25769', '25793', '25817', '25845',
        '25873', '41001', '41298', '41551', '50001', '50251', '50287', '50313',
        '50400', '50568', '50573', '50590', '50689', '73001', '73124', '73268',
        '73443',
},
    'Demanda': {
        '25126', '25175', '25297', '25307', '25320', '25438', '25513', '25662',
        '25875',
},
    'Oferta y demanda': {
        '11001', '25151', '25183', '25269', '25286', '25290', '25377', '25386',
        '25430', '25473', '25754', '25843', '25899',
},
}

MUNS_MUESTRA_178 = {
    '11001', '15001', '15238', '15407', '15646', '15759', '25126', '25151',
    '25175', '25178', '25181', '25183', '25214', '25260', '25269', '25279',
    '25286', '25290', '25297', '25307', '25312', '25320', '25322', '25377',
    '25386', '25430', '25438', '25473', '25513', '25535', '25649', '25662',
    '25740', '25743', '25754', '25769', '25793', '25815', '25843', '25845',
    '25873', '25875', '25899', '41001', '50001', '50313', '50568', '73001',
    '73268', '73443',
}

MUNS_CONMUTADOS = {
    '25126', '25175', '25214', '25269', '25286', '25377', '25430', '25473',
    '25754', '25899',
}

MUNS_REGION_METROPOLITANA = {
    '11001', '25001', '25019', '25035', '25040', '25053', '25086', '25095',
    '25099', '25120', '25123', '25126', '25148', '25151', '25154', '25168',
    '25175', '25178', '25181', '25183', '25200', '25214', '25224', '25245',
    '25258', '25260', '25269', '25279', '25281', '25286', '25288', '25290',
    '25293', '25295', '25297', '25299', '25307', '25312', '25317', '25320',
    '25322', '25324', '25326', '25328', '25335', '25339', '25368', '25372',
    '25377', '25386', '25394', '25398', '25402', '25407', '25426', '25430',
    '25436', '25438', '25473', '25483', '25486', '25488', '25489', '25491',
    '25506', '25513', '25518', '25524', '25530', '25535', '25572', '25580',
    '25592', '25594', '25596', '25599', '25612', '25645', '25649', '25653',
    '25658', '25662', '25718', '25736', '25740', '25743', '25745', '25754',
    '25758', '25769', '25772', '25777', '25779', '25781', '25785', '25793',
    '25797', '25799', '25805', '25807', '25815', '25817', '25823', '25839',
    '25841', '25843', '25845', '25851', '25862', '25867', '25871', '25873',
    '25875', '25878', '25885', '25898', '25899',
}

# Compatibilidad con la lógica histórica:
# "Oferta" incluye los municipios de rol mixto; "Demanda" también.
MUNS_AMBOS = set(MUNS_PRIORI_ROL["Oferta y demanda"])
MUNS_OFERTA = set(MUNS_PRIORI_ROL["Oferta"]) | MUNS_AMBOS
MUNS_DEMANDA = set(MUNS_PRIORI_ROL["Demanda"]) | MUNS_AMBOS

# ── Países de origen internacional ────────────────────────
PAISES_COORDS = {
    # Originales
    "CHILE":           (-71.5430, -35.6751),
    "ECUADOR":         (-78.1834,  -1.8312),
    "ESTADOS UNIDOS DE AMÉRICA": (-95.7129, 37.0902),
    "ESTADOS UNIDOS DE AMERICA": (-95.7129, 37.0902),
    "CANADÁ":          (-96.8165,  56.1304),
    "CANADA":          (-96.8165,  56.1304),
    "PERÚ":            (-75.0152,  -9.1900),
    "PERU":            (-75.0152,  -9.1900),
    "CHINA":           (104.1954,  35.8617),
    "VIETNAM":         (108.2772,  14.0583),
    "ARGENTINA":       (-63.6167, -38.4161),
    "BRASIL":          (-51.9253, -14.2350),
    "VENEZUELA":       (-66.5897,   6.4238),
    "COSTA RICA":      (-83.7534,   9.7489),
    "MEXICO":          (-102.5528,  23.6345),
    "ESPAÑA":          (  -3.7492,  40.4637),
    "SUDAFRICA":       ( 22.9375,  -30.5595),
    # Añadidos tras diagnóstico del parquet (33 países con datos sin coordenadas)
    "BÉLGICA":         (  4.4699,  50.5039),
    "BOLIVIA":         (-64.9909, -16.2902),
    "ITALIA":          ( 12.5674,  41.8719),
    "FRANCIA":         (  2.3488,  46.2276),
    "PANAMÁ":          (-80.7821,   8.9936),
    "URUGUAY":         (-55.7658, -32.5228),
    "PAÍSES BAJOS":    (  5.2913,  52.1326),
    "AFGANISTÁN":      ( 67.7100,  33.9391),
    "INDIA":           ( 78.9629,  20.5937),
    "NICARAGUA":       (-85.2072,  12.8654),
    "NUEVA ZELANDA":   (174.8860, -40.9006),
    "POLONIA":         ( 19.1451,  51.9194),
    "PORTUGAL":        ( -8.2245,  39.3999),
    "ESCOCIA":         ( -4.2026,  56.4907),
    "PARAGUAY":        (-58.4438, -23.4425),
    "GUATEMALA":       (-90.2308,  15.7835),
    "GRECIA":          ( 21.8243,  39.0742),
    "ALEMANIA":        ( 10.4515,  51.1657),
    "TURQUIA":         ( 35.2433,  38.9637),
    "HONDURAS":        (-86.2419,  15.2000),
    "GRAN BRETAÑA":    ( -3.4359,  55.3781),
    "IRLANDA":         ( -8.2439,  53.4129),
    "UZBEKISTÁN":      ( 64.5853,  41.3775),
    "BELICE":          (-88.4976,  17.1899),
    "BULGARIA":        ( 25.4858,  42.7339),
    "TAIWÁN":          (120.9605,  23.6978),
    "BAHAMAS":         (-77.3963,  25.0343),
    "PAKISTÁN":        ( 69.3451,  30.3753),
    "NORUEGA":         (  8.4689,  60.4720),
    "SUECIA":          ( 18.6435,  60.1282),
    "PUERTO RICO":     (-66.5901,  18.2208),
    "UGANDA":          ( 32.2903,   1.3733),
    "MÉXICO":          (-102.5528,  23.6345),
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
            CASE
                WHEN LOWER(TRIM(CAST(central_mayorista AS VARCHAR))) IN
                     ('pereira, la 41', 'pereira, la 41-impala')
                    THEN 'Pereira, La 41'
                WHEN LOWER(TRIM(CAST(central_mayorista AS VARCHAR))) IN
                     ('cali, santa elena', 'cali, santa helena')
                    THEN 'Cali, Santa Elena'
                ELSE TRIM(CAST(central_mayorista AS VARCHAR))
            END                                             AS central_mayorista,
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
            CASE
                WHEN LOWER(TRIM(CAST(central_mayorista AS VARCHAR))) IN
                     ('pereira, la 41', 'pereira, la 41-impala')
                    THEN 'Pereira, La 41'
                WHEN LOWER(TRIM(CAST(central_mayorista AS VARCHAR))) IN
                     ('cali, santa elena', 'cali, santa helena')
                    THEN 'Cali, Santa Elena'
                ELSE TRIM(CAST(central_mayorista AS VARCHAR))
            END                                             AS central_mayorista,
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

@st.cache_resource(show_spinner=False)
def get_con_intl(mtime):
    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=4")
    if RUTA_INTERNACIONALES.exists():
        con.execute(f"""
            CREATE OR REPLACE VIEW internacionales AS
            SELECT
                CAST(fecha_mes AS DATE)                         AS fecha_mes,
                MONTH(CAST(fecha_mes AS DATE))                  AS mes,
                CASE WHEN MONTH(CAST(fecha_mes AS DATE)) BETWEEN 1 AND 6
                     THEN 'Primer semestre' ELSE 'Segundo semestre' END AS semestre,
                STRFTIME(CAST(fecha_mes AS DATE),'%Y-%m')       AS etiqueta_mes,
                TRIM(CAST(grupo AS VARCHAR))                    AS grupo,
                TRIM(CAST(rubro AS VARCHAR))                    AS rubro,
                CASE
                WHEN LOWER(TRIM(CAST(central_mayorista AS VARCHAR))) IN
                     ('pereira, la 41', 'pereira, la 41-impala')
                    THEN 'Pereira, La 41'
                WHEN LOWER(TRIM(CAST(central_mayorista AS VARCHAR))) IN
                     ('cali, santa elena', 'cali, santa helena')
                    THEN 'Cali, Santa Elena'
                ELSE TRIM(CAST(central_mayorista AS VARCHAR))
            END                                             AS central_mayorista,
                UPPER(TRIM(CAST(pais_origen AS VARCHAR)))       AS pais_origen,
                CAST(lon_orig AS DOUBLE)                        AS lon_orig,
                CAST(lat_orig AS DOUBLE)                        AS lat_orig,
                CAST(lon_dest AS DOUBLE)                        AS lon_dest,
                CAST(lat_dest AS DOUBLE)                        AS lat_dest,
                CAST(toneladas AS DOUBLE)                       AS toneladas
            FROM read_parquet('{RUTA_INTL_SQL}')
        """)
    return con

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
    deptos = list(dict.fromkeys(deptos_raw))
    if "INTERNACIONAL" not in deptos:
        deptos.append("INTERNACIONAL")
    municipios_raw = ca.execute("""
        SELECT DISTINCT
            municipio_origen,
            departamento_origen,
            CASE
                WHEN UPPER(municipio_origen) = 'UNE'      THEN '25845'
                WHEN UPPER(municipio_origen) = 'FÓMEQUE'  THEN '25279'
                WHEN UPPER(municipio_origen) = 'FOMEQUE'  THEN '25279'
                WHEN UPPER(municipio_origen) = 'CERRITO'  THEN '68162'
                ELSE cod_municipio
            END AS cod_municipio
        FROM lineas WHERE municipio_origen IS NOT NULL
        ORDER BY municipio_origen
    """).df()
    rango  = ca.execute("SELECT MIN(fecha_mes) AS fmin, MAX(fecha_mes) AS fmax FROM lineas WHERE fecha_mes <= '2026-03-01'").df()
    fecha_min = pd.to_datetime(rango.loc[0,"fmin"]).date()
    fecha_max = pd.to_datetime(rango.loc[0,"fmax"]).date()
    return grupos, rubros, centrales, deptos, municipios_raw, fecha_min, fecha_max

@st.cache_data(show_spinner=False)
def consultar_paises(mtime_intl):
    """Devuelve solo los países que tienen datos en el parquet, excluyendo Colombia."""
    if not RUTA_INTERNACIONALES.exists():
        return []
    con = get_con_intl(mtime_intl)
    df = con.execute("""
        SELECT DISTINCT TRIM(pais_origen) AS pais_origen
        FROM internacionales
        WHERE UPPER(TRIM(pais_origen)) NOT IN ('COLOMBIA', '')
          AND pais_origen IS NOT NULL
        ORDER BY 1
    """).df()
    return ["COLOMBIA"] + df["pais_origen"].tolist()

# =========================================================
# WHERE BUILDER
# =========================================================

BOGOTA_VARIANTES = {
    'BOGOTÁ, D. C.','BOGOTA, D.C.','BOGOTÁ D.C.','BOGOTA D.C.',
    'BOGOTA','BOGOTÁ DC','BOGOTA DC','BOGOTÁ, D.C.','BOGOTÁ'
}

def build_where_ab(fecha_ini, fecha_fin, semestre, grupo, rubros, centrales, deptos, muns_prio=None, municipios=None):
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
    if municipios:
        ml = list(municipios)
        c.append(f"municipio_origen IN ({','.join(['?']*len(ml))})"); p.extend(ml)
    if muns_prio:
        ml = list(muns_prio)
        cod_expr = """CASE
            WHEN UPPER(municipio_origen)='UNE'     THEN '25845'
            WHEN UPPER(municipio_origen)='FÓMEQUE' THEN '25279'
            WHEN UPPER(municipio_origen)='FOMEQUE' THEN '25279'
            WHEN UPPER(municipio_origen)='CERRITO' THEN '68162'
            ELSE cod_municipio
        END"""
        c.append(f"{cod_expr} IN ({','.join(['?']*len(ml))})"); p.extend(ml)
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
                    centrales_t, deptos_t, muns_prio_t, municipios_t, mtime):
    con  = get_con_abast(mtime)
    w, p = build_where_ab(fecha_ini, fecha_fin, semestre, grupo, rubro,
                          centrales_t, deptos_t, set(muns_prio_t) if muns_prio_t else None,
                          set(municipios_t) if municipios_t else None)
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


@st.cache_data(show_spinner=False)
def consultar_precios_por_flujo(
    fecha_ini, fecha_fin, semestre, grupo, rubros,
    centrales_t, deptos_t, muns_prio_t, municipios_t,
    mtime_ab, mtime_pr
):
    """
    Calcula el precio asociado al abastecimiento real:
    1) agrega toneladas por municipio-central-mes;
    2) calcula el precio SIPSA por central-mes;
    3) cruza ambos solo cuando existe flujo;
    4) pondera el precio por las toneladas efectivamente enviadas.

    Devuelve precio por municipio, precio por flujo OD, serie mensual ponderada
    y el precio general ponderado.
    """
    if len(rubros) != 1:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None

    ca = get_con_abast(mtime_ab)
    wa, pa = build_where_ab(
        fecha_ini, fecha_fin, semestre, grupo, rubros,
        centrales_t, deptos_t,
        set(muns_prio_t) if muns_prio_t else None,
        set(municipios_t) if municipios_t else None
    )

    cod_expr = """CASE
        WHEN UPPER(municipio_origen)='UNE'     THEN '25845'
        WHEN UPPER(municipio_origen)='FÓMEQUE' THEN '25279'
        WHEN UPPER(municipio_origen)='FOMEQUE' THEN '25279'
        WHEN UPPER(municipio_origen)='CERRITO' THEN '68162'
        ELSE cod_municipio
    END"""

    flujos_mes = ca.execute(f"""
        SELECT
            etiqueta_mes,
            {cod_expr} AS cod_municipio,
            central_mayorista,
            SUM(toneladas) AS toneladas_total
        FROM lineas
        WHERE {wa}
        GROUP BY 1,2,3
    """, pa).df()

    if flujos_mes.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None

    cp = get_con_precios(mtime_pr)
    wp, pp = build_where_pr(
        fecha_ini, fecha_fin, semestre, grupo, rubros, centrales_t
    )
    precios_mes = cp.execute(f"""
        SELECT
            etiqueta_mes,
            central_mayorista,
            AVG(precio) AS precio_central_mes
        FROM precios
        WHERE {wp}
        GROUP BY 1,2
    """, pp).df()

    if precios_mes.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None

    cruce = flujos_mes.merge(
        precios_mes,
        on=["etiqueta_mes","central_mayorista"],
        how="left"
    )
    cruce = cruce[
        cruce["precio_central_mes"].notna()
        & cruce["toneladas_total"].notna()
        & (cruce["toneladas_total"] > 0)
    ].copy()

    if cruce.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None

    cruce["valor_precio"] = cruce["precio_central_mes"] * cruce["toneladas_total"]

    pm = (
        cruce.groupby("cod_municipio", as_index=False)
        .agg(
            valor_precio=("valor_precio","sum"),
            toneladas_con_precio=("toneladas_total","sum")
        )
    )
    pm["precio_municipio"] = pm["valor_precio"] / pm["toneladas_con_precio"]
    pm = pm[["cod_municipio","precio_municipio","toneladas_con_precio"]]

    pf = (
        cruce.groupby(["cod_municipio","central_mayorista"], as_index=False)
        .agg(
            valor_precio=("valor_precio","sum"),
            toneladas_con_precio=("toneladas_total","sum")
        )
    )
    pf["precio_flujo"] = pf["valor_precio"] / pf["toneladas_con_precio"]
    pf = pf[["cod_municipio","central_mayorista","precio_flujo"]]

    serie = (
        cruce.groupby("etiqueta_mes", as_index=False)
        .agg(
            valor_precio=("valor_precio","sum"),
            toneladas_con_precio=("toneladas_total","sum")
        )
        .sort_values("etiqueta_mes")
    )
    serie["precio_promedio"] = serie["valor_precio"] / serie["toneladas_con_precio"]
    serie = serie[["etiqueta_mes","precio_promedio","toneladas_con_precio"]]

    toneladas_precio = cruce["toneladas_total"].sum()
    precio_general = (
        cruce["valor_precio"].sum() / toneladas_precio
        if toneladas_precio > 0 else None
    )

    return pm, pf, serie, precio_general


# =========================================================
# CARGA INICIAL
# =========================================================

mtime_ab   = obtener_mtime(RUTA_LINEAS)
mtime_pr   = obtener_mtime(RUTA_PRECIOS)
mtime_mun  = obtener_mtime(RUTA_MUNICIPIOS)
mtime_intl = obtener_mtime(RUTA_INTERNACIONALES)

municipios = cargar_municipios(mtime_mun)
grupos, rubros, centrales, deptos, municipios_df, fecha_min_g, fecha_max_g = consultar_catalogos(mtime_ab, mtime_pr)
paises_lista = consultar_paises(mtime_intl)

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

with st.container():

    # ── Estado de filtros SARA (fuente: base territorial suministrada) ──
    solo_priorizados = st.session_state.get("_cb_solo_prio", False)
    sara_region_sel  = st.session_state.get("sara_region", "Sin filtro territorial")
    sara_roles_sel   = st.session_state.get("sara_roles", [])
    sara_muestra     = st.session_state.get("sara_muestra", False)
    sara_conmutados  = st.session_state.get("sara_conmutados", False)
    territorio_sel   = st.session_state.get("sara_territorios", [])

    def _codigos_por_deptos(deptos_objetivo):
        if not deptos_objetivo or "cod_municipio" not in municipios_df.columns:
            return set()
        mask = municipios_df["departamento_origen"].fillna("").str.upper().isin(
            {str(d).upper() for d in deptos_objetivo}
        )
        return set(
            municipios_df.loc[mask, "cod_municipio"]
            .apply(normalizar_codigo_5).dropna().astype(str).tolist()
        )

    def _interseccion_sara(region_sel, roles_sel, muestra_sel, conmutados_sel, territorios_sel):
        conjuntos = []
        if region_sel == "Región Central completa":
            conjuntos.append(_codigos_por_deptos(DEPTOS_REGION_CENTRAL))
        elif region_sel == "Región Central externa":
            conjuntos.append(_codigos_por_deptos(DEPTOS_REGION_CENTRAL_EXTERNA))
        elif region_sel == "Región Metropolitana Bogotá-Cundinamarca":
            conjuntos.append(set(MUNS_REGION_METROPOLITANA))

        if roles_sel:
            rol_union = set()
            for rol in roles_sel:
                rol_union.update(MUNS_PRIORI_ROL.get(rol, set()))
            conjuntos.append(rol_union)

        if muestra_sel:
            conjuntos.append(set(MUNS_MUESTRA_178))
        if conmutados_sel:
            conjuntos.append(set(MUNS_CONMUTADOS))
        if territorios_sel:
            terr_union = set()
            for t in territorios_sel:
                terr_union.update(TERRITORIOS_FUNC.get(t, set()))
            conjuntos.append(terr_union)

        if not conjuntos:
            return None
        resultado = set(conjuntos[0])
        for conjunto in conjuntos[1:]:
            resultado &= set(conjunto)
        return resultado

    _muns_sara_pre = _interseccion_sara(
        sara_region_sel, sara_roles_sel, sara_muestra, sara_conmutados, territorio_sel
    )

    st.markdown('<div class="filter-wrap">', unsafe_allow_html=True)

    # ── Fila 1: Grupo | Rubro | Central | Periodo | Fechas ───
    f1, f2, f3, f4, f5 = st.columns([1.1, 1.5, 1.5, 1.0, 1.4])
    with f1:
        grupo_sel = st.selectbox("Grupo", ["Todos"] + grupos, index=0)
    with f2:
        if solo_priorizados:
            rubros_f_codigos = sorted(RUBROS_PRIORIZADOS_SARA, key=lambda r: label_rubro(r))
        elif grupo_sel != "Todos":
            con_tmp = get_con_abast(mtime_ab)
            rubros_f_codigos = con_tmp.execute(
                "SELECT DISTINCT rubro FROM lineas WHERE grupo=? AND rubro IS NOT NULL ORDER BY rubro",
                [grupo_sel]
            ).df()["rubro"].tolist()
        else:
            rubros_f_codigos = sorted(rubros, key=lambda r: label_rubro(r))
        rubros_f_labels = [label_rubro(r) for r in rubros_f_codigos]
        rubros_sel_labels = st.multiselect(
            "Rubro", options=rubros_f_labels, default=[], placeholder="Todos los rubros"
        )
        rubros_sel = [codigo_rubro(l) for l in rubros_sel_labels]
    with f3:
        centrales_sel = st.multiselect("Central mayorista", options=centrales, default=[])
    with f4:
        semestre_sel = st.selectbox("Periodo", ["Todos","Primer semestre","Segundo semestre"], index=0)
    with f5:
        rango = st.date_input(
            "Fechas", value=(fecha_min_g, fecha_max_g),
            min_value=fecha_min_g, max_value=fecha_max_g
        )

    st.markdown('<div style="height:0.4rem;"></div>', unsafe_allow_html=True)

    # ── Fila 2: Depto | Municipio | País | Toggle | Slider ───
    g1, g2, g3, g4, g5 = st.columns([1.1, 1.2, 1.2, 0.9, 1.0])
    with g1:
        deptos_nacionales = [d for d in deptos if d != "INTERNACIONAL"]
        if _muns_sara_pre is not None and "cod_municipio" in municipios_df.columns:
            _cod_tmp = municipios_df["cod_municipio"].apply(normalizar_codigo_5).astype(str)
            _deptos_sara = set(
                municipios_df.loc[_cod_tmp.isin(_muns_sara_pre), "departamento_origen"]
                .dropna().astype(str).tolist()
            )
            deptos_opciones = [d for d in deptos_nacionales if d in _deptos_sara]
        else:
            deptos_opciones = deptos_nacionales
        deptos_sel = st.multiselect(
            "Depto. origen", deptos_opciones, default=[], placeholder="Todos los departamentos"
        )

    with g2:
        if deptos_sel:
            muns_base_df = municipios_df[
                municipios_df["departamento_origen"].isin(deptos_sel)
            ].copy()
        else:
            muns_base_df = municipios_df.copy()

        if _muns_sara_pre is not None and "cod_municipio" in muns_base_df.columns:
            _cod_m = muns_base_df["cod_municipio"].apply(normalizar_codigo_5).astype(str)
            muns_base_df = muns_base_df[_cod_m.isin(_muns_sara_pre)]

        muns_opciones = sorted(muns_base_df["municipio_origen"].dropna().unique().tolist())
        municipios_sel = st.multiselect(
            "Municipio origen", options=muns_opciones,
            default=[], placeholder="Todos los municipios"
        )

    with g3:
        paises_sel = st.multiselect(
            "País de origen", options=paises_lista,
            default=[], placeholder="Todos los países"
        )
    with g4:
        st.markdown(
            '<div style="font-size:0.82rem;color:#9EABC0;padding-top:1.6rem;">'
            'Flujos internacionales en mapa</div>',
            unsafe_allow_html=True
        )
        mostrar_flujos_intl = st.toggle(
            "Flujos internacionales en mapa", value=False, label_visibility="collapsed"
        )
    with g5:
        st.markdown(
            '<div style="font-size:0.82rem;color:#9EABC0;padding-top:1.6rem;">'
            '🔀 Máx. flujos en mapa</div>', unsafe_allow_html=True
        )
        max_flujos = st.slider(
            "Máx. flujos", min_value=100, max_value=2000,
            value=MAX_LINEAS_MAPA, step=100, label_visibility="collapsed"
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Filtros SARA agrupados y alineados con Municipios_Nacional_Base_Completa.xlsx ──
    with st.expander("**FILTROS DEL PROYECTO SARA**", expanded=False):
        sc1, sc2, sc3 = st.columns([1.05, 1.05, 1.25])

        with sc1:
            st.markdown("**Producto y alcance territorial**")
            solo_priorizados = st.checkbox(
                "Rubros priorizados SARA", value=False, key="_cb_solo_prio",
                help="Restringe el selector de rubro a los rubros priorizados por el proyecto."
            )
            sara_region_sel = st.selectbox(
                "Ámbito territorial",
                [
                    "Sin filtro territorial",
                    "Región Central completa",
                    "Región Central externa",
                    "Región Metropolitana Bogotá-Cundinamarca",
                ],
                key="sara_region",
                help=(
                    "Según la base territorial: Región Central completa = 353 municipios; "
                    "Región Central externa = 236; Región Metropolitana Bogotá-Cundinamarca = 117."
                )
            )

        with sc2:
            st.markdown("**Priorización y muestra UTF/COL/178**")
            sara_roles_sel = st.multiselect(
                "Rol de priorización inicial",
                ["Oferta", "Demanda", "Oferta y demanda"],
                default=[], key="sara_roles",
                help="Categorías exactas de la variable muni_priori178."
            )
            sara_muestra = st.checkbox(
                "Muestra final de información primaria", value=False, key="sara_muestra",
                help="50 municipios marcados con muni_muestra178 = 1."
            )
            sara_conmutados = st.checkbox(
                "Municipios conmutados con Bogotá", value=False, key="sara_conmutados",
                help="10 municipios marcados con muni_conmutados = 1."
            )

        with sc3:
            st.markdown("**Territorio funcional de la RMBC**")
            territorio_sel = st.multiselect(
                "Territorio funcional",
                sorted(TERRITORIOS_FUNC.keys()),
                default=[], key="sara_territorios",
                placeholder="Todos los territorios"
            )
            st.caption(
                "Las selecciones de distintas categorías se combinan por intersección; "
                "varias opciones dentro de un mismo selector se combinan por unión."
            )

    # Aplicar filtro de priorizados al selector de rubros.
    if solo_priorizados and not rubros_sel:
        rubros_sel = list(RUBROS_PRIORIZADOS_SARA)

    muns_prio_consulta = _interseccion_sara(
        sara_region_sel, sara_roles_sel, sara_muestra, sara_conmutados, territorio_sel
    )

    # Variables auxiliares para mapa/leyenda.
    prio_region_central = sara_region_sel == "Región Central completa"
    prio_region_central_externa = sara_region_sel == "Región Central externa"
    prio_region_metropolitana = sara_region_sel == "Región Metropolitana Bogotá-Cundinamarca"
    muns_territorio_consulta = set()
    for t in territorio_sel:
        muns_territorio_consulta.update(TERRITORIOS_FUNC.get(t, set()))
    muns_prio = set()
    for rol in sara_roles_sel:
        muns_prio.update(MUNS_PRIORI_ROL.get(rol, set()))

    _sara_desc = []
    if sara_region_sel != "Sin filtro territorial":
        _sara_desc.append(sara_region_sel)
    if sara_roles_sel:
        _sara_desc.append("Rol: " + ", ".join(sara_roles_sel))
    if sara_muestra:
        _sara_desc.append("Muestra final")
    if sara_conmutados:
        _sara_desc.append("Municipios conmutados")
    if territorio_sel:
        _sara_desc.append("Territorio funcional")
    sara_filtro_label = (
        _sara_desc[0] if len(_sara_desc) == 1
        else "Filtros SARA (intersección)" if _sara_desc
        else ""
    )

    if isinstance(rango, tuple) and len(rango) == 2:
        fecha_ini, fecha_fin = rango
    else:
        fecha_ini, fecha_fin = fecha_min_g, fecha_max_g

    centrales_t  = tuple(centrales_sel)
    deptos_t     = tuple(deptos_sel)
    rubros_t     = tuple(rubros_sel)
    municipios_t = tuple(municipios_sel) if municipios_sel else ()
    if muns_prio_consulta is None:
        muns_prio_t = ()
    elif muns_prio_consulta:
        muns_prio_t = tuple(sorted(muns_prio_consulta))
    else:
        # Intersección SARA válida pero sin municipios: forzar consulta sin resultados.
        muns_prio_t = ("__SIN_COINCIDENCIAS_SARA__",)

    # ── Lógica de país de origen ─────────────────────────────
    # País vacío = todos los orígenes. COLOMBIA permite aislar los flujos internos.
    paises_extranjeros_sel = [p for p in paises_sel if p != "COLOMBIA"]
    mostrar_nacional = (not paises_sel) or ("COLOMBIA" in paises_sel)
    mostrar_internacional = (not paises_sel) or bool(paises_extranjeros_sel)

    # Si se filtra por departamento/municipio y no se pidió explícitamente un país extranjero,
    # el resultado queda naturalmente restringido a Colombia.
    if (deptos_sel or municipios_sel or muns_prio_consulta is not None) and not paises_extranjeros_sel:
        mostrar_internacional = False

    paises_t = tuple(paises_extranjeros_sel)
    incluir_intl = mostrar_internacional
    solo_internacional = mostrar_internacional and not mostrar_nacional

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

    _empty_met = pd.DataFrame([{"vol_filtro":0,"mun_activos":0,"cent_activas":0}])
    _empty_tot = pd.DataFrame([{"vol_total":0}])
    _empty_rap = pd.DataFrame([{"vol_rape":0}])

    if not mostrar_nacional:
        # País(es) extranjero(s) sin Colombia: vaciar abastecimiento nacional.
        met_df      = _empty_met
        tot_df      = _empty_tot
        rape_df     = _empty_rap
        rank_df     = pd.DataFrame()
        serie_ab_df = pd.DataFrame()
        flujos_df   = pd.DataFrame()
        sankey_df   = pd.DataFrame()
    else:
        met_df, tot_df, rape_df, rank_df, serie_ab_df, flujos_df, sankey_df = consultar_abast(
            fecha_ini, fecha_fin, semestre_sel, grupo_sel, rubros_t,
            centrales_t, deptos_t, muns_prio_t, municipios_t, mtime_ab
        )

    # Los precios SIPSA son de la central de destino; siguen siendo consultables
    # aunque el origen seleccionado sea internacional.
    serie_pr_df, precios_central_df = consultar_precios(
        fecha_ini, fecha_fin, semestre_sel, grupo_sel, rubros_t,
        centrales_t, mtime_pr
    )

    precios_municipio_df = pd.DataFrame()
    precios_flujo_df = pd.DataFrame()
    precio_prom_flujo = None
    if mostrar_nacional and tiene_rubro_unico:
        (
            precios_municipio_df,
            precios_flujo_df,
            serie_pr_flujo_df,
            precio_prom_flujo,
        ) = consultar_precios_por_flujo(
            fecha_ini, fecha_fin, semestre_sel, grupo_sel, rubros_t,
            centrales_t, deptos_t, muns_prio_t, municipios_t,
            mtime_ab, mtime_pr
        )
        if not serie_pr_flujo_df.empty:
            # La serie de precio también se pondera por las toneladas realmente
            # abastecidas a cada central en cada mes.
            serie_pr_df = serie_pr_flujo_df

    # ── Datos internacionales ─────────────────────────────────
    @st.cache_data(show_spinner=False)
    def consultar_internacionales(fecha_ini, fecha_fin, semestre, grupo, rubros,
                                   centrales_t, paises_t, mtime_intl):
        if not RUTA_INTERNACIONALES.exists():
            return pd.DataFrame()
        con = get_con_intl(mtime_intl)
        c = [
            "fecha_mes BETWEEN ? AND ?",
            "UPPER(TRIM(pais_origen)) <> 'COLOMBIA'"
        ]
        p = [fecha_ini.isoformat(), fecha_fin.isoformat()]
        if semestre == "Primer semestre":    c.append("mes BETWEEN 1 AND 6")
        elif semestre == "Segundo semestre": c.append("mes BETWEEN 7 AND 12")
        if rubros:
            c.append(f"rubro IN ({','.join(['?']*len(rubros))})"); p.extend(list(rubros))
        elif grupo and grupo != "Todos":
            c.append("grupo = ?"); p.append(grupo)
        if centrales_t:
            c.append(f"central_mayorista IN ({','.join(['?']*len(centrales_t))})"); p.extend(list(centrales_t))
        if paises_t:
            c.append(f"pais_origen IN ({','.join(['?']*len(paises_t))})"); p.extend(list(paises_t))
        w = " AND ".join(c)
        try:
            df = con.execute(f"""
                SELECT pais_origen, central_mayorista,
                       AVG(lon_orig) AS lon_orig, AVG(lat_orig) AS lat_orig,
                       AVG(lon_dest) AS lon_dest, AVG(lat_dest) AS lat_dest,
                       SUM(toneladas) AS toneladas_total
                FROM internacionales WHERE {w}
                GROUP BY 1,2 ORDER BY toneladas_total DESC
            """, p).df()
        except Exception:
            df = pd.DataFrame()
        return df

    if incluir_intl:
        intl_raw = consultar_internacionales(
            fecha_ini, fecha_fin, semestre_sel, grupo_sel, rubros_t,
            centrales_t, paises_t, mtime_intl
        )
    else:
        intl_raw = pd.DataFrame()
    if not intl_raw.empty:
        # Lookup de coordenadas robusto: normaliza acentos y mayúsculas en ambos lados
        # para que MÉXICO, Mexico, MEXICO, etc. todos encuentren su entrada en PAISES_COORDS.
        def _norm_pais(s):
            s = str(s).upper().strip()
            return "".join(c for c in unicodedata.normalize("NFD", s)
                           if unicodedata.category(c) != "Mn")
        _lookup_lon = {_norm_pais(k): v[0] for k, v in PAISES_COORDS.items()}
        _lookup_lat = {_norm_pais(k): v[1] for k, v in PAISES_COORDS.items()}
        _coords_lon = intl_raw["pais_origen"].map(lambda p: _lookup_lon.get(_norm_pais(p)))
        _coords_lat = intl_raw["pais_origen"].map(lambda p: _lookup_lat.get(_norm_pais(p)))
        intl_raw["lon_orig"] = intl_raw["lon_orig"].combine_first(_coords_lon)
        intl_raw["lat_orig"] = intl_raw["lat_orig"].combine_first(_coords_lat)
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

    # Toneladas internacionales — se suman si hay datos intl activos
    vol_intl         = intl_raw["toneladas_total"].sum() if not intl_raw.empty else 0.0
    vol_filtro_total = vol_filtro + vol_intl

    # Precio promedio general — solo válido con un único rubro seleccionado
    precio_prom_general = (
        precio_prom_flujo
        if precio_prom_flujo is not None
        else (
            serie_pr_df["precio_promedio"].mean()
            if not serie_pr_df.empty and tiene_rubro_unico
            else None
        )
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

        # ── Precio por municipio condicionado al flujo real ────────
        # El precio no se asigna simplemente desde la central. Se calcula usando
        # únicamente los meses y centrales a los que el municipio efectivamente
        # envió toneladas, ponderando cada precio central-mes por ese volumen.
        if tiene_rubro_unico and not precios_municipio_df.empty:
            rk = rk.merge(
                precios_municipio_df[["cod_municipio","precio_municipio"]],
                on="cod_municipio", how="left"
            )
        else:
            rk["precio_municipio"] = np.nan

        if not flujos_df.empty:
            if tiene_rubro_unico and not precios_municipio_df.empty:
                flujos_df = flujos_df.merge(
                    precios_municipio_df[["cod_municipio","precio_municipio"]],
                    on="cod_municipio", how="left"
                )
            else:
                flujos_df["precio_municipio"] = np.nan

            if tiene_rubro_unico and not precios_flujo_df.empty:
                flujos_df = flujos_df.merge(
                    precios_flujo_df,
                    on=["cod_municipio","central_mayorista"],
                    how="left"
                )
            else:
                flujos_df["precio_flujo"] = np.nan

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
            flujos_df["precio_fmt"] = flujos_df["precio_municipio"].map(formatear_cop) \
                                      if "precio_municipio" in flujos_df.columns \
                                      else "Sin dato"
            flujos_df["precio_flujo_fmt"] = flujos_df["precio_flujo"].map(formatear_cop) \
                                            if "precio_flujo" in flujos_df.columns \
                                            else "Sin dato"
        else:
            pct_cobertura = 0

        # Agregar internacionales al ranking (para mostrar en tabla)
        if not intl_raw.empty:
            intl_rk_agg = intl_raw.groupby("pais_origen", as_index=False).agg(
                toneladas_total=("toneladas_total","sum"),
                meses_participacion=("toneladas_total","count"),
            )
            intl_rk_agg["municipio_origen"]    = "* " + intl_rk_agg["pais_origen"]
            intl_rk_agg["departamento_origen"] = "Internacional"
            intl_rk_agg["cod_municipio"]       = "intl"
            intl_rk_agg["precio_municipio"]    = np.nan
            intl_rk_agg["ventaja_precio"]      = np.nan
            intl_rk_agg["part_filtro"] = (
                intl_rk_agg["toneladas_total"] / vol_filtro_total * 100
                if vol_filtro_total > 0 else 0.0
            )
            intl_rk_agg["part_total"]  = 0.0
            intl_rk_agg["part_rape"]   = 0.0
            intl_rk_agg["indice"]      = np.nan
            cols_comunes = ["municipio_origen","departamento_origen","cod_municipio",
                            "toneladas_total","meses_participacion","part_filtro",
                            "part_total","part_rape","precio_municipio","ventaja_precio","indice"]
            for col in cols_comunes:
                if col not in rk.columns: rk[col] = np.nan
                if col not in intl_rk_agg.columns: intl_rk_agg[col] = np.nan
            rk = pd.concat([rk, intl_rk_agg[cols_comunes]], ignore_index=True)
            rk = rk.sort_values("toneladas_total", ascending=False).reset_index(drop=True)
            rk["ranking"] = rk.index + 1
    else:
        rk        = pd.DataFrame()
        top30     = set()
        flujos_df = pd.DataFrame()
        sk_top    = pd.DataFrame(columns=["municipio_origen","central_mayorista","toneladas_total"])
        pct_cobertura = 0

    # ── Orígenes internacionales en la tabla, incluso si no hay flujo nacional ──
    if not intl_raw.empty:
        _ya_tiene_intl = (
            not rk.empty
            and "departamento_origen" in rk.columns
            and rk["departamento_origen"].astype(str).eq("Internacional").any()
        )
        if not _ya_tiene_intl:
            intl_rk_agg = (
                intl_raw.groupby("pais_origen", as_index=False)
                .agg(toneladas_total=("toneladas_total","sum"))
            )
            intl_rk_agg["municipio_origen"] = "🌎 " + intl_rk_agg["pais_origen"]
            intl_rk_agg["departamento_origen"] = "Internacional"
            intl_rk_agg["cod_municipio"] = "intl"
            intl_rk_agg["meses_participacion"] = np.nan
            intl_rk_agg["precio_municipio"] = np.nan
            intl_rk_agg["ventaja_precio"] = np.nan
            intl_rk_agg["part_filtro"] = (
                intl_rk_agg["toneladas_total"] / vol_filtro_total * 100
                if vol_filtro_total > 0 else 0.0
            )
            intl_rk_agg["part_total"] = 0.0
            intl_rk_agg["part_rape"] = 0.0
            intl_rk_agg["indice"] = np.nan

            cols_comunes = [
                "municipio_origen","departamento_origen","cod_municipio",
                "toneladas_total","meses_participacion","part_filtro",
                "part_total","part_rape","precio_municipio","ventaja_precio","indice"
            ]
            if rk.empty:
                rk = pd.DataFrame(columns=cols_comunes)
            for col in cols_comunes:
                if col not in rk.columns:
                    rk[col] = np.nan
                if col not in intl_rk_agg.columns:
                    intl_rk_agg[col] = np.nan
            rk = pd.concat([rk, intl_rk_agg[cols_comunes]], ignore_index=True)
            rk = rk.sort_values("toneladas_total", ascending=False).reset_index(drop=True)
            rk["ranking"] = rk.index + 1

    # ── Sankey independiente del ranking ─────────────────────
    # Importante: usa los datos internacionales crudos, no solo los que tienen
    # coordenadas válidas para el mapa. Así un filtro por país nunca vacía el Sankey.
    _sk_parts = []
    if not sankey_df.empty:
        top_mun = (
            sankey_df.groupby("municipio_origen")["toneladas_total"]
            .sum().nlargest(10).index.tolist()
        )
        _sk_parts.append(
            sankey_df[sankey_df["municipio_origen"].isin(top_mun)].copy()
        )

    if not intl_raw.empty:
        intl_sk = (
            intl_raw.groupby(["pais_origen","central_mayorista"], as_index=False)
            .agg(toneladas_total=("toneladas_total","sum"))
            .rename(columns={"pais_origen":"municipio_origen"})
        )
        top_paises = (
            intl_raw.groupby("pais_origen")["toneladas_total"]
            .sum().nlargest(10).index.tolist()
        )
        intl_sk = intl_sk[intl_sk["municipio_origen"].isin(top_paises)].copy()
        intl_sk["municipio_origen"] = "🌎 " + intl_sk["municipio_origen"]
        _sk_parts.append(intl_sk)

    if _sk_parts:
        sk_top = pd.concat(_sk_parts, ignore_index=True)
    else:
        sk_top = pd.DataFrame(
            columns=["municipio_origen","central_mayorista","toneladas_total"]
        )

    # =========================================================
    # MAPA — polígonos con colores por estado
    # deptos_sel resalta municipios del departamento en amarillo
    # top30 en morado, resto gris
    # =========================================================

    deptos_sel_upper = {d.upper() for d in deptos_sel if d != "INTERNACIONAL"}

    # Municipios a resaltar: resultado exacto de la intersección de filtros SARA.
    muns_resalte = set(muns_prio_consulta) if muns_prio_consulta is not None else set()
    muns_region_central_set = set()
    top30_map = set() if solo_internacional else top30

    def color_fill(codigo, depto):
        cod = str(codigo)
        if cod in top30_map:
            return [110, 68, 255, 160]
        if muns_resalte and cod in muns_resalte:
            return [180, 190, 210, 90]
        if muns_region_central_set and cod in muns_region_central_set:
            return [170, 185, 200, 55]
        if deptos_sel_upper and str(depto).upper() in deptos_sel_upper:
            return [180, 190, 205, 60]
        return [40, 48, 62, 18]

    def color_line(codigo, depto):
        cod = str(codigo)
        if cod in top30_map:
            return [170, 130, 255, 240]
        if muns_resalte and cod in muns_resalte:
            return [210, 220, 240, 230]
        if muns_region_central_set and cod in muns_region_central_set:
            return [200, 215, 230, 180]
        if deptos_sel_upper and str(depto).upper() in deptos_sel_upper:
            return [200, 210, 225, 180]
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
    if not rk.empty and "precio_municipio" in rk.columns:
        _pm_map = rk[["cod_municipio","precio_municipio"]].copy()
        _pm_map["cod_municipio"] = _pm_map["cod_municipio"].apply(normalizar_codigo_5)
        _pm_map = _pm_map.dropna(subset=["cod_municipio"]).drop_duplicates("cod_municipio")
        mun_web = mun_web.merge(
            _pm_map, left_on="codigo_origen", right_on="cod_municipio", how="left"
        )
    else:
        mun_web["precio_municipio"] = np.nan

    mun_web["tipo_elemento"] = "Municipio"
    mun_web["detalle_1"] = "Nombre: "       + mun_web["nombre_municipio"].fillna("Sin nombre").astype(str)
    mun_web["detalle_2"] = "Departamento: " + mun_web["departamento"].fillna("Sin dato").astype(str)
    mun_web["detalle_3"] = "Código: "       + mun_web["codigo_origen"].fillna("").astype(str)
    mun_web["detalle_4"] = "Precio estimado: " + mun_web["precio_municipio"].map(formatear_cop)

    geojson_mun = json.loads(
        mun_web[["fill_color","line_color","tipo_elemento",
                 "detalle_1","detalle_2","detalle_3","detalle_4","geometry"]].to_json()
    )

    # Centrales como puntos — fusionar nacionales e internacionales
    # para que siempre sean visibles independientemente del filtro activo
    _cent_parts = []
    if not flujos_df.empty:
        _c_nac = (
            flujos_df.groupby("central_mayorista", as_index=False)
            .agg(lon=("lon_dest","first"), lat=("lat_dest","first"),
                 nombre=("central_mayorista","first"))
        ).dropna(subset=["lon","lat"])
        _cent_parts.append(_c_nac)
    if not intl_df.empty:
        _c_intl = intl_df.groupby("central_mayorista", as_index=False).agg(
            lon=("lon_dest","first"), lat=("lat_dest","first"),
            nombre=("central_mayorista","first")
        ).dropna(subset=["lon","lat"])
        _cent_parts.append(_c_intl)
    if _cent_parts:
        cent_pts = pd.concat(_cent_parts, ignore_index=True)
        cent_pts = cent_pts.drop_duplicates(subset=["central_mayorista"]).reset_index(drop=True)
        if tiene_rubro_unico and not precios_central_df.empty:
            cent_pts = cent_pts.merge(
                precios_central_df[["central_mayorista","precio_central"]],
                on="central_mayorista", how="left"
            )
        else:
            cent_pts["precio_central"] = np.nan
        cent_pts["tipo_elemento"] = "Central mayorista"
        cent_pts["detalle_1"]     = "Central: " + cent_pts["nombre"].fillna("")
        cent_pts["detalle_2"]     = "Precio SIPSA central: " + cent_pts["precio_central"].map(formatear_cop)
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
        flujos_df["detalle_4"]     = "Precio mun.: " + flujos_df["precio_fmt"].fillna("Sin dato") + " | Precio OD: " + flujos_df["precio_flujo_fmt"].fillna("Sin dato")

    # =========================================================
    # LAYOUT PRINCIPAL
    # =========================================================

    @st.fragment
    def render_principal(vol_filtro_total, mun_act, cent_act, precio_prom_general,
                         geojson_mun, flujos_df, cent_pts, intl_df,
                         serie_ab_df, serie_pr_df, sk_top, nivel_sel,
                         deptos_sel, tiene_rubro_unico, max_flujos, pct_cobertura,
                         mostrar_flujos_intl):

        left_col, center_col, right_col = st.columns([1.05, 3.8, 1.45], gap="small")

        # ── IZQUIERDA: Indicadores + Leyenda ──────────────────
        with left_col:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown('<div class="panel-title">Indicadores principales</div>', unsafe_allow_html=True)

            if tiene_rubro_unico and precio_prom_general is not None:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Precio promedio</div>
                    <div class="metric-value" style="font-size:1.65rem;">$ {precio_prom_general:,.0f}</div>
                    <div class="metric-small">Mercado filtrado ($/kg)</div>
                </div>""", unsafe_allow_html=True)

            vol_intl = intl_df["toneladas_total"].sum() if not intl_df.empty else 0

            # Centrales activas bajo los filtros actuales.
            # Se calcula como la UNION de centrales presentes en los flujos nacionales
            # e internacionales ya filtrados. Así Colombia + países extranjeros no
            # subestima el indicador y una misma central se cuenta una sola vez.
            centrales_activas_set = set()
            if not sankey_df.empty and "central_mayorista" in sankey_df.columns:
                centrales_activas_set.update(
                    sankey_df["central_mayorista"].dropna().astype(str).str.strip().tolist()
                )
            if not intl_raw.empty and "central_mayorista" in intl_raw.columns:
                centrales_activas_set.update(
                    intl_raw["central_mayorista"].dropna().astype(str).str.strip().tolist()
                )
            cent_act_display = len(centrales_activas_set)

            # Municipios origen: si solo internacional, contar países de origen
            mun_act_display = mun_act
            if mun_act == 0 and not intl_df.empty:
                mun_act_display = intl_df["pais_origen"].nunique()
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Toneladas abastecidas</div>
                <div class="metric-value">{vol_filtro_total:,.0f}</div>
                <div class="metric-small">Periodo filtrado</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Municipios origen activos</div>
                <div class="metric-value">{mun_act_display:,}</div>
                <div class="metric-small">Con flujo válido</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Centrales activas</div>
                <div class="metric-value">{cent_act_display}</div>
                <div class="metric-small">Bajo filtros actuales</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="panel-title">Leyenda</div>', unsafe_allow_html=True)
            leyenda_depto = ""
            if deptos_sel:
                leyenda_depto = '<div class="legend-item"><span class="legend-box" style="background:#B0B8C8;border:1px solid #888;"></span>Municipios depto. filtrado</div>'
            leyenda_intl = ""
            if mostrar_flujos_intl and not intl_df.empty:
                leyenda_intl = '<div class="legend-item"><span class="legend-box" style="background:#00C878;"></span>Flujos internacionales</div>'
            leyenda_territorio = ""
            if muns_resalte:
                label_t = sara_filtro_label or "Municipios filtrados SARA"
                leyenda_territorio = f'<div class="legend-item"><span class="legend-box" style="background:#B4BED2;border:2px solid #D2DCF0;"></span>{label_t}</div>'
            st.markdown(f"""
            <div class="legend-item"><span class="legend-box" style="background:#6E44FF;"></span>Top 30 abastecedores</div>
            {leyenda_depto}
            {leyenda_territorio}
            <div class="legend-item"><span class="legend-box" style="background:#F5A020;border-radius:50%;"></span>Municipio de origen activo</div>
            <div class="legend-item"><span class="legend-box" style="background:#F5B041;"></span>Arcos de flujo nacional</div>
            {leyenda_intl}
            <div class="legend-item"><span class="legend-box" style="background:#00D2FF;"></span>Central mayorista</div>
            <div class="small-note" style="margin-top:0.75rem;">
                <b>{max_flujos:,}</b> flujos visibles ({pct_cobertura:.0f}% del total).
                El origen nacional/internacional sigue el filtro «País de origen».
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── CENTRO: Mapa + Serie ──────────────────────────────
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
                    ["municipio_origen","departamento_origen","cod_municipio"], as_index=False
                ).agg(
                    lon=("lon_orig","first"), lat=("lat_orig","first"),
                    toneladas_total=("toneladas_total","sum"),
                    precio_municipio=("precio_municipio","first")
                )
                orig_pts = orig_pts.dropna(subset=["lon","lat"])
                orig_pts["tipo_elemento"] = "Municipio de origen"
                orig_pts["detalle_1"] = "Municipio: "    + orig_pts["municipio_origen"].fillna("")
                orig_pts["detalle_2"] = "Departamento: " + orig_pts["departamento_origen"].fillna("")
                orig_pts["detalle_3"] = "Toneladas: "    + orig_pts["toneladas_total"].map(formatear_ton)
                orig_pts["detalle_4"] = "Precio estimado: " + orig_pts["precio_municipio"].map(formatear_cop)
                layers.append(pdk.Layer(
                    "ScatterplotLayer", data=orig_pts,
                    get_position="[lon, lat]", get_radius=4200,
                    get_fill_color=[245,160,32,180], get_line_color=[255,210,100,220],
                    line_width_min_pixels=1, pickable=True, auto_highlight=True
                ))

            if mostrar_flujos_intl and not intl_df.empty:
                layers.append(pdk.Layer(
                    "ArcLayer", data=intl_df,
                    get_source_position=["lon_orig","lat_orig"],
                    get_target_position=["lon_dest","lat_dest"],
                    get_source_color=[0,200,120,130], get_target_color=[0,240,160,130],
                    get_width="ancho", width_scale=1, width_min_pixels=1,
                    pickable=True, auto_highlight=True
                ))
                orig_intl = intl_df.groupby("pais_origen", as_index=False).agg(
                    lon=("lon_orig","first"), lat=("lat_orig","first"),
                    toneladas_total=("toneladas_total","sum")
                ).dropna(subset=["lon","lat"])
                orig_intl["tipo_elemento"] = "Origen internacional"
                orig_intl["detalle_1"] = "Pais: "      + orig_intl["pais_origen"].fillna("")
                orig_intl["detalle_2"] = "Toneladas: " + orig_intl["toneladas_total"].map(formatear_ton)
                orig_intl["detalle_3"] = ""
                orig_intl["detalle_4"] = ""
                layers.append(pdk.Layer(
                    "ScatterplotLayer", data=orig_intl,
                    get_position="[lon, lat]", get_radius=80000,
                    get_fill_color=[0,200,120,200], get_line_color=[0,255,150,255],
                    line_width_min_pixels=2, pickable=True, auto_highlight=True
                ))

            # Centrales mayoristas — siempre visibles si hay datos (nacionales o internacionales)
            if not cent_pts.empty:
                layers.append(pdk.Layer(
                    "ScatterplotLayer", data=cent_pts,
                    get_position="[lon, lat]", get_radius=13500,
                    get_fill_color=[0,210,255,190], get_line_color=[170,245,255,255],
                    line_width_min_pixels=2, pickable=True
                ))
            elif not intl_df.empty:
                # Cuando solo hay internacionales, mostrar centrales de destino
                cent_intl = intl_df.groupby("central_mayorista", as_index=False).agg(
                    lon=("lon_dest","first"), lat=("lat_dest","first"),
                    toneladas_total=("toneladas_total","sum")
                ).dropna(subset=["lon","lat"])
                cent_intl["tipo_elemento"] = "Central mayorista"
                cent_intl["detalle_1"] = "Central: "   + cent_intl["central_mayorista"].fillna("")
                cent_intl["detalle_2"] = "Toneladas: " + cent_intl["toneladas_total"].map(formatear_ton)
                cent_intl["detalle_3"] = ""
                cent_intl["detalle_4"] = ""
                layers.append(pdk.Layer(
                    "ScatterplotLayer", data=cent_intl,
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

        # ── DERECHA: Sankey ───────────────────────────────────
        with right_col:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown(f'<div class="panel-title">Flujos hacia centrales mayoristas<br>'
                        f'<span style="font-weight:400;font-size:0.82rem;color:#AEB9C9;">{nivel_sel}</span></div>',
                        unsafe_allow_html=True)
            if not sk_top.empty:
                st.plotly_chart(construir_sankey(sk_top), use_container_width=True)
            else:
                st.info("No hay datos suficientes.")
            st.markdown('</div>', unsafe_allow_html=True)

    render_principal(
        vol_filtro_total=vol_filtro_total, mun_act=mun_act, cent_act=cent_act,
        precio_prom_general=precio_prom_general,
        geojson_mun=geojson_mun, flujos_df=flujos_df, cent_pts=cent_pts,
        intl_df=intl_df,
        serie_ab_df=serie_ab_df, serie_pr_df=serie_pr_df,
        sk_top=sk_top, nivel_sel=nivel_sel, deptos_sel=deptos_sel,
        tiene_rubro_unico=tiene_rubro_unico,
        max_flujos=max_flujos, pct_cobertura=pct_cobertura,
        mostrar_flujos_intl=mostrar_flujos_intl
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
            <b>Nota sobre el precio:</b> El dato solo se muestra cuando se selecciona un
            <b>único rubro</b>. El precio municipal se calcula con los precios SIPSA de las centrales
            y meses en los que ese municipio efectivamente aportó toneladas, ponderados por el volumen
            abastecido. Por eso dos municipios pueden tener precios ligeramente distintos incluso si
            comparten centrales de destino. Si no existe precio SIPSA compatible, aparece "Sin dato".
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
