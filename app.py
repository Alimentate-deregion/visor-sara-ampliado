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
    "Carne_res","Pescado","Huevos","Leche","Quesos_cuajadas","Panela"
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

# ── Territorios funcionales (fuente: Municipios_RAPE.xlsx) ─
TERRITORIOS_FUNC = {
    "Bogotá, D.C.":     ['11001'],
    "Noroccidental":    ['25148','25214','25260','25320','25394','25398','25402',
                         '25489','25491','25572','25592','25658','25769','25777',
                         '25799','25851','25862','25875','25885'],
    "Norte":            ['25126','25154','25175','25183','25200','25224','25258',
                         '25288','25295','25317','25407','25426','25436','25486',
                         '25513','25518','25653','25736','25745','25758','25772',
                         '25779','25781','25785','25793','25807','25817','25823',
                         '25843','25871','25873','25899'],
    "Occidental":       ['25019','25040','25086','25095','25099','25123','25168',
                         '25269','25286','25328','25430','25473','25580','25596',
                         '25662','25718','25867','25898'],
    "Oriente - Llanos": ['25151','25178','25281','25335','25339','25438','25530',
                         '25594','25845'],
    "Oriente Guavio":   ['25181','25279','25293','25297','25299','25322','25326',
                         '25372','25377','25839','25841'],
    "Suroccidental":    ['25001','25035','25053','25120','25245','25290','25307',
                         '25312','25324','25368','25386','25483','25488','25506',
                         '25524','25535','25599','25612','25645','25649','25740',
                         '25743','25754','25797','25805','25815','25878'],
}

# ── Región Central y Metropolitana ────────────────────────
DEPTOS_REGION_CENTRAL = {
    'BOYACÁ','BOYACA','CUNDINAMARCA','HUILA','META','TOLIMA',
    'BOGOTÁ','BOGOTÁ, D.C.','BOGOTA','BOGOTA D.C.','BOGOTÁ D.C.'
}
MUNS_REGION_METROPOLITANA = {'11001','25754','25290'}  # Bogotá, Soacha, Fusagasugá

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
                TRIM(CAST(central_mayorista AS VARCHAR))        AS central_mayorista,
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
    return df["pais_origen"].tolist()

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

mtime_ab   = obtener_mtime(RUTA_LINEAS)
mtime_pr   = obtener_mtime(RUTA_PRECIOS)
mtime_mun  = obtener_mtime(RUTA_MUNICIPIOS)
mtime_intl = obtener_mtime(RUTA_INTERNACIONALES)
mtime_var  = obtener_mtime(RUTA_VARIEDADES)

@st.cache_resource(show_spinner=False)
def get_con_var(mtime):
    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=4")
    if RUTA_VARIEDADES.exists():
        con.execute(f"""
            CREATE OR REPLACE VIEW variedades AS
            SELECT
                CAST(fecha_mes AS DATE)                        AS fecha_mes,
                MONTH(CAST(fecha_mes AS DATE))                 AS mes,
                CASE WHEN MONTH(CAST(fecha_mes AS DATE)) BETWEEN 1 AND 6
                     THEN 'Primer semestre' ELSE 'Segundo semestre' END AS semestre,
                STRFTIME(CAST(fecha_mes AS DATE),'%Y-%m')      AS etiqueta_mes,
                TRIM(CAST(rubro AS VARCHAR))                   AS rubro,
                TRIM(CAST(alimento AS VARCHAR))                AS alimento,
                TRIM(CAST(central_mayorista AS VARCHAR))       AS central_mayorista,
                UPPER(TRIM(CAST(departamento_origen AS VARCHAR))) AS departamento_origen,
                UPPER(TRIM(CAST(municipio_origen AS VARCHAR))) AS municipio_origen,
                TRIM(CAST(cod_municipio AS VARCHAR))           AS cod_municipio,
                CAST(lon_mun AS DOUBLE)                        AS lon_mun,
                CAST(lat_mun AS DOUBLE)                        AS lat_mun,
                CAST(lon_central AS DOUBLE)                    AS lon_central,
                CAST(lat_central AS DOUBLE)                    AS lat_central,
                CAST(toneladas AS DOUBLE)                      AS toneladas,
                CAST(precio AS DOUBLE)                         AS precio,
                CAST(tipo_precio AS VARCHAR)                   AS tipo_precio,
                CAST(unidad_precio AS VARCHAR)                 AS unidad_precio,
                CAST(nota_precio AS VARCHAR)                   AS nota_precio
            FROM read_parquet('{RUTA_VAR_SQL}')
        """)
    return con

@st.cache_data(show_spinner=False)
def consultar_var_catalogos(mtime_var):
    """Catálogos para la pestaña de variedades."""
    if not RUTA_VARIEDADES.exists():
        return {}, [], []
    con = get_con_var(mtime_var)
    # Rubros priorizados disponibles en el parquet
    rubros_var = con.execute(
        "SELECT DISTINCT rubro FROM variedades WHERE rubro IS NOT NULL ORDER BY rubro"
    ).df()["rubro"].tolist()
    # Alimentos por rubro
    alimentos_df = con.execute(
        "SELECT DISTINCT rubro, alimento FROM variedades WHERE alimento IS NOT NULL ORDER BY rubro, alimento"
    ).df()
    alimentos_por_rubro = alimentos_df.groupby("rubro")["alimento"].apply(list).to_dict()
    # Centrales
    centrales_var = con.execute(
        "SELECT DISTINCT central_mayorista FROM variedades WHERE central_mayorista IS NOT NULL ORDER BY central_mayorista"
    ).df()["central_mayorista"].tolist()
    rango_var = con.execute(
        "SELECT MIN(fecha_mes) AS fmin, MAX(fecha_mes) AS fmax FROM variedades"
    ).df()
    fecha_min_var = pd.to_datetime(rango_var.loc[0,"fmin"]).date()
    fecha_max_var = pd.to_datetime(rango_var.loc[0,"fmax"]).date()
    return alimentos_por_rubro, centrales_var, (fecha_min_var, fecha_max_var)

@st.cache_data(show_spinner=False)
def consultar_var(fecha_ini, fecha_fin, semestre, rubros_t, alimentos_t,
                  centrales_t, deptos_t, municipios_t, muns_prio_t, mtime_var):
    """Consulta principal de variedades con todos los filtros."""
    if not RUTA_VARIEDADES.exists():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    con = get_con_var(mtime_var)

    # WHERE para la variedad seleccionada
    c = ["fecha_mes BETWEEN ? AND ?"]
    p = [fecha_ini.isoformat(), fecha_fin.isoformat()]
    if semestre == "Primer semestre":    c.append("mes BETWEEN 1 AND 6")
    elif semestre == "Segundo semestre": c.append("mes BETWEEN 7 AND 12")
    if rubros_t:
        c.append(f"rubro IN ({','.join(['?']*len(rubros_t))})"); p.extend(list(rubros_t))
    if alimentos_t:
        c.append(f"alimento IN ({','.join(['?']*len(alimentos_t))})"); p.extend(list(alimentos_t))
    if centrales_t:
        c.append(f"central_mayorista IN ({','.join(['?']*len(centrales_t))})"); p.extend(list(centrales_t))
    if deptos_t:
        c.append(f"departamento_origen IN ({','.join(['?']*len(deptos_t))})"); p.extend(list(deptos_t))
    if municipios_t:
        c.append(f"municipio_origen IN ({','.join(['?']*len(municipios_t))})"); p.extend(list(municipios_t))
    if muns_prio_t:
        c.append(f"cod_municipio IN ({','.join(['?']*len(muns_prio_t))})"); p.extend(list(muns_prio_t))
    w = " AND ".join(c)

    # WHERE solo por rubro (para calcular total del rubro de referencia)
    c_rubro = ["fecha_mes BETWEEN ? AND ?"]
    p_rubro = [fecha_ini.isoformat(), fecha_fin.isoformat()]
    if semestre == "Primer semestre":    c_rubro.append("mes BETWEEN 1 AND 6")
    elif semestre == "Segundo semestre": c_rubro.append("mes BETWEEN 7 AND 12")
    if rubros_t:
        c_rubro.append(f"rubro IN ({','.join(['?']*len(rubros_t))})"); p_rubro.extend(list(rubros_t))
    if centrales_t:
        c_rubro.append(f"central_mayorista IN ({','.join(['?']*len(centrales_t))})"); p_rubro.extend(list(centrales_t))
    if deptos_t:
        c_rubro.append(f"departamento_origen IN ({','.join(['?']*len(deptos_t))})"); p_rubro.extend(list(deptos_t))
    if municipios_t:
        c_rubro.append(f"municipio_origen IN ({','.join(['?']*len(municipios_t))})"); p_rubro.extend(list(municipios_t))
    if muns_prio_t:
        c_rubro.append(f"cod_municipio IN ({','.join(['?']*len(muns_prio_t))})"); p_rubro.extend(list(muns_prio_t))
    w_rubro = " AND ".join(c_rubro)

    try:
        # Total del rubro completo (para representatividad)
        rubro_ref = con.execute(f"""
            SELECT rubro,
                   SUM(toneladas)                                           AS ton_rubro,
                   AVG(CASE WHEN precio IS NOT NULL THEN precio END)        AS precio_prom_rubro
            FROM variedades WHERE {w_rubro}
            GROUP BY rubro
        """, p_rubro).df()

        # Mapa: flujos municipio → central
        flujos = con.execute(f"""
            SELECT alimento, rubro, municipio_origen, departamento_origen, cod_municipio,
                   central_mayorista,
                   AVG(lon_mun) AS lon_orig, AVG(lat_mun) AS lat_orig,
                   AVG(lon_central) AS lon_dest, AVG(lat_central) AS lat_dest,
                   SUM(toneladas) AS toneladas_total,
                   AVG(CASE WHEN precio IS NOT NULL THEN precio END) AS precio_prom
            FROM variedades WHERE {w}
              AND lon_mun IS NOT NULL AND lon_central IS NOT NULL
            GROUP BY 1,2,3,4,5,6 ORDER BY toneladas_total DESC
        """, p).df()

        # Serie mensual por alimento
        serie = con.execute(f"""
            SELECT etiqueta_mes, alimento, rubro,
                   SUM(toneladas) AS toneladas_total,
                   AVG(CASE WHEN precio IS NOT NULL THEN precio END) AS precio_prom
            FROM variedades WHERE {w}
            GROUP BY 1,2,3 ORDER BY 1,2
        """, p).df()

        # Tabla de municipios
        tabla = con.execute(f"""
            SELECT municipio_origen, departamento_origen, alimento, rubro,
                   SUM(toneladas)    AS toneladas_total,
                   COUNT(DISTINCT etiqueta_mes) AS meses_activos,
                   AVG(CASE WHEN precio IS NOT NULL THEN precio END) AS precio_prom
            FROM variedades WHERE {w}
            GROUP BY 1,2,3,4 ORDER BY toneladas_total DESC
        """, p).df()

    except Exception:
        flujos = serie = tabla = rubro_ref = pd.DataFrame()
    return flujos, serie, tabla, rubro_ref

municipios = cargar_municipios(mtime_mun)
grupos, rubros, centrales, deptos, municipios_df, fecha_min_g, fecha_max_g = consultar_catalogos(mtime_ab, mtime_pr)
paises_lista = consultar_paises(mtime_intl)
alimentos_por_rubro, centrales_var, rango_var = consultar_var_catalogos(mtime_var)

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

tab1, tab2 = st.tabs(["Abastecimiento por rubro", "Desglose por variedad (Priorizados SARA)"])

with tab1:

    # Leer valores SARA desde session_state ANTES de renderizar filtros
    # Los widgets del expander SARA se definen después pero persisten via session_state
    solo_priorizados        = st.session_state.get("_cb_solo_prio", False)
    prio_oferta             = st.session_state.get("prio_oferta",   False)
    prio_demanda            = st.session_state.get("prio_demanda",  False)
    prio_region_central     = st.session_state.get("prio_rc",       False)
    prio_region_metropolitana = st.session_state.get("prio_rm",     False)
    territorio_sel          = [t for t in sorted(TERRITORIOS_FUNC.keys())
                               if st.session_state.get(f"terr_{t}", False)]

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
        rubros_sel_labels = st.multiselect("Rubro", options=rubros_f_labels,
                                           default=[], placeholder="Todos los rubros")
        rubros_sel = [codigo_rubro(l) for l in rubros_sel_labels]
    with f3:
        centrales_sel = st.multiselect("Central mayorista", options=centrales, default=[])
    with f4:
        semestre_sel = st.selectbox("Periodo", ["Todos","Primer semestre","Segundo semestre"], index=0)
    with f5:
        rango = st.date_input("Fechas", value=(fecha_min_g, fecha_max_g),
                              min_value=fecha_min_g, max_value=fecha_max_g)

    st.markdown('<div style="height:0.4rem;"></div>', unsafe_allow_html=True)

    # ── Fila 2: Depto | Municipio | País | Toggle | Slider ───
    g1, g2, g3, g4, g5 = st.columns([1.1, 1.2, 1.2, 0.9, 1.0])
    with g1:
        # Calcular conjunto de municipios SARA activo para restringir deptos
        _muns_sara = None
        if prio_oferta and prio_demanda:   _muns_sara = MUNS_AMBOS
        elif prio_oferta:                  _muns_sara = MUNS_OFERTA
        elif prio_demanda:                 _muns_sara = MUNS_DEMANDA
        if prio_region_metropolitana:
            _muns_sara = (_muns_sara & MUNS_REGION_METROPOLITANA) if _muns_sara else MUNS_REGION_METROPOLITANA
        if territorio_sel:
            _muns_terr_g1 = set()
            for t in territorio_sel: _muns_terr_g1.update(TERRITORIOS_FUNC.get(t, []))
            _muns_sara = (_muns_sara & _muns_terr_g1) if _muns_sara else _muns_terr_g1
        # Restringir deptos disponibles
        if _muns_sara and "cod_municipio" in municipios_df.columns:
            _deptos_sara = set(
                municipios_df[municipios_df["cod_municipio"].astype(str).isin(_muns_sara)]["departamento_origen"].str.upper().dropna().tolist()
            )
            deptos_opciones = [d for d in deptos if d.upper() in _deptos_sara or d == "INTERNACIONAL"]
        elif prio_region_central:
            deptos_opciones = [d for d in deptos if d.upper() in DEPTOS_REGION_CENTRAL or d == "INTERNACIONAL"]
        else:
            deptos_opciones = deptos
        deptos_sel = st.multiselect("Depto. origen", deptos_opciones, default=[])
    with g2:
        deptos_sin_intl = [d for d in deptos_sel if d != "INTERNACIONAL"]
        if deptos_sin_intl:
            muns_base = municipios_df[municipios_df["departamento_origen"].isin(deptos_sin_intl)]["municipio_origen"].unique().tolist()
        else:
            muns_base = municipios_df["municipio_origen"].unique().tolist()
        # Filtrar municipios por SARA
        _muns_prio_temp = None
        if prio_oferta and prio_demanda:   _muns_prio_temp = MUNS_AMBOS
        elif prio_oferta:                  _muns_prio_temp = MUNS_OFERTA
        elif prio_demanda:                 _muns_prio_temp = MUNS_DEMANDA
        if prio_region_metropolitana:
            _muns_prio_temp = (_muns_prio_temp & MUNS_REGION_METROPOLITANA) if _muns_prio_temp else MUNS_REGION_METROPOLITANA
        if territorio_sel:
            _muns_terr = set()
            for t in territorio_sel: _muns_terr.update(TERRITORIOS_FUNC.get(t, []))
            _muns_prio_temp = (_muns_prio_temp & _muns_terr) if _muns_prio_temp else _muns_terr
        if _muns_prio_temp and "cod_municipio" in municipios_df.columns:
            muns_prio_nombres = set(
                municipios_df[municipios_df["cod_municipio"].astype(str).isin(_muns_prio_temp)]["municipio_origen"].tolist()
            )
            muns_base = [m for m in muns_base if m in muns_prio_nombres] or muns_base
        muns_opciones = sorted(muns_base)
        municipios_sel = st.multiselect("Municipio origen", options=muns_opciones,
                                        default=[], placeholder="Todos los municipios")
    with g3:
        paises_sel = st.multiselect("País de origen", options=paises_lista,
                                    default=[], placeholder="Todos los países")
    with g4:
        st.markdown('<div style="font-size:0.82rem;color:#9EABC0;padding-top:1.6rem;">Flujos internacionales en mapa</div>',
                    unsafe_allow_html=True)
        mostrar_flujos_intl = st.toggle("Flujos internacionales en mapa", value=False,
                                         label_visibility="collapsed")
    with g5:
        st.markdown('<div style="font-size:0.82rem;color:#9EABC0;padding-top:1.6rem;">'
                    '🔀 Máx. flujos en mapa</div>', unsafe_allow_html=True)
        max_flujos = st.slider("Máx. flujos", min_value=100, max_value=2000,
                               value=MAX_LINEAS_MAPA, step=100, label_visibility="collapsed")

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Expander SARA — debajo de los filtros principales ────
    with st.expander("**FILTROS DEL PROYECTO SARA**", expanded=False):
        sc1, sc2, sc3 = st.columns([1.0, 1.2, 1.2])
        with sc1:
            solo_priorizados = st.checkbox(
                "Rubros priorizados SARA", value=False,
                key="_cb_solo_prio",
                help="Filtra el selector de Rubro a los 36 rubros priorizados SARA"
            )
            st.markdown('<div style="font-size:0.8rem;color:#9EABC0;margin-top:0.6rem;">Municipios priorizados</div>', unsafe_allow_html=True)
            prio_oferta  = st.checkbox("Oferta",  value=False, key="prio_oferta")
            prio_demanda = st.checkbox("Demanda", value=False, key="prio_demanda")
            st.markdown('<div style="font-size:0.8rem;color:#9EABC0;margin-top:0.6rem;">Alcance regional</div>', unsafe_allow_html=True)
            prio_region_central      = st.checkbox("Región Central",      value=False, key="prio_rc")
            prio_region_metropolitana = st.checkbox("Región Metropolitana", value=False, key="prio_rm")
        territorios_opciones = sorted(TERRITORIOS_FUNC.keys())
        mitad = (len(territorios_opciones) + 1) // 2
        territorio_checks = {}
        with sc2:
            st.markdown('<div style="font-size:0.8rem;color:#9EABC0;margin-bottom:0.3rem;">Territorio funcional</div>', unsafe_allow_html=True)
            for t in territorios_opciones[:mitad]:
                territorio_checks[t] = st.checkbox(t, value=False, key=f"terr_{t}")
        with sc3:
            st.markdown('<div style="font-size:0.8rem;color:#9EABC0;margin-bottom:0.3rem;">&nbsp;</div>', unsafe_allow_html=True)
            for t in territorios_opciones[mitad:]:
                territorio_checks[t] = st.checkbox(t, value=False, key=f"terr_{t}")
    territorio_sel = [t for t, v in territorio_checks.items() if v]

    # Aplicar filtro de priorizados al selector de rubros
    if solo_priorizados and not rubros_sel:
        rubros_sel = list(RUBROS_PRIORIZADOS_SARA)

    # Resolver municipios priorizados — SOLO oferta/demanda/metropolitana
    # El territorio funcional se aplica SEPARADO (en muns_prio_t para la consulta
    # y en muns_resalte para el mapa), no mezclado con oferta/demanda
    if prio_oferta and prio_demanda:
        muns_prio = MUNS_AMBOS
    elif prio_oferta:
        muns_prio = MUNS_OFERTA
    elif prio_demanda:
        muns_prio = MUNS_DEMANDA
    else:
        muns_prio = None

    if prio_region_metropolitana:
        muns_prio = (muns_prio & MUNS_REGION_METROPOLITANA) if muns_prio else set(MUNS_REGION_METROPOLITANA)

    # Territorio funcional — conjunto separado para consulta y mapa
    muns_territorio_consulta = set()
    if territorio_sel:
        for t in territorio_sel:
            muns_territorio_consulta.update(TERRITORIOS_FUNC.get(t, []))

    # muns_prio_t para la consulta SQL: intersección de oferta/demanda/metro CON territorio si aplica
    if muns_territorio_consulta and muns_prio:
        muns_prio_consulta = muns_prio & muns_territorio_consulta
    elif muns_territorio_consulta:
        muns_prio_consulta = muns_territorio_consulta
    elif muns_prio:
        muns_prio_consulta = muns_prio
    else:
        muns_prio_consulta = None

    if isinstance(rango, tuple) and len(rango) == 2:
        fecha_ini, fecha_fin = rango
    else:
        fecha_ini, fecha_fin = fecha_min_g, fecha_max_g

    centrales_t   = tuple(centrales_sel)
    deptos_t      = tuple([d for d in deptos_sel if d != "INTERNACIONAL"])
    # Región Central restringe deptos cuando no hay otro depto seleccionado
    if prio_region_central and not deptos_t:
        deptos_t = tuple(DEPTOS_REGION_CENTRAL)
    rubros_t      = tuple(rubros_sel)
    municipios_t  = tuple(municipios_sel) if municipios_sel else ()
    paises_t      = tuple(paises_sel) if paises_sel else ()
    muns_prio_t   = tuple(sorted(muns_prio_consulta)) if muns_prio_consulta else ()

    # ── Lógica de internacionales ─────────────────────────────
    # incluir_intl: mostrar datos internacionales
    incluir_intl = (
        "INTERNACIONAL" in deptos_sel
        or bool(paises_sel)
        or not deptos_sel
    )
    # solo_internacional: SOLO mostrar internacionales (sin datos nacionales)
    # Ocurre cuando el único filtro de origen es INTERNACIONAL o países
    solo_internacional = (
        bool(paises_sel) and not deptos_t and not municipios_t
    ) or (
        bool(deptos_sel)
        and all(d == "INTERNACIONAL" for d in deptos_sel)
        and not municipios_t
    )
    # excluir_nacional: hay filtro de depto/municipio nacional pero no INTERNACIONAL
    excluir_intl_por_depto = bool(deptos_t) and "INTERNACIONAL" not in deptos_sel and not paises_sel

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

    if solo_internacional:
        # Solo mostrar internacionales — vaciar todo lo nacional
        met_df             = _empty_met
        tot_df             = _empty_tot
        rape_df            = _empty_rap
        rank_df            = pd.DataFrame()
        serie_ab_df        = pd.DataFrame()
        flujos_df          = pd.DataFrame()
        sankey_df          = pd.DataFrame()
        serie_pr_df        = pd.DataFrame()
        precios_central_df = pd.DataFrame()
    else:
        met_df, tot_df, rape_df, rank_df, serie_ab_df, flujos_df, sankey_df = consultar_abast(
            fecha_ini, fecha_fin, semestre_sel, grupo_sel, rubros_t,
            centrales_t, deptos_t, muns_prio_t, municipios_t, mtime_ab
        )
        serie_pr_df, precios_central_df = consultar_precios(
            fecha_ini, fecha_fin, semestre_sel, grupo_sel, rubros_t,
            centrales_t, mtime_pr
        )

    # ── Datos internacionales ─────────────────────────────────
    @st.cache_data(show_spinner=False)
    def consultar_internacionales(fecha_ini, fecha_fin, semestre, grupo, rubros,
                                   centrales_t, paises_t, mtime_intl):
        if not RUTA_INTERNACIONALES.exists():
            return pd.DataFrame()
        con = get_con_intl(mtime_intl)
        c = ["fecha_mes BETWEEN ? AND ?"]
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

    if incluir_intl and not excluir_intl_por_depto:
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
    vol_intl         = intl_df["toneladas_total"].sum() if not intl_df.empty else 0.0
    vol_filtro_total = vol_filtro + vol_intl

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
            top_mun = sankey_df.groupby("municipio_origen")["toneladas_total"].sum().nlargest(10).index.tolist()
            sk_top  = sankey_df[sankey_df["municipio_origen"].isin(top_mun)].copy()
        else:
            sk_top = pd.DataFrame(columns=["municipio_origen","central_mayorista","toneladas_total"])

        # Agregar internacionales al Sankey
        if not intl_df.empty:
            intl_sk = intl_df.groupby(["pais_origen","central_mayorista"], as_index=False).agg(
                toneladas_total=("toneladas_total","sum")
            ).rename(columns={"pais_origen":"municipio_origen"})
            intl_sk["municipio_origen"] = "* " + intl_sk["municipio_origen"]
            top_paises = intl_df.groupby("pais_origen")["toneladas_total"].sum().nlargest(2).index.tolist()
            intl_sk = intl_sk[intl_sk["municipio_origen"].isin(["* " + p for p in top_paises])]
            sk_top = pd.concat([sk_top, intl_sk], ignore_index=True)

        # Agregar internacionales al ranking (para mostrar en tabla)
        if not intl_df.empty:
            intl_rk_agg = intl_df.groupby("pais_origen", as_index=False).agg(
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

    # =========================================================
    # MAPA — polígonos con colores por estado
    # deptos_sel resalta municipios del departamento en amarillo
    # top30 en morado, resto gris
    # =========================================================

    deptos_sel_upper = {d.upper() for d in deptos_sel if d != "INTERNACIONAL"}

    # Municipios a resaltar en el mapa (separado de la lógica de consulta):
    # Prioridad: territorio > región metropolitana > oferta/demanda
    # El territorio muestra sus polígonos completos (sin intersección)
    if muns_territorio_consulta:
        muns_resalte = muns_territorio_consulta
    elif prio_region_metropolitana:
        muns_resalte = MUNS_REGION_METROPOLITANA
    elif muns_prio:
        muns_resalte = muns_prio
    else:
        muns_resalte = set()

    # Región Central: resaltar todos los municipios de los deptos RAPE
    muns_region_central_set = set()
    if prio_region_central:
        if "cod_municipio" in municipios_df.columns:
            muns_region_central_set = set(
                municipios_df[
                    municipios_df["departamento_origen"].str.upper().isin(DEPTOS_REGION_CENTRAL)
                ]["cod_municipio"].astype(str).tolist()
            )

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
    mun_web["tipo_elemento"] = "Municipio"
    mun_web["detalle_1"] = "Nombre: "       + mun_web["nombre_municipio"].fillna("Sin nombre").astype(str)
    mun_web["detalle_2"] = "Departamento: " + mun_web["departamento"].fillna("Sin dato").astype(str)
    mun_web["detalle_3"] = "Código: "       + mun_web["codigo_origen"].fillna("").astype(str)
    mun_web["detalle_4"] = ""

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
            # Centrales activas: si solo internacional, contar centrales de destino intl
            cent_act_display = cent_act
            if cent_act == 0 and not intl_df.empty:
                cent_act_display = intl_df["central_mayorista"].nunique()
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
            if any(d != "INTERNACIONAL" for d in deptos_sel):
                leyenda_depto = '<div class="legend-item"><span class="legend-box" style="background:#B0B8C8;border:1px solid #888;"></span>Municipios depto. filtrado</div>'
            leyenda_intl = ""
            if mostrar_flujos_intl and not intl_df.empty:
                leyenda_intl = '<div class="legend-item"><span class="legend-box" style="background:#00C878;"></span>Flujos internacionales</div>'
            leyenda_territorio = ""
            if muns_resalte:
                if territorio_sel and muns_prio and not territorio_sel == list(muns_prio):
                    label_t = "Municipios SARA (intersección)"
                elif territorio_sel:
                    label_t = "Municipios territorio SARA"
                else:
                    label_t = "Municipios priorizados SARA"
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
                Datos internacionales siempre en tabla e indicadores.
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
                ).agg(lon=("lon_orig","first"), lat=("lat_orig","first"),
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


# =========================================================
# PESTAÑA 2 — DESGLOSE POR VARIEDAD (PRIORIZADOS SARA)
# =========================================================

with tab2:

    # ── Filtros fila 1 ────────────────────────────────────
    st.markdown('<div class="filter-wrap">', unsafe_allow_html=True)
    vf1, vf2, vf3, vf4, vf5 = st.columns([1.3, 1.6, 1.5, 1.0, 1.4])
    with vf1:
        rubros_var_labels = sorted(
            [label_rubro(r) for r in alimentos_por_rubro.keys()]
        )
        rubros_var_sel_labels = st.multiselect(
            "Rubro(s) priorizado(s)", options=rubros_var_labels,
            default=[], placeholder="Selecciona uno o más rubros", key="var_rubros"
        )
        rubros_var_sel = [codigo_rubro(l) for l in rubros_var_sel_labels]
    with vf2:
        if rubros_var_sel:
            alimentos_disp = sorted(set(
                a for r in rubros_var_sel for a in alimentos_por_rubro.get(r, [])
            ))
        else:
            alimentos_disp = sorted(set(
                a for alims in alimentos_por_rubro.values() for a in alims
            ))
        alimentos_sel = st.multiselect(
            "Alimento(s)", options=alimentos_disp,
            default=[], placeholder="Todos los alimentos del rubro", key="var_alimentos"
        )
    with vf3:
        centrales_var_sel = st.multiselect(
            "Central mayorista", options=centrales_var,
            default=[], placeholder="Todas las centrales", key="var_centrales"
        )
    with vf4:
        semestre_var = st.selectbox(
            "Periodo", ["Todos","Primer semestre","Segundo semestre"],
            index=0, key="var_semestre"
        )
    with vf5:
        fecha_min_v, fecha_max_v = rango_var
        rango_var_sel = st.date_input(
            "Fechas", value=(fecha_min_v, fecha_max_v),
            min_value=fecha_min_v, max_value=fecha_max_v, key="var_fechas"
        )
    st.markdown('<div style="height:0.4rem;"></div>', unsafe_allow_html=True)

    # ── Filtros fila 2: Depto / Municipio / Slider ────────
    vg1, vg2, vg3 = st.columns([1.4, 1.6, 1.4])
    with vg1:
        deptos_var_sel = st.multiselect(
            "Depto. origen", deptos, default=[], key="var_deptos"
        )
    with vg2:
        deptos_var_sin_intl = [d for d in deptos_var_sel if d != "INTERNACIONAL"]
        if deptos_var_sin_intl:
            muns_v_base = municipios_df[
                municipios_df["departamento_origen"].isin(deptos_var_sin_intl)
            ]["municipio_origen"].unique().tolist()
        else:
            muns_v_base = municipios_df["municipio_origen"].unique().tolist()
        municipios_var_sel = st.multiselect(
            "Municipio origen", options=sorted(muns_v_base),
            default=[], placeholder="Todos los municipios", key="var_municipios"
        )
    with vg3:
        st.markdown('<div style="font-size:0.82rem;color:#9EABC0;padding-top:1.6rem;">'
                    '🔀 Máx. flujos en mapa</div>', unsafe_allow_html=True)
        max_flujos_v = st.slider(
            "Máx. flujos var", min_value=100, max_value=2000,
            value=600, step=100, label_visibility="collapsed", key="var_flujos"
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Expander SARA propio ──────────────────────────────
    with st.expander("**FILTROS DEL PROYECTO SARA**", expanded=False):
        vsc1, vsc2, vsc3 = st.columns([1.0, 1.2, 1.2])
        with vsc1:
            v_solo_prio   = st.checkbox("Rubros priorizados SARA", value=True,
                                        key="var_cb_prio",
                                        help="Activo por defecto en esta pestaña")
            st.markdown('<div style="font-size:0.8rem;color:#9EABC0;margin-top:0.6rem;">Municipios priorizados</div>',
                        unsafe_allow_html=True)
            v_prio_oferta  = st.checkbox("Oferta",  value=False, key="var_prio_oferta")
            v_prio_demanda = st.checkbox("Demanda", value=False, key="var_prio_demanda")
            st.markdown('<div style="font-size:0.8rem;color:#9EABC0;margin-top:0.6rem;">Alcance regional</div>',
                        unsafe_allow_html=True)
            v_prio_rc = st.checkbox("Región Central",      value=False, key="var_prio_rc")
            v_prio_rm = st.checkbox("Región Metropolitana", value=False, key="var_prio_rm")
        territorios_v_opts = sorted(TERRITORIOS_FUNC.keys())
        mitad_v = (len(territorios_v_opts) + 1) // 2
        terr_v_checks = {}
        with vsc2:
            st.markdown('<div style="font-size:0.8rem;color:#9EABC0;margin-bottom:0.3rem;">Territorio funcional</div>',
                        unsafe_allow_html=True)
            for t in territorios_v_opts[:mitad_v]:
                terr_v_checks[t] = st.checkbox(t, value=False, key=f"var_terr_{t}")
        with vsc3:
            st.markdown('<div style="font-size:0.8rem;color:#9EABC0;margin-bottom:0.3rem;">&nbsp;</div>',
                        unsafe_allow_html=True)
            for t in territorios_v_opts[mitad_v:]:
                terr_v_checks[t] = st.checkbox(t, value=False, key=f"var_terr_{t}")
    territorio_var_sel = [t for t, v in terr_v_checks.items() if v]

    # ── Resolver filtros SARA ─────────────────────────────
    if v_prio_oferta and v_prio_demanda: muns_var_prio = MUNS_AMBOS
    elif v_prio_oferta:                  muns_var_prio = MUNS_OFERTA
    elif v_prio_demanda:                 muns_var_prio = MUNS_DEMANDA
    else:                                muns_var_prio = None

    if v_prio_rm:
        muns_var_prio = (muns_var_prio & MUNS_REGION_METROPOLITANA) if muns_var_prio else set(MUNS_REGION_METROPOLITANA)

    muns_var_terr = set()
    for t in territorio_var_sel:
        muns_var_terr.update(TERRITORIOS_FUNC.get(t, []))
    if muns_var_terr and muns_var_prio:
        muns_var_prio_consulta = muns_var_prio & muns_var_terr
    elif muns_var_terr:
        muns_var_prio_consulta = muns_var_terr
    elif muns_var_prio:
        muns_var_prio_consulta = muns_var_prio
    else:
        muns_var_prio_consulta = None

    deptos_var_t = tuple([d for d in deptos_var_sel if d != "INTERNACIONAL"])
    if v_prio_rc and not deptos_var_t:
        deptos_var_t = tuple(DEPTOS_REGION_CENTRAL)

    rubros_var_t    = tuple(rubros_var_sel)
    alimentos_var_t = tuple(alimentos_sel)
    centrales_var_t = tuple(centrales_var_sel)
    municipios_var_t = tuple(municipios_var_sel) if municipios_var_sel else ()
    muns_var_prio_t  = tuple(sorted(muns_var_prio_consulta)) if muns_var_prio_consulta else ()

    if isinstance(rango_var_sel, tuple) and len(rango_var_sel) == 2:
        fecha_ini_v, fecha_fin_v = rango_var_sel
    else:
        fecha_ini_v, fecha_fin_v = fecha_min_v, fecha_max_v

    if not rubros_var_sel and not alimentos_sel:
        st.info("Selecciona al menos un rubro o alimento para visualizar el desglose por variedad.")
    else:
        with st.spinner("Consultando variedades..."):
            flujos_v, serie_v, tabla_v, rubro_ref_v = consultar_var(
                fecha_ini_v, fecha_fin_v, semestre_var,
                rubros_var_t, alimentos_var_t, centrales_var_t,
                deptos_var_t, municipios_var_t, muns_var_prio_t, mtime_var
            )

        if flujos_v.empty:
            st.warning("No hay datos para los filtros seleccionados.")
        else:
            # ── Representatividad por alimento ────────────────
            ton_variedad = flujos_v.groupby(["alimento","rubro"])["toneladas_total"].sum().reset_index()
            precio_variedad = tabla_v.groupby("alimento")["precio_prom"].mean().reset_index()

            if not rubro_ref_v.empty:
                ton_variedad = ton_variedad.merge(rubro_ref_v, on="rubro", how="left")
                ton_variedad["repr_pct"] = (
                    ton_variedad["toneladas_total"] / ton_variedad["ton_rubro"] * 100
                ).round(1)
                ton_variedad = ton_variedad.merge(precio_variedad, on="alimento", how="left")
                ton_variedad["vs_rubro"] = ton_variedad["precio_prom"] - ton_variedad["precio_prom_rubro"]

            # ── Layout 3 cols: indicadores | mapa | serie ─────
            col_ind, col_map, col_ser = st.columns([1.0, 2.5, 1.8], gap="small")

            with col_ind:
                st.markdown('<div class="panel">', unsafe_allow_html=True)
                st.markdown('<div class="panel-title">Indicadores</div>', unsafe_allow_html=True)
                tot_ton_v = flujos_v["toneladas_total"].sum()
                n_mun_v   = flujos_v["municipio_origen"].nunique()
                n_cent_v  = flujos_v["central_mayorista"].nunique()

                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Toneladas acumuladas</div>
                    <div class="metric-value" style="font-size:1.4rem;">{tot_ton_v:,.0f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Municipios origen</div>
                    <div class="metric-value" style="font-size:1.4rem;">{n_mun_v:,}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Centrales activas</div>
                    <div class="metric-value" style="font-size:1.4rem;">{n_cent_v}</div>
                </div>
                """, unsafe_allow_html=True)

                # Representatividad por alimento
                st.markdown('<div class="panel-title" style="margin-top:0.7rem;">Representatividad sobre el rubro</div>',
                            unsafe_allow_html=True)
                if not rubro_ref_v.empty and "repr_pct" in ton_variedad.columns:
                    for _, row in ton_variedad.iterrows():
                        repr_pct = row.get("repr_pct", None)
                        vs       = row.get("vs_rubro", None)
                        precio_v = row.get("precio_prom", None)
                        color_vs = "#00C878" if pd.notna(vs) and vs < 0 else ("#FF6B6B" if pd.notna(vs) and vs > 0 else "#9EABC0")
                        texto_vs = ""
                        if pd.notna(vs) and pd.notna(precio_v):
                            signo = "↓ más bajo" if vs < 0 else "↑ más alto"
                            texto_vs = f'<div style="font-size:0.75rem;color:{color_vs};">{signo} que promedio del rubro (${abs(vs):,.0f}/kg)</div>'
                        repr_txt = f"{repr_pct:.1f}% del rubro" if pd.notna(repr_pct) else "—"
                        st.markdown(f"""
                        <div style="background:#1E2530;border-radius:6px;padding:0.5rem 0.6rem;
                            margin-bottom:0.4rem;border-left:3px solid #4DA3FF;">
                            <div style="font-size:0.8rem;font-weight:600;color:#C8D8F0;">{row['alimento']}</div>
                            <div style="font-size:0.85rem;color:#F5F7FA;">{repr_txt}</div>
                            {texto_vs}
                        </div>
                        """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col_map:
                st.markdown('<div class="panel-title">Mapa de flujos por alimento</div>',
                            unsafe_allow_html=True)

                ALIM_COLORS = [
                    [245,176,65],[0,200,180],[255,99,132],[100,160,255],
                    [255,206,86],[75,192,192],[200,100,255],[255,159,64],
                    [180,220,120],[0,200,120],
                ]
                alimentos_unicos = sorted(flujos_v["alimento"].unique())
                color_map_v = {a: ALIM_COLORS[i % len(ALIM_COLORS)]
                               for i, a in enumerate(alimentos_unicos)}

                flujos_top = flujos_v.nlargest(max_flujos_v, "toneladas_total").copy()
                vmin = flujos_top["toneladas_total"].min()
                vmax = flujos_top["toneladas_total"].max()
                flujos_top["ancho"] = 1 + 7 * (
                    (flujos_top["toneladas_total"] - vmin) / (vmax - vmin + 1e-9)
                )
                flujos_top["color"] = flujos_top["alimento"].map(
                    lambda a: color_map_v.get(a, [200,200,200])
                )
                precio_fmt = flujos_top["precio_prom"].map(
                    lambda x: f"$ {x:,.0f}/kg" if pd.notna(x) else "Sin dato precio"
                )
                flujos_top["tipo_elemento"] = "Flujo variedad"
                flujos_top["detalle_1"] = "Alimento: "  + flujos_top["alimento"].fillna("")
                flujos_top["detalle_2"] = "Municipio: " + flujos_top["municipio_origen"].fillna("")
                flujos_top["detalle_3"] = "Toneladas: " + flujos_top["toneladas_total"].map(formatear_ton)
                flujos_top["detalle_4"] = "Precio: "    + precio_fmt

                layers_v = [pdk.Layer(
                    "ArcLayer", data=flujos_top,
                    get_source_position=["lon_orig","lat_orig"],
                    get_target_position=["lon_dest","lat_dest"],
                    get_source_color="color", get_target_color="color",
                    get_width="ancho", width_scale=1, width_min_pixels=1,
                    pickable=True, auto_highlight=True
                )]

                orig_v = flujos_top.groupby(
                    ["municipio_origen","departamento_origen","cod_municipio"], as_index=False
                ).agg(lon=("lon_orig","first"), lat=("lat_orig","first"),
                      toneladas_total=("toneladas_total","sum"),
                      precio_prom=("precio_prom","mean"))
                orig_v = orig_v.dropna(subset=["lon","lat"])
                orig_v["tipo_elemento"] = "Municipio de origen"
                orig_v["detalle_1"] = "Municipio: "   + orig_v["municipio_origen"].fillna("")
                orig_v["detalle_2"] = "Departamento: "+ orig_v["departamento_origen"].fillna("")
                orig_v["detalle_3"] = "Toneladas: "   + orig_v["toneladas_total"].map(formatear_ton)
                orig_v["detalle_4"] = "Precio prom.: "+ orig_v["precio_prom"].map(
                    lambda x: f"$ {x:,.0f}/kg" if pd.notna(x) else "Sin dato")
                layers_v.append(pdk.Layer(
                    "ScatterplotLayer", data=orig_v,
                    get_position="[lon, lat]", get_radius=4200,
                    get_fill_color=[245,160,32,180], get_line_color=[255,210,100,220],
                    line_width_min_pixels=1, pickable=True, auto_highlight=True
                ))

                cent_v = flujos_top.groupby("central_mayorista", as_index=False).agg(
                    lon=("lon_dest","first"), lat=("lat_dest","first"))
                cent_v = cent_v.dropna(subset=["lon","lat"])
                cent_v["tipo_elemento"] = "Central mayorista"
                cent_v["detalle_1"] = "Central: " + cent_v["central_mayorista"].fillna("")
                cent_v["detalle_2"] = cent_v["detalle_3"] = cent_v["detalle_4"] = ""
                layers_v.append(pdk.Layer(
                    "ScatterplotLayer", data=cent_v,
                    get_position="[lon, lat]", get_radius=13500,
                    get_fill_color=[0,210,255,190], get_line_color=[170,245,255,255],
                    line_width_min_pixels=2, pickable=True
                ))

                deck_v = pdk.Deck(
                    layers=layers_v,
                    initial_view_state=pdk.ViewState(
                        latitude=4.5, longitude=-74.1, zoom=4.6, pitch=0),
                    tooltip={
                        "html": "<b>{tipo_elemento}</b><br/>{detalle_1}<br/>{detalle_2}<br/>{detalle_3}<br/>{detalle_4}",
                        "style": {"backgroundColor":"rgba(18,22,29,0.95)",
                                  "color":"#F5F7FA","fontSize":"12px"}
                    },
                    map_style="dark",
                )
                st.pydeck_chart(deck_v, use_container_width=True)

                # Leyenda colores
                leyenda_v = "".join([
                    f'<span style="display:inline-flex;align-items:center;gap:4px;'
                    f'margin-right:8px;font-size:0.78rem;color:#C8D8F0;">'
                    f'<span style="width:12px;height:12px;border-radius:2px;'
                    f'background:rgb({c[0]},{c[1]},{c[2]});display:inline-block;"></span>{a}</span>'
                    for a, c in list(color_map_v.items())[:10]
                ])
                st.markdown(f'<div style="margin-top:0.4rem;flex-wrap:wrap;">{leyenda_v}</div>',
                            unsafe_allow_html=True)

            with col_ser:
                st.markdown('<div class="panel-title">Serie mensual</div>',
                            unsafe_allow_html=True)
                if not serie_v.empty:
                    fig_v = go.Figure()
                    for alim in alimentos_unicos:
                        s = serie_v[serie_v["alimento"] == alim]
                        c = color_map_v.get(alim, [200,200,200])
                        cstr = f"rgb({c[0]},{c[1]},{c[2]})"
                        fig_v.add_trace(go.Bar(
                            x=s["etiqueta_mes"], y=s["toneladas_total"],
                            name=f"{alim}", marker_color=cstr,
                            opacity=0.75, yaxis="y1",
                            hovertemplate=f"<b>{alim}</b><br>%{{x}}<br>%{{y:,.0f}} ton<extra></extra>"
                        ))
                        s_pr = s[s["precio_prom"].notna()]
                        if not s_pr.empty:
                            fig_v.add_trace(go.Scatter(
                                x=s_pr["etiqueta_mes"], y=s_pr["precio_prom"],
                                name=f"{alim} $/kg", mode="lines+markers",
                                line=dict(color=cstr, width=2, dash="dot"),
                                marker=dict(size=5), yaxis="y2",
                                hovertemplate=f"<b>{alim}</b><br>%{{x}}<br>${{y:,.0f}}/kg<extra></extra>"
                            ))
                    fig_v.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="#171A21", plot_bgcolor="#171A21",
                        margin=dict(l=10, r=10, t=10, b=5), height=500,
                        legend=dict(orientation="h", y=-0.22, x=0, font=dict(size=9)),
                        barmode="stack",
                        xaxis=dict(showgrid=False, tickangle=-45),
                        yaxis=dict(title="Toneladas", gridcolor="#2B3240"),
                        yaxis2=dict(title="Precio ($/kg)", overlaying="y",
                                    side="right", showgrid=False,
                                    tickprefix="$", tickformat=",.0f")
                    )
                    st.plotly_chart(fig_v, use_container_width=True)

            # ── Tabla de municipios ───────────────────────────
            st.markdown("---")
            st.markdown('<div class="panel-title">Ranking de municipios abastecedores por alimento</div>',
                        unsafe_allow_html=True)

            if not tabla_v.empty:
                # Agregar representatividad a la tabla
                if not rubro_ref_v.empty and "ton_rubro" in ton_variedad.columns:
                    repr_map = ton_variedad.set_index("alimento")[["repr_pct","vs_rubro"]].to_dict("index")
                else:
                    repr_map = {}

                tc1, tc2 = st.columns([2, 1])
                with tc1:
                    ord_col = st.selectbox("Ordenar por",
                        ["Toneladas","Precio prom. ($/kg)","Meses activos"],
                        index=0, key="var_ord_col", label_visibility="collapsed")
                with tc2:
                    ord_dir = st.radio("Dir", ["↓ Mayor","↑ Menor"], index=0,
                        horizontal=True, key="var_ord_dir", label_visibility="collapsed")

                tabla_show = tabla_v.copy()
                sort_map = {"Toneladas":"toneladas_total",
                            "Precio prom. ($/kg)":"precio_prom",
                            "Meses activos":"meses_activos"}
                tabla_show = tabla_show.sort_values(
                    sort_map[ord_col], ascending=(ord_dir=="↑ Menor")
                ).reset_index(drop=True)
                tabla_show.insert(0, "Ranking", tabla_show.index + 1)
                tabla_show["Toneladas"] = tabla_show["toneladas_total"].map("{:,.1f}".format)
                tabla_show["Precio prom. ($/kg)"] = tabla_show["precio_prom"].map(
                    lambda x: f"$ {x:,.0f}" if pd.notna(x) else "Sin dato"
                )
                tabla_show["Meses activos"] = tabla_show["meses_activos"].astype(int)
                tabla_show["% sobre rubro"] = tabla_show["alimento"].map(
                    lambda a: f"{repr_map[a]['repr_pct']:.1f}%" if a in repr_map and pd.notna(repr_map[a].get("repr_pct")) else "—"
                )
                tabla_show["vs. promedio rubro"] = tabla_show["alimento"].map(
                    lambda a: (f"↓ -${abs(repr_map[a]['vs_rubro']):,.0f}" if pd.notna(repr_map[a].get("vs_rubro")) and repr_map[a]['vs_rubro'] < 0
                               else f"↑ +${abs(repr_map[a]['vs_rubro']):,.0f}" if pd.notna(repr_map[a].get("vs_rubro")) and repr_map[a]['vs_rubro'] > 0
                               else "—") if a in repr_map else "—"
                )
                cols_tabla = ["Ranking","municipio_origen","departamento_origen","alimento",
                              "Toneladas","% sobre rubro","Precio prom. ($/kg)",
                              "vs. promedio rubro","Meses activos"]
                tabla_final = tabla_show[cols_tabla].rename(columns={
                    "municipio_origen":"Municipio",
                    "departamento_origen":"Departamento",
                    "alimento":"Alimento",
                })
                st.dataframe(tabla_final.head(300), use_container_width=True,
                             hide_index=True, height=380)

            st.markdown("""
            <div class="method-note">
                <b>% sobre rubro:</b> participación en toneladas del alimento sobre el total
                del rubro bajo los mismos filtros. <b>vs. promedio rubro:</b> diferencia entre
                el precio promedio de la variedad y el precio promedio de todos los alimentos
                del rubro (negativo = más económico que el promedio del rubro).
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div style="margin-top:0.8rem;padding-top:0.6rem;border-top:1px solid #2B3240;
                color:#8FA0B7;font-size:0.8rem;text-align:center;">
                Fuente: SIPSA · DANE · 2020–2026 | Solo rubros priorizados SARA
            </div>
            """, unsafe_allow_html=True)
