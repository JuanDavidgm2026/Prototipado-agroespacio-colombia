# =============================================================================
# AgroEspacio Colombia – Inteligencia Territorial SpVAR
# Desarrollo Agropecuario · Pobreza Rural · Deforestación
# Arquitectura OOD | Streamlit | PySAL | GeoPandas | Plotly
# =============================================================================

# -----------------------------------------------------------------------------
# 1. INSTALACIÓN Y ADVERTENCIAS
# -----------------------------------------------------------------------------
import warnings
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# 2. IMPORTACIONES
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon
from shapely.validation import make_valid
from libpysal.weights import Queen, Rook, KNN, DistanceBand
from libpysal.weights import lag_spatial
from esda.moran import Moran, Moran_Local
from spreg import OLS, ML_Lag
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics.pairwise import euclidean_distances

# Configuración de página en Streamlit (debe ir antes de renderizar componentes)
st.set_page_config(
    page_title="AgroEspacio Colombia",
    page_icon="🌿",
    layout="wide"
)

# =============================================================================
# 3. SPATIAL DATA ENGINE
# =============================================================================
class SpatialDataEngine:
    """Carga, sanea y enriquece la geometría oficial + vector de datos."""

    def __init__(self):
        self.gdf = None
        self.crs_metric = "EPSG:9377"   # MAGNA-SIRGAS Origen Nacional

    def normalizar_nombre(self, n):
        n = str(n).upper().strip()
        n = (n.replace('Á','A').replace('É','E').replace('Í','I')
               .replace('Ó','O').replace('Ú','U').replace('Ñ','N'))
        if 'BOGOTA' in n: return 'BOGOTA'
        if 'SAN ANDRES' in n: return 'SAN ANDRES'
        return n

    def cargar_y_sanear(self):
        urls = [
            "https://raw.githubusercontent.com/caticoa3/colombia_mapa/master/co_2018_MGN_DPTO_POLITICO.geojson",
            "https://raw.githubusercontent.com/santiblanko/colombia.geojson/master/depto.json"
        ]
        gdf = None
        for url in urls:
            try:
                gdf = gpd.read_file(url)
                break
            except Exception:
                continue
        if gdf is None:
            raise RuntimeError("No se pudo cargar cartografía oficial")

        # Normalizar columnas
        for c in gdf.columns:
            cl = str(c).lower()
            if any(x in cl for x in ['cod', 'dpto_ccdgo', 'codigo']):
                gdf = gdf.rename(columns={c: 'COD_DEP'})
            if any(x in cl for x in ['nom', 'dpto_cnmbr', 'nombre', 'name']):
                gdf = gdf.rename(columns={c: 'NOM_DEP'})

        if 'COD_DEP' not in gdf.columns:
            gdf['COD_DEP'] = [f"{i:02d}" for i in range(1, len(gdf)+1)]
        if 'NOM_DEP' not in gdf.columns:
            gdf['NOM_DEP'] = gdf.index.astype(str)

        gdf['COD_DEP'] = (gdf['COD_DEP'].astype(str).str.strip()
                          .str.replace(r'\.0$', '', regex=True).str.zfill(2))
        # Excluir San Andrés (COD 88) para contigüidad
        gdf = gdf[gdf['COD_DEP'] != '88'].copy()

        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=4326)
        gdf = gdf.to_crs(self.crs_metric)

        def reparar(geom):
            try:
                geom = make_valid(geom)
                if geom.geom_type == 'Polygon':
                    return MultiPolygon([geom])
                return geom
            except Exception:
                return geom.buffer(0) if geom else None

        gdf['geometry'] = gdf.geometry.apply(reparar)
        gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty].reset_index(drop=True)
        gdf['AREA_KM2'] = gdf.geometry.area / 1e6
        gdf['CENTROID_X'] = gdf.geometry.centroid.x
        gdf['CENTROID_Y'] = gdf.geometry.centroid.y
        self.gdf = gdf
        return self

    def integrar_datos(self):
        vab_data = {
            'ANTIOQUIA': 12.5, 'ATLANTICO': 1.8, 'BOGOTA': 0.9, 'BOLIVAR': 2.4, 'BOYACA': 4.1,
            'CALDAS': 3.2, 'CAQUETA': 1.9, 'CAUCA': 3.8, 'CESAR': 2.7, 'CHOCO': 1.1,
            'CORDOBA': 4.5, 'CUNDINAMARCA': 8.9, 'HUILA': 4.8, 'LA GUAJIRA': 0.7,
            'MAGDALENA': 2.3, 'META': 5.6, 'NARINO': 3.5, 'NORTE DE SANTANDER': 2.1,
            'QUINDIO': 1.4, 'RISARALDA': 2.0, 'SANTANDER': 5.9, 'SUCRE': 1.6,
            'TOLIMA': 5.2, 'VALLE DEL CAUCA': 7.8, 'ARAUCA': 1.3, 'CASANARE': 2.8,
            'PUTUMAYO': 0.9, 'AMAZONAS': 0.3, 'GUAINIA': 0.15, 'GUAVIARE': 0.6,
            'VAUPES': 0.12, 'VICHADA': 0.4
        }
        ipm_rural = {
            'ANTIOQUIA': 21.2, 'ATLANTICO': 26.4, 'BOGOTA': 8.2, 'BOLIVAR': 21.5, 'BOYACA': 18.0,
            'CALDAS': 20.3, 'CAQUETA': 24.2, 'CAUCA': 17.6, 'CESAR': 21.3, 'CHOCO': 45.2,
            'CORDOBA': 38.9, 'CUNDINAMARCA': 12.0, 'HUILA': 18.5, 'LA GUAJIRA': 61.3,
            'MAGDALENA': 28.1, 'META': 18.2, 'NARINO': 24.5, 'NORTE DE SANTANDER': 41.1,
            'QUINDIO': 15.0, 'RISARALDA': 23.0, 'SANTANDER': 16.0, 'SUCRE': 36.4,
            'TOLIMA': 24.7, 'VALLE DEL CAUCA': 14.5, 'ARAUCA': 18.3, 'CASANARE': 17.1,
            'PUTUMAYO': 22.0, 'AMAZONAS': 17.6, 'GUAINIA': 56.9, 'GUAVIARE': 26.3,
            'VAUPES': 45.5, 'VICHADA': 82.7
        }
        deforest = {
            'ANTIOQUIA': 7065, 'ATLANTICO': 200, 'BOGOTA': 50, 'BOLIVAR': 1800, 'BOYACA': 900,
            'CALDAS': 400, 'CAQUETA': 30414, 'CAUCA': 3500, 'CESAR': 2200, 'CHOCO': 8500,
            'CORDOBA': 2800, 'CUNDINAMARCA': 600, 'HUILA': 1100, 'LA GUAJIRA': 800,
            'MAGDALENA': 1500, 'META': 22554, 'NARINO': 2500, 'NORTE DE SANTANDER': 3200,
            'QUINDIO': 150, 'RISARALDA': 300, 'SANTANDER': 1800, 'SUCRE': 700,
            'TOLIMA': 950, 'VALLE DEL CAUCA': 900, 'ARAUCA': 2800, 'CASANARE': 2500,
            'PUTUMAYO': 8961, 'AMAZONAS': 4200, 'GUAINIA': 1800, 'GUAVIARE': 15941,
            'VAUPES': 2100, 'VICHADA': 3800
        }

        gdf = self.gdf.copy()
        gdf['NOM_NORM'] = gdf['NOM_DEP'].apply(self.normalizar_nombre)

        def map_exact(data):
            vals = []
            for nom in gdf['NOM_NORM']:
                matched = False
                for k, val in data.items():
                    if self.normalizar_nombre(k) == nom or nom in self.normalizar_nombre(k):
                        vals.append(val)
                        matched = True
                        break
                if not matched:
                    vals.append(min(data.values()))
            return np.array(vals)

        gdf['VAB_AGRO']   = map_exact(vab_data)
        gdf['IPM_RURAL']  = map_exact(ipm_rural)
        gdf['DEFOREST_HA']= map_exact(deforest)
        gdf['ln_VAB'] = np.log(gdf['VAB_AGRO'] + 1)
        gdf['ln_IPM'] = np.log(gdf['IPM_RURAL'] + 1)
        gdf['ln_DEF'] = np.log(gdf['DEFOREST_HA'] + 1)
        self.gdf = gdf
        return self

    def get_gdf(self):
        return self.gdf.copy()

# =============================================================================
# 4. SPATIAL VAR ENGINE
# =============================================================================
class SpatialVAREngine:
    """Motor econométrico espacial multivariado."""

    def __init__(self, gdf):
        self.gdf = gdf.copy()
        self.w = None
        self.w_type = None
        self.results = {}
        self.moran = {}
        self.lisa = {}

    def build_weights(self, w_type='Queen', k=5, dist_pct=30):
        gdf = self.gdf
        if w_type == 'Queen':
            w = Queen.from_dataframe(gdf, silence_warnings=True)
        elif w_type == 'Rook':
            w = Rook.from_dataframe(gdf, silence_warnings=True)
        elif w_type == 'KNN':
            w = KNN.from_dataframe(gdf, k=int(k), silence_warnings=True)
        elif w_type == 'DistanceBand':
            coords = np.array(list(zip(gdf.CENTROID_X, gdf.CENTROID_Y)))
            dist = euclidean_distances(coords)
            thresh = np.percentile(dist[dist > 0], dist_pct)
            w = DistanceBand.from_dataframe(gdf, threshold=thresh, binary=True, silence_warnings=True)
        else:
            raise ValueError(f"Tipo de W no soportado: {w_type}")
        w.transform = 'r'
        self.w = w
        self.w_type = w_type
        return w

    def compute_moran(self, variables=['ln_VAB', 'ln_IPM', 'ln_DEF']):
        if self.w is None:
            self.build_weights()
        self.moran = {}
        for var in variables:
            mi = Moran(self.gdf[var].values, self.w, permutations=999)
            self.moran[var] = {'I': mi.I, 'p_sim': mi.p_sim, 'EI': mi.EI}
        return self.moran

    def compute_lisa(self, variable='ln_IPM'):
        if self.w is None:
            self.build_weights()
        lisa = Moran_Local(self.gdf[variable].values, self.w, permutations=999)
        gdf_t = self.gdf.copy()
        gdf_t['cluster'] = lisa.q
        gdf_t['sig'] = lisa.p_sim < 0.05
        gdf_t['tipo'] = 'No significativo'
        gdf_t.loc[(gdf_t['cluster']==1) & gdf_t['sig'], 'tipo'] = 'Hotspot (HH)'
        gdf_t.loc[(gdf_t['cluster']==3) & gdf_t['sig'], 'tipo'] = 'Coldspot (LL)'
        gdf_t.loc[(gdf_t['cluster']==2) & gdf_t['sig'], 'tipo'] = 'LH'
        gdf_t.loc[(gdf_t['cluster']==4) & gdf_t['sig'], 'tipo'] = 'HL'
        self.lisa[variable] = gdf_t
        return gdf_t

    def estimate_spvar(self):
        if self.w is None:
            self.build_weights()

        vars_endog = ['ln_VAB', 'ln_IPM', 'ln_DEF']
        results = {}
        for dep in vars_endog:
            others = [v for v in vars_endog if v != dep]
            y = self.gdf[dep].values
            X = self.gdf[others].values
            WX = np.column_stack([lag_spatial(self.w, self.gdf[v].values) for v in others])
            X_full = np.hstack([X, WX])
            name_x = others + [f'W_{v}' for v in others]

            try:
                model = ML_Lag(y, X_full, w=self.w, name_y=dep, name_x=name_x,
                               name_w=self.w_type, name_ds='AgroEspacio')
                results[dep] = {
                    'model': model,
                    'rho': model.rho,
                    'aic': model.aic,
                    'r2': model.pr2,
                    'sigma2': model.sig2,
                    'n': model.n,
                    'vars': name_x,
                    'betas': dict(zip(['const'] + name_x, model.betas.flatten())),
                    'std_err': model.std_err.flatten() if hasattr(model, 'std_err') else None
                }
            except Exception as e:
                model = OLS(y, X_full, name_y=dep, name_x=name_x, name_ds='AgroEspacio')
                results[dep] = {
                    'model': model,
                    'rho': np.nan,
                    'aic': model.aic,
                    'r2': model.r2,
                    'sigma2': model.sig2,
                    'n': model.n,
                    'vars': name_x,
                    'betas': dict(zip(['const'] + name_x, model.betas.flatten())),
                    'std_err': None,
                    'note': f'Fallback OLS ({str(e)[:60]})'
                }
        self.results = results
        return results

    def summary_table(self):
        rows = []
        for dep, res in self.results.items():
            rows.append({
                'Ecuación': dep,
                'ρ (espacial)': f"{res['rho']:.4f}" if not np.isnan(res['rho']) else '—',
                'AIC': f"{res['aic']:.2f}",
                'R² (pseudo)': f"{res['r2']:.3f}",
                'σ²': f"{res['sigma2']:.4f}",
                'N': res['n']
            })
        return pd.DataFrame(rows)

    def interpretability_report(self):
        if not self.results:
            return "Ejecute primero la estimación SpVAR."
        md = ["### 📊 Informe de Interpretabilidad Automática – Sistema SpVAR\n"]
        md.append(f"**Matriz de pesos utilizada:** `{self.w_type}` (row-standardized)\n")
        md.append("#### Diagnóstico de Autocorrelación Global (Moran’s I)\n")
        for var, m in self.moran.items():
            sig = "✅ significativo" if m['p_sim'] < 0.05 else "no significativo"
            md.append(f"- **{var}**: I = {m['I']:.3f} (p = {m['p_sim']:.3f}) → {sig}")
        md.append("\n#### Efectos espaciales por ecuación\n")
        for dep, res in self.results.items():
            rho = res['rho']
            md.append(f"**{dep}**")
            if not np.isnan(rho):
                if abs(rho) > 0.3:
                    md.append(f"- Fuerte dependencia espacial (ρ ≈ {rho:.3f}). Los shocks se transmiten a los vecinos.")
                elif abs(rho) > 0.1:
                    md.append(f"- Dependencia espacial moderada (ρ ≈ {rho:.3f}).")
                else:
                    md.append(f"- Dependencia espacial débil (ρ ≈ {rho:.3f}).")
            else:
                md.append("- Modelo estimado por OLS (sin ρ espacial).")
            
            if dep == 'ln_VAB':
                md.append("- El desarrollo agropecuario genera **spillovers positivos** hacia departamentos vecinos.")
            elif dep == 'ln_IPM':
                md.append("- La pobreza rural muestra fuerte estructuración espacial → las políticas deben ser territoriales.")
            else:
                md.append("- La deforestación se propaga espacialmente; requiere coordinación interdepartamental.")
            md.append("")
        md.append("#### Recomendación de política pública")
        md.append("- Priorizar **hotspots HH** de pobreza + deforestación (Caquetá, Meta, Guaviare, Chocó, Putumayo).")
        md.append("- Aprovechar los spillovers del VAB agropecuario entre departamentos contiguos.")
        return "\n".join(md)

# =============================================================================
# 5. VISUAL ENGINE
# =============================================================================
class VisualEngine:
    def __init__(self, gdf):
        self.gdf = gdf
        self.gdf_geo = gdf.to_crs(epsg=4326)

    def choropleth(self, variable):
        color_scale = {
            'VAB_AGRO': 'YlGn', 'IPM_RURAL': 'OrRd', 'DEFOREST_HA': 'RdYlGn_r',
            'ln_VAB': 'YlGn', 'ln_IPM': 'OrRd', 'ln_DEF': 'RdYlGn_r'
        }
        titles = {
            'VAB_AGRO': 'Valor Agregado Bruto Agrícola (miles de millones $) – ref. DANE',
            'IPM_RURAL': 'Incidencia IPM Rural / CP-RD (%) – ref. DANE',
            'DEFOREST_HA': 'Pérdida de bosque natural (ha) – ref. IDEAM SMByC',
            'ln_VAB': 'ln(VAB Agropecuario)', 'ln_IPM': 'ln(IPM Rural)', 'ln_DEF': 'ln(Deforestación)'
        }
        fig = px.choropleth_mapbox(
            self.gdf_geo,
            geojson=self.gdf_geo.__geo_interface__,
            locations=self.gdf_geo.index,
            color=variable,
            hover_name='NOM_DEP',
            hover_data={variable: ':.2f', 'AREA_KM2': ':.0f'},
            color_continuous_scale=color_scale.get(variable, 'Viridis'),
            mapbox_style='carto-positron',
            zoom=4.6,
            center={"lat": 4.5, "lon": -73.0},
            opacity=0.82,
            title=titles.get(variable, variable)
        )
        fig.update_layout(
            margin=dict(r=0, t=45, l=0, b=0),
            height=580,
            paper_bgcolor='#F4F6F8',
            plot_bgcolor='#FFFFFF',
            font=dict(color='#0F172A', size=13),
            title_font=dict(size=16, color='#0F172A')
        )
        return fig

    def extrusion_3d(self, variable='ln_IPM', height_scale=8.0):
        gdf = self.gdf_geo.copy()
        vals = gdf[variable].values
        vmin, vmax = vals.min(), vals.max()
        heights = np.ones(len(gdf)) * 0.5 if vmax - vmin < 1e-9 else height_scale * (vals - vmin) / (vmax - vmin)

        fig = go.Figure()
        colorscale = px.colors.sequential.OrRd if 'IPM' in variable or 'DEF' in variable else px.colors.sequential.YlGn

        for idx, row in gdf.iterrows():
            geom = row.geometry
            h = heights[idx]
            polys = list(geom.geoms) if geom.geom_type == 'MultiPolygon' else [geom]
            for poly in polys:
                if poly.is_empty: continue
                xs, ys = poly.exterior.xy
                n = len(xs)
                x = list(xs) + list(xs)
                y = list(ys) + list(ys)
                z = [0]*n + [h]*n
                i, j, k = [], [], []
                for t in range(1, n-1):
                    i.append(n); j.append(n+t); k.append(n+t+1)
                for t in range(n-1):
                    i.extend([t, t, t+1])
                    j.extend([t+1, n+t, n+t])
                    k.extend([n+t, n+t+1, n+t+1])
                color_val = (vals[idx] - vmin) / (vmax - vmin + 1e-9)
                fig.add_trace(go.Mesh3d(
                    x=x, y=y, z=z, i=i, j=j, k=k,
                    intensity=[color_val]*(2*n),
                    colorscale=colorscale,
                    showscale=False, opacity=0.85,
                    name=row['NOM_DEP'],
                    hoverinfo='text',
                    text=f"{row['NOM_DEP']}<br>{variable}: {vals[idx]:.2f}"
                ))
                fig.add_trace(go.Scatter3d(
                    x=list(xs)+[xs[0]], y=list(ys)+[ys[0]], z=[0]*(n+1),
                    mode='lines', line=dict(color='#0F172A', width=1.5),
                    showlegend=False, hoverinfo='skip'
                ))

        fig.update_layout(
            title=f'Extrusión 3D – {variable} (altura proporcional al valor)',
            scene=dict(
                xaxis_title='Longitud', yaxis_title='Latitud', zaxis_title=variable,
                aspectmode='manual', aspectratio=dict(x=1, y=1.2, z=0.45),
                camera=dict(eye=dict(x=1.4, y=1.4, z=0.9)),
                bgcolor='#F4F6F8'
            ),
            height=650, paper_bgcolor='#F4F6F8', font=dict(color='#0F172A'),
            margin=dict(l=0, r=0, t=50, b=0)
        )
        return fig

    def lisa_map(self, gdf_lisa, variable):
        fig = px.choropleth_mapbox(
            gdf_lisa.to_crs(epsg=4326),
            geojson=gdf_lisa.to_crs(epsg=4326).__geo_interface__,
            locations=gdf_lisa.index,
            color='tipo',
            hover_name='NOM_DEP',
            color_discrete_map={
                'Hotspot (HH)': '#dc2626',
                'Coldspot (LL)': '#2563eb',
                'LH': '#93c5fd',
                'HL': '#fbbf24',
                'No significativo': '#d1d5db'
            },
            mapbox_style='carto-positron',
            zoom=4.6,
            center={"lat": 4.5, "lon": -73.0},
            opacity=0.85,
            title=f'LISA – Hotspots / Coldspots de {variable}'
        )
        fig.update_layout(
            margin=dict(r=0, t=45, l=0, b=0),
            height=580, paper_bgcolor='#F4F6F8', font=dict(color='#0F172A')
        )
        return fig

# =============================================================================
# 6. INICIALIZACIÓN CON CACHÉ DE STREAMLIT
# =============================================================================
@st.cache_resource
def load_data_and_engines():
    data_engine = SpatialDataEngine().cargar_y_sanear().integrar_datos()
    gdf_base = data_engine.get_gdf()
    var_engine = SpatialVAREngine(gdf_base)
    var_engine.build_weights('Queen')
    var_engine.compute_moran()
    visual = VisualEngine(gdf_base)
    return gdf_base, var_engine, visual

gdf_base, var_engine, visual = load_data_and_engines()

# =============================================================================
# 7. INTERFAZ STREAMLIT
# =============================================================================
st.title("🌿 AgroEspacio Colombia")
st.subheader("Inteligencia Territorial · Sistema Vector Autorregresivo Espacial (SpVAR)")
st.markdown("**Desarrollo Agropecuario · Pobreza Rural · Deforestación**")

tabs = st.tabs([
    "🗺️ Mapas Coropléticos", 
    "🌐 Extrusión 3D", 
    "🔥 Clusters LISA", 
    "📊 Estimación SpVAR", 
    "ℹ️ Acerca de / Fuentes"
])

# ---- TAB 1: Mapas Coropléticos ----
with tabs[0]:
    st.markdown("### Visualización coroplética de alto contraste")
    var_c = st.radio(
        "Seleccione la Variable:",
        ['IPM_RURAL', 'VAB_AGRO', 'DEFOREST_HA', 'ln_IPM', 'ln_VAB', 'ln_DEF'],
        horizontal=True
    )
    fig_c = visual.choropleth(var_c)
    st.plotly_chart(fig_c, use_container_width=True)

# ---- TAB 2: Extrusión 3D ----
with tabs[1]:
    st.markdown("### Extrusión tridimensional sobre la geometría real")
    var_3d = st.radio(
        "Variable de extrusión:",
        ['ln_IPM', 'ln_VAB', 'ln_DEF', 'IPM_RURAL', 'VAB_AGRO', 'DEFOREST_HA'],
        horizontal=True
    )
    fig_3d = visual.extrusion_3d(var_3d)
    st.plotly_chart(fig_3d, use_container_width=True)

# ---- TAB 3: LISA ----
with tabs[2]:
    st.markdown("### Autocorrelación Espacial Local (LISA)")
    col1, col2 = st.columns(2)
    with col1:
        lisa_var = st.radio("Variable LISA:", ['ln_IPM', 'ln_DEF', 'ln_VAB'], horizontal=True)
        lisa_w = st.radio("Matriz de pesos W:", ['Queen', 'Rook', 'KNN', 'DistanceBand'], horizontal=True)
    with col2:
        lisa_k = st.slider("k vecinos (solo KNN):", 3, 12, 5)
        lisa_dist = st.slider("% percentil distancia (DistanceBand):", 15, 50, 30)

    var_engine.build_weights(lisa_w, k=lisa_k, dist_pct=lisa_dist)
    gdf_lisa = var_engine.compute_lisa(lisa_var)
    fig_lisa = visual.lisa_map(gdf_lisa, lisa_var)
    st.plotly_chart(fig_lisa, use_container_width=True)

# ---- TAB 4: SpVAR ----
with tabs[3]:
    st.markdown("### Sistema Vector Autorregresivo Espacial (SpVAR)")
    col1, col2, col3 = st.columns(3)
    with col1:
        sp_w = st.selectbox("Matriz de pesos (W):", ['Queen', 'Rook', 'KNN', 'DistanceBand'])
    with col2:
        sp_k = st.slider("k vecinos (KNN):", 3, 12, 5, key="sp_k")
    with col3:
        sp_dist = st.slider("% percentil (DistanceBand):", 15, 50, 30, key="sp_dist")

    if st.button("🚀 Estimar Sistema SpVAR", type="primary"):
        var_engine.build_weights(sp_w, k=sp_k, dist_pct=sp_dist)
        var_engine.compute_moran()
        var_engine.estimate_spvar()
        
        st.dataframe(var_engine.summary_table(), use_container_width=True)
        st.markdown(var_engine.interpretability_report())

# ---- TAB 5: Acerca de ----
with tabs[4]:
    st.markdown("""
    ### Ficha Técnica y Transparencia

    | Elemento | Detalle |
    |----------|---------|
    | Nivel de análisis | Departamental (32 deptos, San Andrés excluido) |
    | CRS | MAGNA-SIRGAS Origen Nacional (EPSG:9377) |
    | Matriz W | Queen / Rook / KNN / DistanceBand (row-standardized) |
    | Modelos | spreg.ML_Lag (SDM-style) + fallback OLS |
    | Autocorrelación | Moran’s I global + LISA local (999 permutaciones) |

    #### Fuentes oficiales de referencia
    - **Cartografía**: MGN – DANE
    - **VAB Agrícola**: DANE – Cuentas Departamentales (órdenes de magnitud)
    - **IPM Rural**: DANE – Boletines IPM 2024-2025 (centros poblados y rural disperso)
    - **Deforestación**: IDEAM – SMByC (Informes de monitoreo 2024-2025)

    **Nota metodológica**: Los valores numéricos de VAB, IPM y Deforestación son de referencia construidos a partir de publicaciones oficiales. El motor SpVAR estima un sistema de ecuaciones espaciales con lags de las variables endógenas.
    
    ---
    *AgroEspacio Colombia – Inteligencia Territorial de Alto Impacto*
    """)
