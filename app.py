import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from textblob import TextBlob
from wordcloud import WordCloud
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from scipy.stats import chi2_contingency, norm
import spacy

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Lexicométrico", layout="wide")

# --- INYECCIÓN CSS ---
st.markdown("""
    <style>
    .block-container {padding-top: 1rem; padding-bottom: 5rem;}
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
        font-weight: 600;
    }
    .stDataFrame {font-size: 1.0rem;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- 2. GESTIÓN DE ESTADO ---
if 'selected_word' not in st.session_state:
    st.session_state['selected_word'] = None

# --- 3. CONFIGURACIÓN GLOBAL ---
LANG_MAP = {'Español': 'spanish', 'Inglés': 'english'}

@st.cache_resource
def download_nltk_resources():
    resources = ['punkt', 'punkt_tab', 'stopwords']
    for res in resources:
        try:
            nltk.data.find(f'tokenizers/{res}')
        except LookupError:
            nltk.download(res)
        except ValueError:
            try: nltk.data.find(f'corpora/{res}')
            except LookupError: nltk.download(res)
download_nltk_resources()

# --- CARGA SIMPLE DE SPACY (Ya instalado por requirements.txt) ---
@st.cache_resource
def load_spacy_model():
    # Ya no intentamos descargar aquí, confiamos en requirements.txt
    try:
        nlp = spacy.load("es_core_news_sm")
    except OSError:
        # Fallback silencioso a inglés si fallara la instalación (para no romper la app)
        st.warning("Modelo español no encontrado. Usando tokenización simple.")
        return None
    return nlp

# --- 4. FUNCIONES ---
def clean_text(text, language='Español', custom_stops=[], min_len=2, apply_lemma=False):
    if pd.isna(text): return []
    lang_code = LANG_MAP.get(language, 'spanish')
    stop_words = set(stopwords.words(lang_code))
    stop_words.update(set(custom_stops))
    
    tokens = []
    
    # Lógica de lematización
    if apply_lemma and language == 'Español':
        nlp = load_spacy_model()
        if nlp:
            nlp.max_length = 2000000 
            doc = nlp(str(text).lower())
            tokens = [token.lemma_ for token in doc if token.is_alpha and token.lemma_ not in stop_words and len(token.lemma_) >= min_len]
        else:
            # Fallback si nlp es None
            tokens = word_tokenize(str(text).lower())
            tokens = [word for word in tokens if word.isalpha() and word not in stop_words and len(word) >= min_len]
    else:
        tokens = word_tokenize(str(text).lower())
        tokens = [word for word in tokens if word.isalpha() and word not in stop_words and len(word) >= min_len]
    
    return tokens

def get_significance_stars(p_value):
    if p_value < 0.001: return "***"
    if p_value < 0.01:  return "**"
    if p_value < 0.05:  return "*"
    return "NS"

def calculate_mtld(tokens, threshold=0.72):
    def count_factors(token_list):
        factors = 0; ttr = 1.0; word_set = set(); length = 0
        for word in token_list:
            length += 1; word_set.add(word); ttr = len(word_set) / length
            if ttr < threshold: factors += 1; length = 0; word_set = set(); ttr = 1.0
        if length > 0: factors += (1 - ttr) / (1 - threshold)
        return factors
    if not tokens: return 0
    return len(tokens) / ((count_factors(tokens) + count_factors(tokens[::-1])) / 2)

def simple_correspondence_analysis(contingency_table):
    X = contingency_table.values
    X = X[~np.all(X == 0, axis=1)]
    X = X[:, ~np.all(X == 0, axis=0)]
    N = np.sum(X)
    P = X / N
    r = np.sum(P, axis=1)
    c = np.sum(P, axis=0)
    Dr_inv_sqrt = np.diag(1 / np.sqrt(r))
    Dc_inv_sqrt = np.diag(1 / np.sqrt(c))
    expected = np.outer(r, c)
    Z = (P - expected) / np.sqrt(expected)
    U, s, Vt = np.linalg.svd(Z, full_matrices=False)
    row_coords = Dr_inv_sqrt @ U[:, :2] @ np.diag(s[:2])
    col_coords = Dc_inv_sqrt @ Vt.T[:, :2] @ np.diag(s[:2])
    return row_coords, col_coords, s

# --- 5. INTERFAZ ---
st.title("📊 Lexicométrico")

st.sidebar.header("1. Carga de Datos")
uploaded_file = st.sidebar.file_uploader("Arrastrar y soltar archivo CSV aquí", type=["csv"])
lang_opt = st.sidebar.selectbox("Idioma del texto", ["Español", "Inglés"])

st.sidebar.markdown("---")
st.sidebar.header("2. Procesamiento")
use_lemmatization = st.sidebar.checkbox("Aplicar Lematización", help="Agrupa variantes (ej: políticos, política -> político)")

st.sidebar.header("3. Filtros")
min_freq_filter = st.sidebar.slider("Seleccione Frecuencia mínima de aparición:", 1, 50, 2)
custom_stopwords_input = st.sidebar.text_area("Excluir palabras (separar por coma):", placeholder="ej: respuesta, ns, nc")
custom_stopwords_list = [x.strip().lower() for x in custom_stopwords_input.split(',')] if custom_stopwords_input else []

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        text_col = df.columns[-1]
        cat_cols = df.columns[:-1].tolist()

        msg = 'Procesando corpus textual...'
        if use_lemmatization: msg += ' (Lematizando...)'
        
        with st.spinner(msg):
            df['tokens'] = df[text_col].apply(lambda x: clean_text(x, lang_opt, custom_stopwords_list, 2, use_lemmatization))
            all_tokens_raw = [t for sub in df['tokens'] for t in sub]
            freq_raw = Counter(all_tokens_raw)
            valid_words = set(w for w, c in freq_raw.items() if c >= min_freq_filter)
            df['tokens'] = df['tokens'].apply(lambda tokens: [t for t in tokens if t in valid_words])
            df['str_processed'] = df['tokens'].apply(lambda x: ' '.join(x))
            df['polaridad'] = df[text_col].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
            all_tokens = [token for sublist in df['tokens'] for token in sublist]

        if len(all_tokens) == 0:
            st.error("No hay palabras suficientes con los filtros actuales.")
        else:
            freq_dist = Counter(all_tokens)
            top_n = 40
            common_words = freq_dist.most_common(top_n)
            df_freq = pd.DataFrame(common_words, columns=['Término', 'Frecuencia'])
            
            available_words = set(df_freq['Término'])
            if st.session_state['selected_word'] is None or st.session_state['selected_word'] not in available_words:
                if not df_freq.empty: st.session_state['selected_word'] = df_freq.iloc[0]['Término']

            if len(df_freq) > 5:
                vectorizer = TfidfVectorizer(vocabulary=df_freq['Término'].values)
                X = vectorizer.fit_transform(df['str_processed'])
                kmeans = KMeans(n_clusters=min(5, len(df_freq)), random_state=42, n_init=10)
                word_to_cluster = dict(zip(vectorizer.get_feature_names_out(), kmeans.fit_predict(X.T)))
                df_freq['Grupo'] = df_freq['Término'].map(word_to_cluster).astype(str)
            else:
                df_freq['Grupo'] = '0'
                word_to_cluster = {w: '0' for w in df_freq['Término']}

            palette = px.colors.qualitative.Bold 
            unique_groups = sorted(df_freq['Grupo'].unique())
            group_color_map = {grp: palette[i % len(palette)] for i, grp in enumerate(unique_groups)}
            word_color_map = {row['Término']: group_color_map[row['Grupo']] for _, row in df_freq.iterrows()}
            def color_func(word, **kwargs): return word_color_map.get(word, '#888888')

            # --- PESTAÑAS ---
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "📊 Frecuencia & KWIC", "🔥 Mapa calor", "🤝 Similitud vocabularios", 
                "🕸️ Red co-ocurrencias", "🗺️ An. correspondencias", "❤️ An. sentimientos"
            ])

            # 1. FRECUENCIA
            with tab1:
                c1, c2 = st.columns([1.2, 0.8])
                with c1:
                    st.subheader("Glosario de términos más utilizados")
                    st.markdown("**Haz clic en una barra para mostrar el contexto:**")
                    fig_bar = px.bar(df_freq, x='Frecuencia', y='Término', orientation='h', color='Grupo', text='Frecuencia', color_discrete_map=group_color_map)
                    fig_bar.update_layout(yaxis=dict(categoryorder='total ascending', tickfont=dict(size=16, color='black', family="Arial Black")), xaxis=dict(showticklabels=False), showlegend=False, height=650, margin=dict(l=0,r=0,t=0,b=0))
                    fig_bar.update_traces(textposition='outside', textfont_size=18, cliponaxis=False, width=0.7)
                    event = st.plotly_chart(fig_bar, use_container_width=True, on_select="rerun", key="bar")
                    if event and event['selection']['points']: st.session_state['selected_word'] = event['selection']['points'][0]['y']
                with c2:
                    st.subheader("Nube Semántica")
                    wc = WordCloud(width=500, height=500, background_color='white', max_words=top_n, color_func=color_func, prefer_horizontal=1.0, relative_scaling=0, margin=0, min_font_size=8).generate_from_frequencies(dict(common_words))
                    fig_wc, ax = plt.subplots(figsize=(6,6))
                    ax.imshow(wc, interpolation='bilinear'); ax.axis('off')
                    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                    st.pyplot(fig_wc)
                
                st.markdown("---"); st.markdown("### 📝 Análisis de Contexto (KWIC)")
                c_s, c_i = st.columns([1,3])
                with c_s: m_search = st.text_input("🔍 Buscar palabra:", value="")
                cur_w = m_search if m_search else st.session_state['selected_word']
                st.markdown(f"<div style='background-color:#f0f2f6; padding:15px; border-radius:10px; border-left: 6px solid #e74c3c;'><h4 style='margin:0; color:#2c3e50;'>Contextos para: <span style='color:#c0392b; font-size:1.3em;'>{cur_w}</span></h4></div>", unsafe_allow_html=True)
                if cur_w:
                    mask = df[text_col].str.contains(cur_w, case=False, na=False)
                    if len(df[mask]) > 0: st.dataframe(df[mask][[cat_cols[0], text_col]], use_container_width=True, hide_index=True)
                    else: st.warning(f"No se encontraron coincidencias directas.")

            # 2. MAPA CALOR
            with tab2:
                cat_heat = st.selectbox("Seleccione la Variable Categórica (Filas):", cat_cols)
                st.markdown(f"### {cat_heat} vs Vocabulario")
                with st.info("Guía: Residuos positivos (rojo) = sobre-uso; negativos = infra-uso. *** = Alta significancia."): pass
                df_exp = df.explode('tokens')
                df_heat = df_exp[df_exp['tokens'].isin(df_freq['Término'].head(20))]
                if not df_heat.empty:
                    obs = pd.crosstab(df_heat[cat_heat], df_heat['tokens'])
                    custom_colors = [[0.0, "#FFFFCC"], [0.2, "#FED976"], [0.4, "#FD8D3C"], [0.6, "#E31A1C"], [0.8, "#800026"], [1.0, "#4A0012"]]
                    st.subheader("1. Representación Visual")
                    fig_h = px.imshow(obs, text_auto=False, aspect="auto", color_continuous_scale=custom_colors, labels=dict(x="", y=cat_heat, color="Freq"))
                    fig_h.update_layout(height=500, yaxis_title=cat_heat)
                    fig_h.update_xaxes(side="top", tickfont=dict(size=14)); fig_h.update_yaxes(tickfont=dict(size=14), tickmode='linear', dtick=1)
                    st.plotly_chart(fig_h, use_container_width=True)
                    st.subheader("2. Tabla de Estadísticos y Significación")
                    chi2, p, dof, ex = chi2_contingency(obs)
                    res = (obs - ex) / np.sqrt(ex)
                    p_val = np.array(2 * (1 - norm.cdf(abs(res))))
                    stats = []
                    for c in obs.index:
                        for w in obs.columns:
                            stats.append({"Categoría": c, "Término": w, "Frecuencia": int(obs.loc[c,w]), "Valor-p": f"{p_val[obs.index.get_loc(c), obs.columns.get_loc(w)]:.3f}", "Sig.": get_significance_stars(p_val[obs.index.get_loc(c), obs.columns.get_loc(w)])})
                    st.dataframe(pd.DataFrame(stats), use_container_width=False, height=400, width=800, hide_index=True)
                else: st.warning("Datos insuficientes.")

            # 3. SIMILITUD
            with tab3:
                cat_v = st.selectbox("Comparar grupos de la variable:", cat_cols, key='voc')
                st.subheader("1. Matriz de Similitud (Jaccard)")
                grp = df.groupby(cat_v)['tokens'].apply(list)
                g_vocab = {k: set([i for s in v for i in s]) for k, v in grp.items()}
                g_list = sorted(list(g_vocab.keys())); sz = len(g_list)
                jac = np.zeros((sz, sz))
                for i in range(sz):
                    for j in range(sz):
                        u = len(g_vocab[g_list[i]].union(g_vocab[g_list[j]]))
                        jac[i,j] = len(g_vocab[g_list[i]].intersection(g_vocab[g_list[j]])) / u if u > 0 else 0
                fig_j = px.imshow(jac, x=g_list, y=g_list, text_auto='.2f', color_continuous_scale='Blues', range_color=[0,1], title=f"Jaccard: {cat_v}")
                fig_j.update_layout(height=500, xaxis=dict(tickmode='linear', dtick=1, side="top"), yaxis=dict(tickmode='linear', dtick=1))
                st.plotly_chart(fig_j, use_container_width=True)
                st.markdown("---"); st.subheader("2. Métricas de Diversidad Léxica")
                with st.info("MTLD: Medida robusta de riqueza léxica (indep. longitud). Valores altos = Mayor riqueza."): pass
                div_d = []
                for k, v in grp.items():
                    fl = [i for s in v for i in s]; n=len(fl); u=len(set(fl))
                    div_d.append({"Categoría": k, "Total (N)": n, "Únicas (V)": u, "TTR": round(u/n if n else 0,3), "MTLD": round(calculate_mtld(fl),2)})
                c_d1, c_d2 = st.columns([1,1])
                with c_d1: st.dataframe(pd.DataFrame(div_d), use_container_width=True, hide_index=True)
                with c_d2: st.plotly_chart(px.bar(div_d, x='Categoría', y=['MTLD', 'TTR'], barmode='group'), use_container_width=True)

            # 4. REDES
            with tab4:
                st.subheader("Red de co-ocurrencias")
                st.markdown("**Interpretación:** Nodos = Palabras (Tamaño proporcional a frecuencia). Líneas = Co-ocurrencia.")
                vec = CountVectorizer(max_features=40, stop_words=stopwords.words(LANG_MAP.get(lang_opt, 'spanish')))
                try:
                    Xn = vec.fit_transform(df['str_processed'])
                    adj = (Xn.T * Xn); adj.setdiag(0)
                    G = nx.from_pandas_adjacency(pd.DataFrame(adj.toarray(), index=vec.get_feature_names_out(), columns=vec.get_feature_names_out()))
                    edges_del = [(u,v) for u,v,d in G.edges(data=True) if d['weight'] < 2]
                    G.remove_edges_from(edges_del); G.remove_nodes_from(list(nx.isolates(G)))
                    
                    # Normalización Min-Max corregida
                    node_freqs_values = [freq_dist.get(node, 1) for node in G.nodes()]
                    if node_freqs_values:
                        min_f, max_f = min(node_freqs_values), max(node_freqs_values)
                        if max_f > min_f:
                            node_sizes = [300 + ((f - min_f) / (max_f - min_f) * 2200) for f in node_freqs_values]
                        else:
                            node_sizes = [1000 for _ in node_freqs_values]
                    else: node_sizes = []

                    fig_net, ax_net = plt.subplots(figsize=(7, 5))
                    pos = nx.spring_layout(G, k=0.6, seed=42)
                    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='#aaddff', alpha=0.9, ax=ax_net)
                    nx.draw_networkx_edges(G, pos, edge_color='#cccccc', width=1, ax=ax_net)
                    nx.draw_networkx_labels(G, pos, font_size=9, ax=ax_net)
                    ax_net.axis('off'); st.pyplot(fig_net)
                except Exception as e: st.warning(f"Datos insuficientes: {e}")

            # 5. CORRESPONDENCIAS
            with tab5:
                st.subheader("Análisis de Correspondencias Simple (ACS)")
                st.info("Mapa perceptual: La cercanía entre puntos indica asociación fuerte.")
                ac_mode = st.radio("Seleccione Tipo de Análisis:", ["🔀 Cruce de Variables (Variables Activas)", "📄 Discurso Puro (Lexical CA - Documentos como base)"])
                cat_illus = []
                if ac_mode.startswith("🔀"):
                    cat_active = st.multiselect("Variables Activas (Definen ejes):", cat_cols, key='ca_act')
                    avail_ill = [c for c in cat_cols if c not in cat_active]
                    cat_illus = st.multiselect("Variables Ilustrativas (Se proyectan):", avail_ill, key='ca_ill_1')
                    if cat_active:
                        df['active_group'] = df[cat_active].apply(lambda x: '_'.join(x.astype(str)), axis=1)
                        df_exp = df.explode('tokens'); top_30 = df_freq['Término'].head(30).tolist(); df_ca = df_exp[df_exp['tokens'].isin(top_30)]
                        cont_table = pd.crosstab(df_ca['active_group'], df_ca['tokens']) if not df_ca.empty else pd.DataFrame()
                    else: st.info("👆 Seleccione variables activas."); cont_table = pd.DataFrame()
                else:
                    st.markdown("**Análisis Léxico:** Ejes basados en documentos."); cat_illus = st.multiselect("Variables Ilustrativas (Proyectar categorías):", cat_cols, key='ca_ill_2')
                    vec_ca = CountVectorizer(max_features=30, stop_words=stopwords.words(LANG_MAP.get(lang_opt, 'spanish')))
                    X_docs = vec_ca.fit_transform(df['str_processed'])
                    cont_table = pd.DataFrame(X_docs.toarray(), index=df.index, columns=vec_ca.get_feature_names_out())
                    cont_table = cont_table.loc[(cont_table!=0).any(axis=1)]

                if not cont_table.empty and cont_table.shape[0] > 1 and cont_table.shape[1] > 1:
                    row_coords, col_coords, s_vals = simple_correspondence_analysis(cont_table)
                    inertia = s_vals**2; expl = inertia / np.sum(inertia); d1, d2 = expl[0]*100, expl[1]*100
                    fig_ca = go.Figure()
                    if ac_mode.startswith("🔀"):
                        fig_ca.add_trace(go.Scatter(x=row_coords[:,0], y=row_coords[:,1], mode='markers+text', text=cont_table.index, textposition="top center", marker=dict(size=12, color='red', symbol='square'), name="Activas"))
                    fig_ca.add_trace(go.Scatter(x=col_coords[:,0], y=col_coords[:,1], mode='markers+text', text=cont_table.columns, textposition="bottom center", marker=dict(size=8, color='blue', opacity=0.6), name="Palabras"))
                    
                    if cat_illus:
                        for ill_var in cat_illus:
                            subset_df = df.loc[cont_table.index]; ill_coords_list = []; ill_names = []
                            for cat_val in subset_df[ill_var].unique():
                                idx_group = subset_df[subset_df[ill_var] == cat_val].index
                                numeric_indices = [cont_table.index.get_loc(i) for i in idx_group if i in cont_table.index]
                                if numeric_indices:
                                    ill_coords_list.append(np.mean(row_coords[numeric_indices], axis=0))
                                    ill_names.append(f"{ill_var}:{cat_val}")
                            if ill_coords_list:
                                ill_arr = np.array(ill_coords_list)
                                fig_ca.add_trace(go.Scatter(x=ill_arr[:,0], y=ill_arr[:,1], mode='markers+text', text=ill_names, textposition="top center", marker=dict(size=10, color='green', symbol='diamond'), name=f"Ilustrativa: {ill_var}"))

                    fig_ca.update_layout(title=f"Mapa Perceptual (Dim 1: {d1:.1f}% + Dim 2: {d2:.1f}%)", xaxis_title="Dim 1", yaxis_title="Dim 2", height=600, template="plotly_white")
                    fig_ca.add_vline(x=0, line_width=1, line_dash="dash", line_color="gray"); fig_ca.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray")
                    st.plotly_chart(fig_ca, use_container_width=True)

            # 6. SENTIMIENTOS
            with tab6:
                c1, c2 = st.columns(2)
                with c1: st.plotly_chart(px.histogram(df, x='polaridad', nbins=20, title="Distribución Polaridad", color_discrete_sequence=['teal']), use_container_width=True)
                with c2: 
                    cat_s = st.selectbox("Variable:", cat_cols, key='s')
                    st.plotly_chart(px.box(df, x=cat_s, y='polaridad', color=cat_s, title="Polaridad por Categoría"), use_container_width=True)

    except Exception as e: st.error(f"Error procesando el archivo: {e}")
