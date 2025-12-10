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

# --- INYECCIÓN CSS (DISEÑO) ---
st.markdown("""
    <style>
    .block-container {padding-top: 1rem; padding-bottom: 5rem;}
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
        font-weight: 600;
    }
    .stDataFrame {font-size: 1.0rem;}
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- 2. GESTIÓN DE ESTADO ---
if 'selected_word' not in st.session_state:
    st.session_state['selected_word'] = None

# --- 3. RECURSOS (AUTO-DESCARGA) ---
LANG_MAP = {'Español': 'spanish', 'Inglés': 'english'}

@st.cache_resource
def load_resources():
    # NLTK
    for res in ['punkt', 'punkt_tab', 'stopwords']:
        nltk.download(res, quiet=True)
    # spaCy Español
    model_name = "es_core_news_sm"
    try:
        nlp = spacy.load(model_name)
    except OSError:
        from spacy.cli import download
        download(model_name)
        nlp = spacy.load(model_name)
    return nlp

nlp_spacy = load_resources()

# --- 4. FUNCIONES MATEMÁTICAS Y DE LIMPIEZA ---

def clean_text(text, language='Español', custom_stops=[], min_len=2, apply_lemma=False):
    if pd.isna(text): return []
    lang_code = LANG_MAP.get(language, 'spanish')
    stop_words = set(stopwords.words(lang_code))
    stop_words.update(set(custom_stops))
    
    if apply_lemma and language == 'Español':
        doc = nlp_spacy(str(text).lower())
        tokens = [t.lemma_ for t in doc if t.is_alpha and t.lemma_ not in stop_words and len(t.lemma_) >= min_len]
    else:
        raw_tokens = word_tokenize(str(text).lower())
        tokens = [w for w in raw_tokens if w.isalpha() and w not in stop_words and len(w) >= min_len]
    return tokens

def get_significance_stars(p_value):
    if p_value < 0.001: return "***"
    if p_value < 0.01:  return "**"
    if p_value < 0.05:  return "*"
    return "NS"

def calculate_mtld(tokens, threshold=0.72):
    def count_factors(t_list):
        factors, ttr, length, word_set = 0, 1.0, 0, set()
        for word in t_list:
            length += 1; word_set.add(word); ttr = len(word_set) / length
            if ttr < threshold: factors += 1; length, word_set, ttr = 0, set(), 1.0
        if length > 0: factors += (1 - ttr) / (1 - threshold)
        return factors
    if not tokens: return 0
    f_avg = (count_factors(tokens) + count_factors(tokens[::-1])) / 2
    return len(tokens) / f_avg if f_avg > 0 else 0

def simple_correspondence_analysis(table):
    X = table.values
    N = np.sum(X)
    P = X / N
    r, c = np.sum(P, axis=1), np.sum(P, axis=0)
    Dr_inv_sqrt = np.diag(1 / np.sqrt(r))
    Dc_inv_sqrt = np.diag(1 / np.sqrt(c))
    Z = (P - np.outer(r, c)) / np.sqrt(np.outer(r, c))
    U, s, Vt = np.linalg.svd(Z, full_matrices=False)
    row_coords = Dr_inv_sqrt @ U[:, :2] @ np.diag(s[:2])
    col_coords = Dc_inv_sqrt @ Vt.T[:, :2] @ np.diag(s[:2])
    return row_coords, col_coords, s

# --- 5. INTERFAZ PRINCIPAL ---
st.title("📊 Lexicométrico")

st.sidebar.header("1. Carga de Datos")
uploaded_file = st.sidebar.file_uploader("Arrastrar y soltar archivo CSV aquí", type=["csv"])
lang_opt = st.sidebar.selectbox("Idioma", ["Español", "Inglés"])

st.sidebar.markdown("---")
st.sidebar.header("2. Procesamiento")
use_lemma = st.sidebar.checkbox("Lematización", help="Agrupa raíces (ej: niños -> niño)")
min_f = st.sidebar.slider("Frecuencia mínima:", 1, 50, 2)
ex_words = st.sidebar.text_area("Excluir (separar por comas):")
ex_list = [x.strip().lower() for x in ex_words.split(',')] if ex_words else []

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        txt_col, cat_cols = df.columns[-1], df.columns[:-1].tolist()

        with st.spinner('Procesando corpus...'):
            df['tokens'] = df[txt_col].apply(lambda x: clean_text(x, lang_opt, ex_list, 2, use_lemma))
            all_t_raw = [t for sub in df['tokens'] for t in sub]
            freq_raw = Counter(all_t_raw)
            valid_w = set(w for w, c in freq_raw.items() if c >= min_f)
            df['tokens'] = df['tokens'].apply(lambda ts: [t for t in ts if t in valid_w])
            df['str_processed'] = df['tokens'].apply(lambda x: ' '.join(x))
            df['polaridad'] = df[txt_col].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
            all_tokens = [t for sub in df['tokens'] for t in sub]

        if not all_tokens: st.error("No hay palabras con los filtros actuales.")
        else:
            freq_dist = Counter(all_tokens)
            df_freq = pd.DataFrame(freq_dist.most_common(40), columns=['Término', 'Frecuencia'])
            if st.session_state['selected_word'] not in df_freq['Término'].values:
                st.session_state['selected_word'] = df_freq.iloc[0]['Término']

            # Colores consistentes
            palette = px.colors.qualitative.Bold
            kmeans = KMeans(n_clusters=min(5, len(df_freq)), random_state=42, n_init=10)
            clusters = kmeans.fit_predict(TfidfVectorizer(vocabulary=df_freq['Término'].values).fit_transform(df['str_processed']).T)
            df_freq['Grupo'] = clusters.astype(str)
            color_map = {grp: palette[i % len(palette)] for i, grp in enumerate(df_freq['Grupo'].unique())}
            word_color = {row['Término']: color_map[row['Grupo']] for _, row in df_freq.iterrows()}

            # PESTAÑAS
            tabs = st.tabs(["📊 Frecuencia & KWIC", "🔥 Mapa calor", "🤝 Similitud vocabularios", "🕸️ Red co-ocurrencias", "🗺️ An. correspondencias", "❤️ An. sentimientos"])

            with tabs[0]: # Pestaña 1
                cl, cr = st.columns([1.2, 0.8])
                with cl:
                    st.subheader("Glosario de términos más utilizados")
                    fig = px.bar(df_freq, x='Frecuencia', y='Término', orientation='h', color='Grupo', text='Frecuencia', color_discrete_map=color_map)
                    fig.update_layout(yaxis=dict(categoryorder='total ascending', tickfont=dict(size=16, family="Arial Black")), showlegend=False, height=600)
                    ev = st.plotly_chart(fig, use_container_width=True, on_select="rerun")
                    if ev and ev['selection']['points']: st.session_state['selected_word'] = ev['selection']['points'][0]['y']
                with cr:
                    st.subheader("Nube Semántica")
                    wc = WordCloud(width=500, height=800, background_color='white', max_words=40, color_func=lambda w,**k: word_color.get(w, '#888'), prefer_horizontal=0.9).generate_from_frequencies(freq_dist)
                    fig_wc, ax = plt.subplots(figsize=(6,8)); ax.imshow(wc, interpolation='bilinear'); ax.axis('off')
                    st.pyplot(fig_wc)
                st.markdown("---"); st.subheader(f"📝 Contextos: {st.session_state['selected_word']}")
                res = df[df[txt_col].str.contains(st.session_state['selected_word'], case=False, na=False)]
                st.dataframe(res[[cat_cols[0], txt_col]], use_container_width=True, hide_index=True)

            with tabs[1]: # Pestaña 2
                ch = st.selectbox("Variable:", cat_cols, key='h')
                exp = df.explode('tokens'); df_h = exp[exp['tokens'].isin(df_freq['Término'].head(20))]
                if not df_h.empty:
                    obs = pd.crosstab(df_h[ch], df_h['tokens'])
                    st.subheader("1. Representación Visual")
                    custom_c = [[0, "#FFFFCC"], [0.4, "#FD8D3C"], [0.8, "#800026"], [1, "#4A0012"]]
                    st.plotly_chart(px.imshow(obs, aspect="auto", color_continuous_scale=custom_c, labels=dict(y=ch)), use_container_width=True)
                    st.markdown("---"); st.subheader("2. Tabla de Estadísticos")
                    st.info("Guía: *** Muy significativo; NS No significativo.")
                    chi, p, d, e = chi2_contingency(obs)
                    res_m = (obs - e) / np.sqrt(e)
                    p_matrix = 2 * (1 - norm.cdf(abs(res_m)))
                    stats = []
                    for i, r_cat in enumerate(obs.index):
                        for j, c_word in enumerate(obs.columns):
                            pval = p_matrix[i, j]
                            stats.append({"Categoría": r_cat, "Término": c_word, "Frecuencia": int(obs.iloc[i,j]), "Valor-p": f"{pval:.3f}", "Sig": get_significance_stars(pval)})
                    st.dataframe(pd.DataFrame(stats), use_container_width=False, width=800, hide_index=True)

            with tabs[2]: # Pestaña 3
                st.subheader("Análisis de Similitud y Diversidad")
                cv = st.selectbox("Comparar por:", cat_cols, key='sim')
                grp = df.groupby(cv)['tokens'].apply(list)
                g_voc = {k: set([i for s in v for i in s]) for k, v in grp.items()}
                g_l = sorted(list(g_voc.keys())); sz = len(g_l)
                jac = np.zeros((sz, sz))
                for i, j in np.ndindex(sz, sz):
                    u = len(g_voc[g_l[i]].union(g_voc[g_l[j]]))
                    jac[i,j] = len(g_voc[g_l[i]].intersection(g_voc[g_l[j]])) / u if u > 0 else 0
                st.plotly_chart(px.imshow(jac, x=g_l, y=g_l, text_auto='.2f', color_continuous_scale='Blues'), use_container_width=True)
                st.markdown("---"); st.subheader("Diversidad Léxica")
                div_d = []
                for k, v in grp.items():
                    fl = [i for s in v for i in s]; n, u = len(fl), len(set(fl))
                    div_d.append({"Cat": k, "Total (N)": n, "MTLD": round(calculate_mtld(fl),2), "TTR": round(u/n if n else 0,3)})
                st.dataframe(pd.DataFrame(div_d), use_container_width=True, hide_index=True)

            with tabs[3]: # Pestaña 4
                st.subheader("Red de co-ocurrencias")
                v_net = CountVectorizer(max_features=40, stop_words=LANG_MAP.get(lang_opt, 'spanish'))
                try:
                    adj = (v_net.fit_transform(df['str_processed']).T * v_net.fit_transform(df['str_processed']))
                    adj.setdiag(0)
                    G = nx.from_pandas_adjacency(pd.DataFrame(adj.toarray(), index=v_net.get_feature_names_out(), columns=v_net.get_feature_names_out()))
                    f_v = [freq_dist.get(n, 1) for n in G.nodes()]
                    szs = [300 + ((f - min(f_v)) / (max(f_v) - min(f_v) + 1) * 2000) for f in f_v]
                    fig_n, ax_n = plt.subplots(figsize=(7,5))
                    nx.draw(G, nx.spring_layout(G, seed=42), node_size=szs, node_color='#aaddff', with_labels=True, font_size=8, ax=ax_n)
                    st.pyplot(fig_n)
                except: st.error("Datos insuficientes para red.")

            with tabs[4]: # Pestaña 5
                st.subheader("Análisis de Correspondencias")
                ac_m = st.radio("Modo:", ["Variables Activas", "Léxico Puro"])
                if ac_m == "Variables Activas":
                    av = st.multiselect("Activas:", cat_cols)
                    if av:
                        df['tmp'] = df[av].apply(lambda x: '_'.join(x.astype(str)), axis=1)
                        tab_c = pd.crosstab(df['tmp'], df.explode('tokens')['tokens']).reindex(columns=df_freq['Término'].head(25), fill_value=0)
                        r, c, s = simple_correspondence_analysis(tab_c)
                        fig_ac = go.Figure()
                        fig_ac.add_trace(go.Scatter(x=r[:,0], y=r[:,1], mode='markers+text', text=tab_c.index, name='Cat', marker=dict(color='red')))
                        fig_ac.add_trace(go.Scatter(x=c[:,0], y=c[:,1], mode='markers+text', text=tab_c.columns, name='Pal', marker=dict(color='blue')))
                        st.plotly_chart(fig_ac, use_container_width=True)
                else: st.info("Modo Léxico: Documentos x Palabras.")

            with tabs[5]: # Pestaña 6
                st.subheader("Sentimientos")
                cv = st.selectbox("Cruzar:", cat_cols, key='sent')
                st.plotly_chart(px.box(df, x=cv, y='polaridad', color=cv), use_container_width=True)

    except Exception as e: st.error(f"Error: {e}")
