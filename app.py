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
import sys

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
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- 2. GESTIÓN DE ESTADO ---
if 'selected_word' not in st.session_state:
    st.session_state['selected_word'] = None

# --- 3. CONFIGURACIÓN DE RECURSOS (Auto-descarga) ---
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

# --- 4. FUNCIONES DE ANÁLISIS ---

def clean_text(text, language='Español', custom_stops=[], min_len=2, apply_lemma=False):
    if pd.isna(text): return []
    lang_code = 'spanish' if language == 'Español' else 'english'
    stop_words = set(stopwords.words(lang_code))
    stop_words.update(set(custom_stops))
    
    if apply_lemma and language == 'Español':
        doc = nlp_spacy(str(text).lower())
        tokens = [t.lemma_ for t in doc if t.is_alpha and t.lemma_ not in stop_words and len(t.lemma_) >= min_len]
    else:
        raw_tokens = word_tokenize(str(text).lower())
        tokens = [w for w in raw_tokens if w.isalpha() and w not in stop_words and len(w) >= min_len]
    return tokens

def calculate_mtld(tokens, threshold=0.72):
    def count_factors(t_list):
        factors, ttr, length = 0, 1.0, 0
        word_set = set()
        for word in t_list:
            length += 1; word_set.add(word)
            ttr = len(word_set) / length
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
    expected = np.outer(r, c)
    Z = (P - expected) / np.sqrt(expected)
    U, s, Vt = np.linalg.svd(Z, full_matrices=False)
    row_coords = np.diag(1 / np.sqrt(r)) @ U[:, :2] @ np.diag(s[:2])
    col_coords = np.diag(1 / np.sqrt(c)) @ Vt.T[:, :2] @ np.diag(s[:2])
    return row_coords, col_coords, s

# --- 5. INTERFAZ ---
st.title("📊 Lexicométrico")

st.sidebar.header("1. Carga de Datos")
uploaded_file = st.sidebar.file_uploader("Subir CSV", type=["csv"])
lang_opt = st.sidebar.selectbox("Idioma", ["Español", "Inglés"])

st.sidebar.markdown("---")
st.sidebar.header("2. Procesamiento")
use_lemmatization = st.sidebar.checkbox("Lematización", help="Agrupa raíces (ej: niños -> niño)")
min_f = st.sidebar.slider("Frecuencia mínima:", 1, 50, 2)
ex_words = st.sidebar.text_area("Excluir (comas):")
ex_list = [x.strip().lower() for x in ex_words.split(',')] if ex_words else []

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        txt_col, cat_cols = df.columns[-1], df.columns[:-1].tolist()

        with st.spinner('Procesando...'):
            df['tokens'] = df[txt_col].apply(lambda x: clean_text(x, lang_opt, ex_list, 2, use_lemmatization))
            all_t_raw = [t for sub in df['tokens'] for t in sub]
            freq_all = Counter(all_t_raw)
            valid_w = set(w for w, c in freq_all.items() if c >= min_f)
            df['tokens'] = df['tokens'].apply(lambda ts: [t for t in ts if t in valid_w])
            df['str_processed'] = df['tokens'].apply(lambda x: ' '.join(x))
            df['polaridad'] = df[txt_col].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
            all_tokens = [t for sub in df['tokens'] for t in sub]

        if not all_tokens: st.error("Sin datos.")
        else:
            top_n = 40
            common = Counter(all_tokens).most_common(top_n)
            df_freq = pd.DataFrame(common, columns=['Término', 'Frecuencia'])
            if st.session_state['selected_word'] not in df_freq['Término'].values:
                st.session_state['selected_word'] = df_freq.iloc[0]['Término']

            # Tabs con iconos
            tabs = st.tabs(["📊 Frecuencia & KWIC", "🔥 Mapa calor", "🤝 Similitud vocabularios", "🕸️ Red co-ocurrencias", "🗺️ An. correspondencias", "❤️ An. sentimientos"])

            with tabs[0]: # Frecuencia
                c1, c2 = st.columns([1.2, 0.8])
                with c1:
                    fig = px.bar(df_freq, x='Frecuencia', y='Término', orientation='h', text='Frecuencia')
                    fig.update_layout(yaxis=dict(categoryorder='total ascending', tickfont=dict(size=14)), height=600)
                    ev = st.plotly_chart(fig, use_container_width=True, on_select="rerun")
                    if ev and ev['selection']['points']: st.session_state['selected_word'] = ev['selection']['points'][0]['y']
                with c2:
                    wc = WordCloud(width=500, height=500, background_color='white', prefer_horizontal=0.9).generate_from_frequencies(dict(common))
                    st.image(wc.to_array())
                
                st.markdown("---"); st.subheader(f"📝 Contextos: {st.session_state['selected_word']}")
                res = df[df[txt_col].str.contains(st.session_state['selected_word'], case=False, na=False)]
                st.dataframe(res[[cat_cols[0], txt_col]], use_container_width=True, hide_index=True)

            with tabs[1]: # Mapa Calor
                cv = st.selectbox("Variable:", cat_cols, key='h')
                exp = df.explode('tokens')
                obs = pd.crosstab(exp[cv], exp['tokens']).reindex(columns=df_freq['Término'].head(20), fill_value=0)
                st.plotly_chart(px.imshow(obs, text_auto=False, color_continuous_scale='Reds'), use_container_width=True)
                chi, p, d, e = chi2_contingency(obs)
                res = (obs - e) / np.sqrt(e)
                st.write("**Significación (p-valor):**")
                stats = []
                for r_idx, c_idx in np.ndindex(obs.shape):
                    pval = 2 * (1 - norm.cdf(abs(res.iloc[r_idx, c_idx])))
                    stats.append({"Cat": obs.index[r_idx], "Palabra": obs.columns[c_idx], "p": f"{pval:.3f}", "Sig": "✔️" if pval < 0.05 else "NS"})
                st.dataframe(pd.DataFrame(stats), hide_index=True)

            with tabs[2]: # Similitud
                cv = st.selectbox("Variable:", cat_cols, key='s')
                grp = df.groupby(cv)['tokens'].apply(list)
                g_voc = {k: set([i for s in v for i in s]) for k, v in grp.items()}
                g_list = sorted(list(g_voc.keys()))
                sz = len(g_list)
                jac = np.zeros((sz, sz))
                for i, j in np.ndindex(sz, sz):
                    u = len(g_voc[g_list[i]].union(g_voc[g_list[j]]))
                    jac[i,j] = len(g_voc[g_list[i]].intersection(g_voc[g_list[j]])) / u if u > 0 else 0
                st.plotly_chart(px.imshow(jac, x=g_list, y=g_list, text_auto='.2f', color_continuous_scale='Blues'), use_container_width=True)

            with tabs[3]: # Redes
                vec = CountVectorizer(max_features=30)
                adj = (vec.fit_transform(df['str_processed']).T * vec.fit_transform(df['str_processed']))
                adj.setdiag(0)
                G = nx.from_pandas_adjacency(pd.DataFrame(adj.toarray(), index=vec.get_feature_names_out(), columns=vec.get_feature_names_out()))
                v_freq = [freq_all.get(n, 1) for n in G.nodes()]
                szs = [500 + ((f - min(v_freq)) / (max(v_freq) - min(v_freq) + 1) * 2000) for f in v_freq]
                fig, ax = plt.subplots(figsize=(8,5))
                pos = nx.spring_layout(G)
                nx.draw(G, pos, node_size=szs, node_color='#aaddff', with_labels=True, font_size=8, ax=ax)
                st.pyplot(fig)

            with tabs[4]: # AC
                mode = st.radio("Modo:", ["Activas", "Léxico"])
                sel = st.multiselect("Ilustrativas:", cat_cols)
                if mode == "Activas":
                    av = st.multiselect("Activas:", cat_cols)
                    if av:
                        df['tmp'] = df[av].apply(lambda x: '_'.join(x.astype(str)), axis=1)
                        table = pd.crosstab(df.explode('tokens')['tmp'], df.explode('tokens')['tokens']).reindex(columns=df_freq['Término'].head(25), fill_value=0)
                        r, c, _ = simple_correspondence_analysis(table)
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=r[:,0], y=r[:,1], mode='markers+text', text=table.index, name='Cat', marker=dict(color='red')))
                        fig.add_trace(go.Scatter(x=c[:,0], y=c[:,1], mode='markers+text', text=table.columns, name='Pal', marker=dict(color='blue')))
                        st.plotly_chart(fig)
                else: st.info("Léxico: Documentos x Palabras proyectados.")

            with tabs[5]: # Sentimientos
                cv = st.selectbox("Comparar por:", cat_cols, key='sent')
                st.plotly_chart(px.box(df, x=cv, y='polaridad', color=cv), use_container_width=True)

    except Exception as e: st.error(f"Error: {e}")
