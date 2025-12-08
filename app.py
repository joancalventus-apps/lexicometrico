import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
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
import os

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Lexicométrico", layout="wide")

# --- INYECCIÓN CSS ---
st.markdown("""
    <style>
    .block-container {padding-top: 1rem; padding-bottom: 5rem;}
    
    /* ETIQUETAS PESTAÑAS */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.3rem;
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

# --- 3. CONFIGURACIÓN GLOBAL & MODELOS ---
LANG_MAP = {'Español': 'spanish', 'Inglés': 'english'}

# Descarga de recursos NLTK
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Función para cargar/descargar modelo spaCy solo si es necesario
@st.cache_resource
def load_spacy_model():
    model_name = "es_core_news_sm"
    try:
        nlp = spacy.load(model_name)
    except OSError:
        from spacy.cli import download
        download(model_name)
        nlp = spacy.load(model_name)
    return nlp

# --- 4. FUNCIONES DE PROCESAMIENTO ---

def clean_text(text, language='Español', custom_stops=[], min_len=2, apply_lemma=False):
    if pd.isna(text): return []
    
    # 1. Limpieza básica y Tokenización
    lang_code = LANG_MAP.get(language, 'spanish')
    stop_words = set(stopwords.words(lang_code))
    stop_words.update(set(custom_stops))
    
    # 2. Lematización con SpaCy (Si está activado)
    if apply_lemma and language == 'Español':
        nlp = load_spacy_model()
        doc = nlp(str(text).lower())
        tokens = [token.lemma_ for token in doc if token.is_alpha and token.lemma_ not in stop_words and len(token.lemma_) >= min_len]
    else:
        # Tokenización estándar NLTK
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
        factors = 0
        ttr = 1.0
        word_set = set()
        length = 0
        for word in token_list:
            length += 1
            word_set.add(word)
            ttr = len(word_set) / length
            if ttr < threshold:
                factors += 1
                length = 0
                word_set = set()
                ttr = 1.0
        if length > 0:
            factors += (1 - ttr) / (1 - threshold)
        return factors

    if not tokens: return 0
    f_forward = count_factors(tokens)
    f_backward = count_factors(tokens[::-1])
    factor_avg = (f_forward + f_backward) / 2
    return len(tokens) / factor_avg if factor_avg > 0 else 0

# --- 5. INTERFAZ PRINCIPAL ---

st.title("📊 Lexicométrico")

# --- BARRA LATERAL ---
st.sidebar.header("1. Carga de Datos")
uploaded_file = st.sidebar.file_uploader("Arrastrar y soltar archivo CSV aquí", type=["csv"])
lang_opt = st.sidebar.selectbox("Idioma del texto", ["Español", "Inglés"])

# --- NUEVA OPCIÓN DE LEMATIZACIÓN ---
st.sidebar.markdown("---")
st.sidebar.header("2. Procesamiento")
use_lemmatization = st.sidebar.checkbox("Aplicar Lematización", help="Agrupa palabras (ej: políticos, política -> político)")

st.sidebar.header("3. Filtros")
min_freq_filter = st.sidebar.slider("Seleccione Frecuencia mínima de aparición:", 1, 50, 2)
custom_stopwords_input = st.sidebar.text_area("Excluir palabras (separar por coma):", placeholder="ej: respuesta, ns, nc")
custom_stopwords_list = [x.strip().lower() for x in custom_stopwords_input.split(',')] if custom_stopwords_input else []

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        text_col = df.columns[-1]
        cat_cols = df.columns[:-1].tolist()

        with st.spinner('Procesando corpus textual (esto puede tardar si usas lematización)...'):
            # Aplicamos la función clean_text con el parámetro use_lemmatization
            df['tokens'] = df[text_col].apply(lambda x: clean_text(x, lang_opt, custom_stopwords_list, min_len=2, apply_lemma=use_lemmatization))
            
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
            # --- CÁLCULOS BASE ---
            freq_dist = Counter(all_tokens)
            top_n = 40
            common_words = freq_dist.most_common(top_n)
            df_freq = pd.DataFrame(common_words, columns=['Término', 'Frecuencia'])
            
            # Default Selection
            available_words = set(df_freq['Término'])
            if st.session_state['selected_word'] is None or st.session_state['selected_word'] not in available_words:
                if not df_freq.empty:
                    st.session_state['selected_word'] = df_freq.iloc[0]['Término']

            # Clustering
            if len(df_freq) > 5:
                vectorizer = TfidfVectorizer(vocabulary=df_freq['Término'].values)
                X = vectorizer.fit_transform(df['str_processed'])
                n_clusters = min(5, len(df_freq))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(X.T)
                word_to_cluster = dict(zip(vectorizer.get_feature_names_out(), clusters))
                df_freq['Grupo'] = df_freq['Término'].map(word_to_cluster).astype(str)
            else:
                df_freq['Grupo'] = '0'
                word_to_cluster = {w: '0' for w in df_freq['Término']}

            palette = px.colors.qualitative.Bold 
            unique_groups = sorted(df_freq['Grupo'].unique())
            group_color_map = {grp: palette[i % len(palette)] for i, grp in enumerate(unique_groups)}
            word_color_map = {row['Término']: group_color_map[row['Grupo']] for _, row in df_freq.iterrows()}

            def color_func(word, **kwargs):
                return word_color_map.get(word, '#888888')

            # --- PESTAÑAS ---
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Frecuencia & KWIC", "🔥 Mapa de calor", "🤝 Similitud entre vocabularios", "🕸️ Redes", "❤️ Sentimientos"])

            # --- 1. FRECUENCIA ---
            with tab1:
                col_left, col_right = st.columns([1.2, 0.8])
                with col_left:
                    st.subheader("Glosario de términos más utilizados")
                    st.markdown("**Haz clic en una barra para mostrar el contexto:**")
                    fig_bar = px.bar(
                        df_freq, x='Frecuencia', y='Término', orientation='h', 
                        color='Grupo', text='Frecuencia', color_discrete_map=group_color_map
                    )
                    fig_bar.update_layout(
                        yaxis=dict(categoryorder='total ascending', tickfont=dict(size=16, color='black', family="Arial Black")),
                        xaxis=dict(showticklabels=False),
                        showlegend=False, height=650, margin=dict(l=0, r=0, t=0, b=0)
                    )
                    fig_bar.update_traces(textposition='outside', textfont_size=18, cliponaxis=False, width=0.7)
                    event_bar = st.plotly_chart(fig_bar, use_container_width=True, on_select="rerun", key="bar_chart")
                    if event_bar and event_bar['selection']['points']:
                        st.session_state['selected_word'] = event_bar['selection']['points'][0]['y']

                with col_right:
                    st.subheader("Nube Semántica")
                    wc = WordCloud(
                        width=500, height=500, background_color='white', 
                        max_words=top_n, color_func=color_func, 
                        prefer_horizontal=1.0, relative_scaling=0, margin=0, min_font_size=8
                    ).generate_from_frequencies(dict(common_words))
                    fig_wc, ax = plt.subplots(figsize=(6,6))
                    ax.imshow(wc, interpolation='bilinear'); ax.axis('off')
                    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                    st.pyplot(fig_wc)

                st.markdown("---")
                st.markdown("### 📝 Análisis de Contexto (KWIC)")
                col_search, col_info = st.columns([1, 3])
                with col_search:
                     manual_search = st.text_input("🔍 O escribe una palabra aquí:", value="")
                current_word = manual_search if manual_search else st.session_state['selected_word']

                st.markdown(f"""
                <div style="background-color:#f0f2f6; padding:15px; border-radius:10px; border-left: 6px solid #e74c3c;">
                    <h4 style="margin:0; color:#2c3e50;">
                        Contextos para: <span style="color:#c0392b; font-size:1.3em;">{current_word}</span>
                    </h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Búsqueda KWIC (Mejorada para coincidir con lemas si aplica)
                if current_word:
                    # Si se lematizó, la búsqueda debe ser flexible o sobre la columna original
                    # Aquí buscamos sobre el texto original para mostrar la frase real
                    mask = df[text_col].str.contains(current_word, case=False, na=False)
                    resul = df[mask]
                    if len(resul) > 0:
                        st.dataframe(resul[[cat_cols[0], text_col]], use_container_width=True, hide_index=True)
                    else:
                        st.warning(f"No se encontraron coincidencias directas para '{current_word}'.")

            # --- 2. MAPA DE CALOR ---
            with tab2:
                cat_heatmap = st.selectbox("Seleccione la Variable Categórica (Filas):", cat_cols)
                st.markdown(f"### {cat_heatmap} vs Vocabulario")
                
                with st.info("Guía: Residuos positivos (rojo) indican sobre-uso; negativos indican infra-uso. *** = Alta significancia."):
                    pass
                
                df_exploded = df.explode('tokens')
                top_words_list = df_freq['Término'].head(20).tolist()
                df_heatmap_filtered = df_exploded[df_exploded['tokens'].isin(top_words_list)]
                
                if not df_heatmap_filtered.empty:
                    observed = pd.crosstab(df_heatmap_filtered[cat_heatmap], df_heatmap_filtered['tokens'])
                    custom_colors = [[0.0, "#FFFFCC"], [0.2, "#FED976"], [0.4, "#FD8D3C"], [0.6, "#E31A1C"], [0.8, "#800026"], [1.0, "#4A0012"]]

                    st.subheader("1. Representación Visual")
                    fig_heat = px.imshow(
                        observed, text_auto=False, aspect="auto", color_continuous_scale=custom_colors,
                        labels=dict(x="", y=cat_heatmap, color="Frecuencia")
                    )
                    fig_heat.update_layout(height=500, yaxis_title=cat_heatmap)
                    fig_heat.update_xaxes(side="top", tickfont=dict(size=14))
                    fig_heat.update_yaxes(tickfont=dict(size=14), tickmode='linear', dtick=1)
                    st.plotly_chart(fig_heat, use_container_width=True)
                    
                    st.subheader("2. Tabla de Estadísticos y Significación")
                    chi2, p_global, dof, expected = chi2_contingency(observed)
                    residuals = (observed - expected) / np.sqrt(expected)
                    p_values_matrix = np.array(2 * (1 - norm.cdf(abs(residuals))))
                    
                    stats_data = []
                    for cat in observed.index:
                        for word in observed.columns:
                            freq_val = observed.loc[cat, word]
                            p_val = p_values_matrix[observed.index.get_loc(cat), observed.columns.get_loc(word)]
                            stats_data.append({
                                "Categoría": cat, "Término": word, "Frecuencia": int(freq_val),
                                "Valor-p": f"{p_val:.3f}", "Sig.": get_significance_stars(p_val)
                            })
                    
                    df_stats = pd.DataFrame(stats_data)
                    st.dataframe(
                        df_stats, use_container_width=False, height=400, width=800,
                        column_config={"Categoría": st.column_config.TextColumn(width="small"), "Frecuencia": st.column_config.NumberColumn(width="small")},
                        hide_index=True
                    )
                else:
                    st.warning("No hay suficientes datos cruzados.")

            # --- 3. SIMILITUD ---
            with tab3:
                cat_vocab = st.selectbox("Comparar grupos de la variable:", cat_cols, key='vocab_cat')
                
                st.subheader("1. Matriz de Similitud (Jaccard)")
                df_grouped = df.groupby(cat_vocab)['tokens'].apply(list)
                group_vocab = {cat: set([item for sublist in list_of_lists for item in sublist]) for cat, list_of_lists in df_grouped.items()}
                groups = sorted(list(group_vocab.keys()))
                size = len(groups)
                jaccard_matrix = np.zeros((size, size))
                
                for i in range(size):
                    for j in range(size):
                        if i == j: jaccard_matrix[i, j] = 1.0
                        else:
                            s1, s2 = group_vocab[groups[i]], group_vocab[groups[j]]
                            jaccard_matrix[i, j] = len(s1.intersection(s2)) / len(s1.union(s2)) if len(s1.union(s2)) > 0 else 0
                
                fig_j = px.imshow(jaccard_matrix, x=groups, y=groups, text_auto='.2f', color_continuous_scale='Blues', range_color=[0, 1], title=f"Similitud Jaccard: {cat_vocab}")
                fig_j.update_layout(height=500, xaxis=dict(tickmode='linear', dtick=1, side="top"), yaxis=dict(tickmode='linear', dtick=1))
                st.plotly_chart(fig_j, use_container_width=True)

                st.markdown("---")
                st.subheader("2. Métricas de Diversidad Léxica")
                with st.info("MTLD es ideal para comparar textos de diferente longitud. Valores altos = Mayor riqueza."): pass

                diversity_data = []
                for cat, list_of_lists in df_grouped.items():
                    flat = [item for sublist in list_of_lists for item in sublist]
                    n, v = len(flat), len(set(flat))
                    diversity_data.append({
                        "Categoría": cat, "Total (N)": n, "Únicas (V)": v,
                        "TTR": round(v/n if n>0 else 0, 3), "Guiraud": round(v/np.sqrt(n) if n>0 else 0, 2),
                        "MTLD": round(calculate_mtld(flat), 2)
                    })
                
                col_d1, col_d2 = st.columns([1, 1])
                with col_d1: st.dataframe(pd.DataFrame(diversity_data), use_container_width=True, hide_index=True)
                with col_d2: 
                    st.plotly_chart(px.bar(diversity_data, x='Categoría', y=['MTLD', 'Guiraud'], barmode='group', title="Riqueza Léxica"), use_container_width=True)

            # --- 4. REDES ---
            with tab4:
                st.subheader("Red de Co-ocurrencia")
                st.markdown("""
                **Interpretación:**
                * **Círculo (Nodo):** Palabra. Su tamaño indica la frecuencia de uso.
                * **Línea:** Indica que ambas palabras aparecen juntas.
                """)
                lang_code_net = LANG_MAP.get(lang_opt, 'spanish')
                vectorizer_net = CountVectorizer(max_features=40, stop_words=stopwords.words(lang_code_net))
                try:
                    X_net = vectorizer_net.fit_transform(df['str_processed'])
                    adj = (X_net.T * X_net)
                    adj.setdiag(0)
                    df_cooc = pd.DataFrame(adj.toarray(), index=vectorizer_net.get_feature_names_out(), columns=vectorizer_net.get_feature_names_out())
                    G = nx.from_pandas_adjacency(df_cooc)
                    
                    edges_del = [(u,v) for u,v,d in G.edges(data=True) if d['weight'] < 2]
                    G.remove_edges_from(edges_del)
                    G.remove_nodes_from(list(nx.isolates(G)))
                    
                    # CÁLCULO DE TAMAÑO DE NODOS PROPORCIONAL A FRECUENCIA
                    # 1. Obtener frecuencia del corpus original para los nodos presentes
                    node_sizes = []
                    for node in G.nodes():
                        # freq_dist es el Counter global calculado al inicio
                        freq = freq_dist.get(node, 1) 
                        # Escalamos el tamaño: Base 300 + factor * frecuencia
                        node_sizes.append(300 + (freq * 20))

                    # GRÁFICO REDUCIDO (7x5)
                    fig_net, ax_net = plt.subplots(figsize=(7, 5))
                    pos = nx.spring_layout(G, k=0.6, seed=42)
                    
                    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='#aaddff', alpha=0.9, ax=ax_net)
                    nx.draw_networkx_edges(G, pos, edge_color='#cccccc', width=1, ax=ax_net)
                    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax_net)
                    
                    ax_net.axis('off')
                    st.pyplot(fig_net)
                except Exception as e:
                    st.warning(f"Se necesitan más datos: {e}")

            # --- 5. SENTIMIENTOS ---
            with tab5:
                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(px.histogram(df, x='polaridad', nbins=20, title="Distribución de Polaridad", color_discrete_sequence=['teal']), use_container_width=True)
                with c2:
                    cat_sel = st.selectbox("Cruzar con variable:", cat_cols, key='sent_cat')
                    st.plotly_chart(px.box(df, x=cat_sel, y='polaridad', color=cat_sel, title="Polaridad por Categoría"), use_container_width=True)

    except Exception as e:
        st.error(f"Error procesando el archivo: {e}")
