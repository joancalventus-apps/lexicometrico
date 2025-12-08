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
# CORRECCIÓN AQUÍ: Se añade CountVectorizer que faltaba antes
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="LexicoMapper Pro", layout="wide")

# --- GESTIÓN DE ESTADO (MEMORIA DE LA APP) ---
if 'selected_word' not in st.session_state:
    st.session_state['selected_word'] = ""
if 'current_view' not in st.session_state:
    st.session_state['current_view'] = "📊 Frecuencia & Semántica"

# --- NLTK ---
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')

# --- FUNCIONES ---
def clean_text(text, language='spanish', custom_stops=[], min_len=2):
    if pd.isna(text): return []
    stop_words = set(stopwords.words(language))
    stop_words.update(set(custom_stops))
    tokens = word_tokenize(str(text).lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words and len(word) >= min_len]
    return tokens

# --- INTERFAZ PRINCIPAL ---

st.title("📊 LexicoMapper Pro")

# BARRA LATERAL
st.sidebar.header("1. Datos y Filtros")
uploaded_file = st.sidebar.file_uploader("Subir CSV", type=["csv"])
lang_opt = st.sidebar.selectbox("Idioma", ["spanish", "english"])
custom_stopwords_input = st.sidebar.text_area("Excluir palabras:", placeholder="ej: respuesta, ns, nc")
custom_stopwords_list = [x.strip().lower() for x in custom_stopwords_input.split(',')] if custom_stopwords_input else []
min_freq_filter = st.sidebar.slider("Frecuencia mínima:", 1, 50, 2)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        text_col = df.columns[-1]
        cat_cols = df.columns[:-1].tolist()

        with st.spinner('Procesando corpus...'):
            # Procesamiento de datos
            df['tokens'] = df[text_col].apply(lambda x: clean_text(x, lang_opt, custom_stopwords_list))
            
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
            # --- DATOS PARA GRÁFICOS ---
            freq_dist = Counter(all_tokens)
            top_n = 40
            common_words = freq_dist.most_common(top_n)
            df_freq = pd.DataFrame(common_words, columns=['Término', 'Frecuencia'])
            
            # Clustering Semántico
            if len(df_freq) > 5:
                vectorizer = TfidfVectorizer(vocabulary=df_freq['Término'].values)
                X = vectorizer.fit_transform(df['str_processed'])
                # KMeans con manejo de errores si hay pocos datos
                n_clusters = min(5, len(df_freq))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(X.T)
                word_to_cluster = dict(zip(vectorizer.get_feature_names_out(), clusters))
                df_freq['Grupo'] = df_freq['Término'].map(word_to_cluster).astype(str)
            else:
                df_freq['Grupo'] = '0'
                word_to_cluster = {w: '0' for w in df_freq['Término']}

            # Colores consistentes
            palette = px.colors.qualitative.Bold 
            unique_groups = sorted(df_freq['Grupo'].unique())
            group_color_map = {grp: palette[i % len(palette)] for i, grp in enumerate(unique_groups)}
            word_color_map = {row['Término']: group_color_map[row['Grupo']] for _, row in df_freq.iterrows()}

            def color_func(word, **kwargs):
                return word_color_map.get(word, '#888888')

            # --- MENÚ DE NAVEGACIÓN (Sustituye a st.tabs) ---
            # Usamos radio horizontal para poder controlarlo programáticamente
            options = ["📊 Frecuencia & Semántica", "🔍 KWIC (Concordancias)", "🕸️ Redes", "❤️ Sentimientos"]
            
            # Aseguramos que el estado sea válido
            if st.session_state['current_view'] not in options:
                st.session_state['current_view'] = options[0]

            selection = st.radio("", options, horizontal=True, key="current_view")
            st.markdown("---")

            # --- VISTA 1: FRECUENCIA ---
            if selection == "📊 Frecuencia & Semántica":
                col_left, col_right = st.columns([1, 1])
                
                with col_left:
                    st.subheader("Top Términos (Clic para explorar)")
                    fig_bar = px.bar(
                        df_freq, x='Frecuencia', y='Término', orientation='h', 
                        color='Grupo', text='Frecuencia', color_discrete_map=group_color_map
                    )
                    fig_bar.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False, height=600)
                    
                    # CAPTURAR CLIC
                    event = st.plotly_chart(fig_bar, use_container_width=True, on_select="rerun")
                    
                    # LÓGICA DE SALTO AUTOMÁTICO
                    if event and len(event['selection']['points']) > 0:
                        clicked_word = event['selection']['points'][0]['y']
                        # Si es una nueva selección, actualizamos y recargamos
                        if clicked_word != st.session_state['selected_word']:
                            st.session_state['selected_word'] = clicked_word
                            st.session_state['current_view'] = "🔍 KWIC (Concordancias)" # <--- CAMBIO DE PESTAÑA MÁGICO
                            st.rerun()

                with col_right:
                    st.subheader("Nube Semántica")
                    wc = WordCloud(width=800, height=800, background_color='white', max_words=top_n, color_func=color_func).generate_from_frequencies(dict(common_words))
                    fig_wc, ax = plt.subplots(figsize=(8,8))
                    ax.imshow(wc, interpolation='bilinear'); ax.axis('off')
                    st.pyplot(fig_wc)

            # --- VISTA 2: KWIC ---
            elif selection == "🔍 KWIC (Concordancias)":
                st.header("Contexto de Palabras")
                
                # Botón para volver
                if st.button("⬅️ Volver al gráfico"):
                    st.session_state['current_view'] = "📊 Frecuencia & Semántica"
                    st.rerun()

                search_term = st.text_input("Palabra seleccionada:", value=st.session_state['selected_word'])
                
                if search_term:
                    mask = df['str_processed'].str.contains(search_term, case=False, na=False)
                    resul = df[mask]
                    st.success(f"Contextos encontrados para '{search_term}': {len(resul)}")
                    st.dataframe(resul[[cat_cols[0], text_col]], use_container_width=True, hide_index=True)
                else:
                    st.info("Selecciona una palabra en el gráfico o escribe una aquí.")

            # --- VISTA 3: REDES ---
            elif selection == "🕸️ Redes":
                st.subheader("Red de Co-ocurrencia")
                vectorizer_net = CountVectorizer(max_features=40, stop_words=stopwords.words(lang_opt))
                try:
                    X_net = vectorizer_net.fit_transform(df['str_processed'])
                    adj = (X_net.T * X_net)
                    adj.setdiag(0)
                    df_cooc = pd.DataFrame(adj.toarray(), index=vectorizer_net.get_feature_names_out(), columns=vectorizer_net.get_feature_names_out())
                    G = nx.from_pandas_adjacency(df_cooc)
                    
                    # Filtro visual
                    edges_del = [(u,v) for u,v,d in G.edges(data=True) if d['weight'] < 2]
                    G.remove_edges_from(edges_del)
                    G.remove_nodes_from(list(nx.isolates(G)))
                    
                    fig_net, ax_net = plt.subplots(figsize=(10,6))
                    pos = nx.spring_layout(G, k=0.5)
                    nx.draw(G, pos, with_labels=True, node_color='#aaddff', edge_color='#cccccc', node_size=1000, font_size=9, ax=ax_net)
                    st.pyplot(fig_net)
                except Exception as e:
                    st.warning(f"No hay suficientes datos para la red: {e}")

            # --- VISTA 4: SENTIMIENTOS ---
            elif selection == "❤️ Sentimientos":
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("Histograma de Polaridad")
                    fig_h = px.histogram(df, x='polaridad', nbins=20, color_discrete_sequence=['teal'])
                    st.plotly_chart(fig_h, use_container_width=True)
                with c2:
                    st.subheader("Cruce Categórico")
                    cat_sel = st.selectbox("Variable:", cat_cols)
                    fig_b = px.box(df, x=cat_sel, y='polaridad', color=cat_sel)
                    st.plotly_chart(fig_b, use_container_width=True)

    except Exception as e:
        st.error(f"Error general: {e}")
