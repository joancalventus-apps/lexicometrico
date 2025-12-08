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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# --- CONFIGURACIÓN INICIAL ---
st.set_page_config(page_title="LexicoMapper Pro", layout="wide")

# Inicializar estado para la selección de palabras
if 'selected_word' not in st.session_state:
    st.session_state['selected_word'] = ""

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

def get_lexical_stats(tokens):
    n_tokens = len(tokens)
    n_types = len(set(tokens))
    ttr = (n_types / n_tokens) * 100 if n_tokens > 0 else 0
    freqs = Counter(tokens)
    hapax = sum(1 for word, count in freqs.items() if count == 1)
    return n_tokens, n_types, ttr, hapax

# --- INTERFAZ ---

st.title("📊 LexicoMapper Pro: Interactivo")
st.markdown("Haz **clic en las barras** del gráfico para filtrar automáticamente las concordancias (KWIC).")

# BARRA LATERAL
st.sidebar.header("1. Datos y Filtros")
uploaded_file = st.sidebar.file_uploader("Subir CSV", type=["csv"])
lang_opt = st.sidebar.selectbox("Idioma", ["spanish", "english"])
custom_stopwords_input = st.sidebar.text_area("Excluir palabras (coma):", placeholder="ej: respuesta, ns, nc")
custom_stopwords_list = [x.strip().lower() for x in custom_stopwords_input.split(',')] if custom_stopwords_input else []
min_freq_filter = st.sidebar.slider("Frecuencia mínima:", 1, 50, 2)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        text_col = df.columns[-1]
        cat_cols = df.columns[:-1].tolist()

        with st.spinner('Procesando...'):
            # 1. Limpieza
            df['tokens'] = df[text_col].apply(lambda x: clean_text(x, lang_opt, custom_stopwords_list))
            
            # 2. Filtro Frecuencia
            all_tokens_raw = [t for sub in df['tokens'] for t in sub]
            freq_raw = Counter(all_tokens_raw)
            valid_words = set(w for w, c in freq_raw.items() if c >= min_freq_filter)
            
            df['tokens'] = df['tokens'].apply(lambda tokens: [t for t in tokens if t in valid_words])
            df['str_processed'] = df['tokens'].apply(lambda x: ' '.join(x))
            df['polaridad'] = df[text_col].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
            
            all_tokens = [token for sublist in df['tokens'] for token in sublist]

        if len(all_tokens) == 0:
            st.error("Filtros demasiado estrictos. No hay palabras.")
        else:
            # --- PESTAÑAS ---
            tab1, tab2, tab3, tab4 = st.tabs(["Frecuencia & Semántica", "KWIC (Concordancias)", "Redes", "Sentimientos"])

            # PREPARACIÓN DE DATOS SEMÁNTICOS (GLOBAL PARA TAB 1)
            freq_dist = Counter(all_tokens)
            top_n = 40
            common_words = freq_dist.most_common(top_n)
            df_freq = pd.DataFrame(common_words, columns=['Término', 'Frecuencia'])
            
            # Clustering Semántico
            if len(df_freq) > 5:
                vectorizer = TfidfVectorizer(vocabulary=df_freq['Término'].values)
                X = vectorizer.fit_transform(df['str_processed'])
                kmeans = KMeans(n_clusters=min(5, len(df_freq)), random_state=42, n_init=10)
                clusters = kmeans.fit_predict(X.T)
                word_to_cluster = dict(zip(vectorizer.get_feature_names_out(), clusters))
                df_freq['Grupo'] = df_freq['Término'].map(word_to_cluster).astype(str)
            else:
                df_freq['Grupo'] = '0'
                word_to_cluster = {w: '0' for w in df_freq['Término']}

            # --- MAPEO DE COLORES CONSISTENTE ---
            # Definimos una paleta fija de Plotly
            palette = px.colors.qualitative.Bold 
            # Creamos diccionario: Grupo '0' -> Color[0], Grupo '1' -> Color[1]...
            unique_groups = sorted(df_freq['Grupo'].unique())
            group_color_map = {grp: palette[i % len(palette)] for i, grp in enumerate(unique_groups)}
            
            # Diccionario Palabra -> Color (para la Nube)
            word_color_map = {row['Término']: group_color_map[row['Grupo']] for _, row in df_freq.iterrows()}

            # Función de color para WordCloud
            def similar_color_func(word, **kwargs):
                return word_color_map.get(word, 'gray') # Si no está en el top, gris

            # --- TAB 1: VISUALIZACIÓN ---
            with tab1:
                col_left, col_right = st.columns([1, 1])
                
                with col_left:
                    st.subheader("Top Términos por Grupo Semántico")
                    st.caption("Selecciona una barra para ver sus concordancias.")
                    
                    fig_bar = px.bar(
                        df_freq, 
                        x='Frecuencia', 
                        y='Término', 
                        orientation='h', 
                        color='Grupo',
                        text='Frecuencia',
                        color_discrete_map=group_color_map # Forzar mismos colores
                    )
                    fig_bar.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False, height=600)
                    
                    # --- INTERACTIVIDAD: CAPTURAR CLIC ---
                    event = st.plotly_chart(fig_bar, use_container_width=True, on_select="rerun")
                    
                    # Si el usuario hizo clic, guardamos la palabra
                    if event and len(event['selection']['points']) > 0:
                        selected_point = event['selection']['points'][0]
                        # Plotly devuelve el índice o el valor de y
                        clicked_word = selected_point['y']
                        st.session_state['selected_word'] = clicked_word
                        st.toast(f"Palabra seleccionada: {clicked_word}") # Notificación visual pequeña

                with col_right:
                    st.subheader("Nube Semántica")
                    if len(all_tokens) > 0:
                        wc = WordCloud(
                            width=800, height=800, 
                            background_color='white', 
                            max_words=top_n,
                            color_func=similar_color_func # <--- AQUI APLICAMOS LOS MISMOS COLORES
                        ).generate_from_frequencies(dict(common_words))
                        
                        fig_wc, ax = plt.subplots(figsize=(8,8))
                        ax.imshow(wc, interpolation='bilinear')
                        ax.axis('off')
                        st.pyplot(fig_wc)

            # --- TAB 2: KWIC INTERACTIVO ---
            with tab2:
                st.header("Análisis de Concordancia (KWIC)")
                
                # Input busca en session_state si hay algo seleccionado
                default_search = st.session_state['selected_word']
                
                col_search, col_reset = st.columns([4,1])
                search_term = col_search.text_input("Palabra a analizar:", value=default_search)
                
                if col_reset.button("Limpiar selección"):
                    st.session_state['selected_word'] = ""
                    st.rerun()

                if search_term:
                    st.markdown(f"Mostrando contextos para: **{search_term}**")
                    # Filtrado
                    mask = df['str_processed'].str.contains(search_term, case=False, na=False)
                    resul = df[mask]
                    
                    if len(resul) > 0:
                        st.info(f"Se encontraron {len(resul)} apariciones.")
                        # Mostramos tabla formateada
                        st.dataframe(
                            resul[[cat_cols[0], text_col]], 
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        st.warning("No se encontraron coincidencias exactas con los filtros actuales.")
                else:
                    st.info("Haz clic en una barra del gráfico anterior o escribe una palabra.")

            # --- TAB 3: REDES ---
            with tab3:
                st.subheader("Red de Co-ocurrencia")
                # Red simple rápida
                vectorizer_net = CountVectorizer(max_features=30, stop_words=stopwords.words(lang_opt))
                try:
                    X_net = vectorizer_net.fit_transform(df['str_processed'])
                    adj = (X_net.T * X_net)
                    adj.setdiag(0)
                    df_cooc = pd.DataFrame(adj.toarray(), index=vectorizer_net.get_feature_names_out(), columns=vectorizer_net.get_feature_names_out())
                    G = nx.from_pandas_adjacency(df_cooc)
                    
                    # Eliminar conexiones débiles
                    edges_del = [(u,v) for u,v,d in G.edges(data=True) if d['weight'] < 2]
                    G.remove_edges_from(edges_del)
                    G.remove_nodes_from(list(nx.isolates(G)))
                    
                    fig_net, ax_net = plt.subplots(figsize=(10,6))
                    pos = nx.spring_layout(G, k=0.5)
                    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='#ccc', node_size=1500, font_size=10, ax=ax_net)
                    st.pyplot(fig_net)
                except:
                    st.write("Insuficientes datos para red.")

            # --- TAB 4: SENTIMIENTOS ---
            with tab4:
                st.subheader("Análisis de Polaridad")
                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    fig_hist = px.histogram(df, x='polaridad', nbins=20, title="Distribución Global")
                    st.plotly_chart(fig_hist, use_container_width=True)
                with col_s2:
                    cat_select = st.selectbox("Cruzar con variable:", cat_cols)
                    fig_box = px.box(df, x=cat_select, y='polaridad', color=cat_select)
                    st.plotly_chart(fig_box, use_container_width=True)

    except Exception as e:
        st.error(f"Error procesando archivo: {e}")

else:
    st.info("Esperando archivo CSV en la barra lateral...")
