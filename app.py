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
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans

# Configuración de página
st.set_page_config(page_title="LexicoMapper Pro", layout="wide")

# --- CONFIGURACIÓN NLTK ---
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')

# --- FUNCIONES AUXILIARES ---

def clean_text(text, language='spanish', custom_stops=[], min_len=2):
    if pd.isna(text): return []
    # Stopwords base
    stop_words = set(stopwords.words(language))
    # Añadir stopwords del usuario
    stop_words.update(set(custom_stops))
    
    tokens = word_tokenize(str(text).lower())
    # Filtro: alfabético, no stopword, longitud mínima
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words and len(word) >= min_len]
    return tokens

def get_lexical_stats(tokens):
    n_tokens = len(tokens)
    n_types = len(set(tokens))
    ttr = (n_types / n_tokens) * 100 if n_tokens > 0 else 0
    freqs = Counter(tokens)
    hapax = sum(1 for word, count in freqs.items() if count == 1)
    hapax_ratio = (hapax / n_tokens) * 100 if n_tokens > 0 else 0
    return n_tokens, n_types, ttr, hapax, hapax_ratio

# --- INTERFAZ PRINCIPAL ---

st.title("📊 LexicoMapper Pro")
st.markdown("Análisis avanzado con filtros dinámicos y detección de grupos semánticos.")

# BARRA LATERAL (CONTROLES)
st.sidebar.header("1. Carga de Datos")
uploaded_file = st.sidebar.file_uploader("Subir CSV", type=["csv"])
lang_opt = st.sidebar.selectbox("Idioma", ["spanish", "english"])

st.sidebar.divider()
st.sidebar.header("2. Filtros de Limpieza")
# Filtro de palabras a excluir
custom_stopwords_input = st.sidebar.text_area("Excluir palabras (separar por coma):", placeholder="ej: respuesta, ns, nc")
custom_stopwords_list = [x.strip().lower() for x in custom_stopwords_input.split(',')] if custom_stopwords_input else []

# Filtro de Frecuencia Mínima
min_freq_filter = st.sidebar.slider("Frecuencia mínima de palabra para análisis:", 1, 50, 1)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Identificación de variables
        text_col = df.columns[-1]
        cat_cols = df.columns[:-1].tolist()

        # Procesamiento Inicial
        with st.spinner('Procesando corpus y aplicando filtros...'):
            # 1. Tokenización básica
            df['tokens'] = df[text_col].apply(lambda x: clean_text(x, lang_opt, custom_stopwords_list))
            
            # 2. Filtro Global de Frecuencia
            # Contamos todo primero
            all_tokens_raw = [t for sub in df['tokens'] for t in sub]
            freq_raw = Counter(all_tokens_raw)
            # Creamos set de palabras permitidas
            valid_words = set(w for w, c in freq_raw.items() if c >= min_freq_filter)
            
            # 3. Refiltrar los tokens del dataframe
            df['tokens'] = df['tokens'].apply(lambda tokens: [t for t in tokens if t in valid_words])
            df['str_processed'] = df['tokens'].apply(lambda x: ' '.join(x))
            
            # Métricas extra
            df['longitud'] = df['tokens'].apply(len)
            df['polaridad'] = df[text_col].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
            
            # Lista final aplanada para gráficos
            all_tokens = [token for sublist in df['tokens'] for token in sublist]

        if len(all_tokens) == 0:
            st.error("Los filtros aplicados han eliminado todas las palabras. Reduce la frecuencia mínima.")
        else:
            # --- PESTAÑAS ---
            tab1, tab2, tab3, tab4 = st.tabs(["☁️ Frecuencia & Semántica", "📈 Estadística General", "🕸️ Redes", "❤️ Sentimientos"])

            # 1. FRECUENCIA Y SEMÁNTICA (LO QUE PEDISTE)
            with tab1:
                col_left, col_right = st.columns([1, 1])
                
                # Preparar datos para Top Términos
                freq_dist = Counter(all_tokens)
                top_n = 30
                common_words = freq_dist.most_common(top_n)
                df_freq = pd.DataFrame(common_words, columns=['Término', 'Frecuencia'])
                
                # --- LÓGICA DE CLUSTERING (CAMPOS SEMÁNTICOS SIMULADOS) ---
                if len(df_freq) > 5:
                    # Usamos TF-IDF transpuesta para ver qué palabras aparecen en los mismos documentos
                    vectorizer = TfidfVectorizer(vocabulary=df_freq['Término'].values)
                    X = vectorizer.fit_transform(df['str_processed'])
                    # Transponemos: Filas=Palabras, Columnas=Documentos
                    word_vectors = X.T 
                    
                    # KMeans para agrupar palabras
                    num_clusters = min(5, len(df_freq)) # Máximo 5 grupos
                    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
                    clusters = kmeans.fit_predict(word_vectors)
                    
                    # Asignamos el cluster al dataframe
                    # Nota: KMeans respeta el orden del vocabulario del vectorizer
                    word_to_cluster = dict(zip(vectorizer.get_feature_names_out(), clusters))
                    df_freq['Grupo Semántico'] = df_freq['Término'].map(word_to_cluster).astype(str)
                else:
                    df_freq['Grupo Semántico'] = '1'

                with col_left:
                    st.subheader("Top Términos (Agrupados)")
                    st.caption("Los colores indican palabras que suelen aparecer en contextos similares.")
                    
                    # Gráfico de Barras Mejorado
                    fig_bar = px.bar(
                        df_freq, 
                        x='Frecuencia', 
                        y='Término', 
                        orientation='h', 
                        color='Grupo Semántico', # Colores por cluster
                        text='Frecuencia',       # Etiqueta numérica al final
                        color_discrete_sequence=px.colors.qualitative.Bold
                    )
                    fig_bar.update_traces(textposition='outside') # Número fuera de la barra
                    fig_bar.update_layout(
                        yaxis={'categoryorder':'total ascending'}, 
                        showlegend=False,
                        height=600
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                with col_right:
                    st.subheader("Nube de Palabras")
                    if len(all_tokens) > 0:
                        # Nube
                        wc = WordCloud(
                            width=800, 
                            height=800, 
                            background_color='white', 
                            max_words=100,
                            colormap='viridis'
                        ).generate_from_frequencies(freq_dist)
                        
                        fig_wc, ax = plt.subplots(figsize=(8,8))
                        ax.imshow(wc, interpolation='bilinear')
                        ax.axis('off')
                        st.pyplot(fig_wc)
                    else:
                        st.warning("No hay suficientes datos para la nube.")

            # 2. ESTADÍSTICA
            with tab2:
                total, types, ttr, hapax, hapax_ratio = get_lexical_stats(all_tokens)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Tokens", total)
                c2.metric("Vocabulario", types)
                c3.metric("Riqueza (TTR)", f"{ttr:.2f}%")
                c4.metric("Hapax", f"{hapax}")
                
                st.divider()
                st.subheader("KWIC (Concordancias)")
                search_kwic = st.text_input("Buscar palabra en contexto:", "")
                if search_kwic:
                    # Búsqueda simple
                    resul = df[df['str_processed'].str.contains(search_kwic, case=False, na=False)]
                    st.write(f"Encontrado en {len(resul)} respuestas.")
                    st.dataframe(resul[[cat_cols[0], 'str_processed']].head(20), use_container_width=True)

            # 3. REDES
            with tab3:
                st.subheader("Red Semántica")
                # Lógica de red simplificada
                top_nodes = 40
                vectorizer_net = CountVectorizer(max_features=top_nodes, stop_words=stopwords.words(lang_opt))
                try:
                    X_net = vectorizer_net.fit_transform(df['str_processed'])
                    adj_matrix = (X_net.T * X_net)
                    adj_matrix.setdiag(0)
                    
                    df_cooc = pd.DataFrame(adj_matrix.toarray(), index=vectorizer_net.get_feature_names_out(), columns=vectorizer_net.get_feature_names_out())
                    G = nx.from_pandas_adjacency(df_cooc)
                    
                    # Limpieza de aristas débiles para visualización
                    threshold = st.slider("Umbral de conexión (co-ocurrencias)", 2, 20, 3)
                    edges_to_del = [(u,v) for u,v,d in G.edges(data=True) if d['weight'] < threshold]
                    G.remove_edges_from(edges_to_del)
                    G.remove_nodes_from(list(nx.isolates(G)))
                    
                    if G.number_of_nodes() > 0:
                        pos = nx.spring_layout(G, k=0.6)
                        # Dibujado rápido con matplotlib para compatibilidad robusta
                        fig_net, ax_net = plt.subplots(figsize=(10,6))
                        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=1000, font_size=9, ax=ax_net)
                        st.pyplot(fig_net)
                    else:
                        st.info("Sube el volumen de datos o baja el umbral para ver la red.")
                except ValueError:
                    st.error("Datos insuficientes para generar la red.")

            # 4. SENTIMIENTOS
            with tab4:
                st.subheader("Distribución de Polaridad")
                fig_hist = px.histogram(df, x='polaridad', nbins=20, color_discrete_sequence=['teal'])
                st.plotly_chart(fig_hist, use_container_width=True)
                
                cat_select = st.selectbox("Cruzar sentimientos con:", cat_cols)
                fig_box = px.box(df, x=cat_select, y='polaridad', color=cat_select)
                st.plotly_chart(fig_box, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")
