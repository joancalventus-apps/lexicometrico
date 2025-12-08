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

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Lexicométrico", layout="wide")

# Estilos CSS para limpieza y fuentes
st.markdown("""
    <style>
    .block-container {padding-top: 1rem; padding-bottom: 5rem;}
    .stDataFrame {font-size: 1.1rem;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- 2. GESTIÓN DE ESTADO ---
if 'selected_word' not in st.session_state:
    st.session_state['selected_word'] = None

# --- 3. CONFIGURACIÓN GLOBAL ---
LANG_MAP = {'Español': 'spanish', 'Inglés': 'english'}

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')

# --- 4. FUNCIONES ---
def clean_text(text, language='Español', custom_stops=[], min_len=2):
    if pd.isna(text): return []
    lang_code = LANG_MAP.get(language, 'spanish')
    stop_words = set(stopwords.words(lang_code))
    stop_words.update(set(custom_stops))
    tokens = word_tokenize(str(text).lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words and len(word) >= min_len]
    return tokens

# --- 5. INTERFAZ PRINCIPAL ---

st.title("📊 Lexicométrico")

# --- BARRA LATERAL ---
st.sidebar.header("1. Datos y Filtros")
uploaded_file = st.sidebar.file_uploader("Arrastrar y soltar archivo CSV aquí", type=["csv"])
lang_opt = st.sidebar.selectbox("Idioma del texto", ["Español", "Inglés"])

st.sidebar.markdown("---")
st.sidebar.header("2. Limpieza")
custom_stopwords_input = st.sidebar.text_area("Excluir palabras (separar por coma):", placeholder="ej: respuesta, ns, nc")
custom_stopwords_list = [x.strip().lower() for x in custom_stopwords_input.split(',')] if custom_stopwords_input else []
min_freq_filter = st.sidebar.slider("Frecuencia mínima de aparición:", 1, 50, 2)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        text_col = df.columns[-1]
        cat_cols = df.columns[:-1].tolist()

        with st.spinner('Procesando corpus textual...'):
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
            # --- CÁLCULOS ---
            freq_dist = Counter(all_tokens)
            top_n = 40
            common_words = freq_dist.most_common(top_n)
            df_freq = pd.DataFrame(common_words, columns=['Término', 'Frecuencia'])
            
            # --- SELECCIÓN POR DEFECTO ---
            # Si es el inicio, seleccionamos la palabra #1
            available_words = set(df_freq['Término'])
            if st.session_state['selected_word'] is None or st.session_state['selected_word'] not in available_words:
                if not df_freq.empty:
                    st.session_state['selected_word'] = df_freq.iloc[0]['Término']

            # --- CLUSTERING SEMÁNTICO ---
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

            # --- COLORES ---
            palette = px.colors.qualitative.Bold 
            unique_groups = sorted(df_freq['Grupo'].unique())
            group_color_map = {grp: palette[i % len(palette)] for i, grp in enumerate(unique_groups)}
            # Diccionario palabra -> color (para la Nube)
            word_color_map = {row['Término']: group_color_map[row['Grupo']] for _, row in df_freq.iterrows()}

            # Función para colorear la nube (Match exacto con barras)
            def color_func(word, **kwargs):
                return word_color_map.get(word, '#888888')

            # --- PESTAÑAS ---
            tab1, tab2, tab3 = st.tabs(["📊 Frecuencia & KWIC", "🕸️ Redes", "❤️ Sentimientos"])

            with tab1:
                col_left, col_right = st.columns([1.2, 0.8]) # Dar un poco más de espacio a la barra
                
                # --- A. BARRAS INTERACTIVAS ---
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
                    
                    # Interacción robusta
                    event_bar = st.plotly_chart(fig_bar, use_container_width=True, on_select="rerun", key="bar_chart")
                    if event_bar and event_bar['selection']['points']:
                        new_word = event_bar['selection']['points'][0]['y']
                        st.session_state['selected_word'] = new_word

                # --- B. NUBE VISUAL (Formato Script 1 - Perfecto) ---
                with col_right:
                    st.subheader("Nube Semántica")
                    st.markdown("*(Representación visual de campos semánticos)*")
                    
                    # Recuperamos el formato original de WordCloud (Matplotlib)
                    wc = WordCloud(
                        width=600, height=800, 
                        background_color='white', 
                        max_words=top_n,
                        color_func=color_func, # Aplicar mismos colores
                        prefer_horizontal=0.9,
                        relative_scaling=0.5
                    ).generate_from_frequencies(dict(common_words))
                    
                    fig_wc, ax = plt.subplots(figsize=(6,8))
                    ax.imshow(wc, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig_wc) # Renderizado estático de alta calidad

                # --- C. SECCIÓN KWIC ---
                st.markdown("---")
                st.markdown("### 📝 Análisis de Contexto (KWIC)")
                
                # Buscador manual opcional por si quieren buscar algo de la nube sin clicar la barra
                col_search, col_info = st.columns([1, 3])
                with col_search:
                     manual_search = st.text_input("🔍 O escribe una palabra aquí:", value="")
                
                # Lógica de prioridad: Si escriben, usa eso. Si no, usa el clic/default.
                current_word = manual_search if manual_search else st.session_state['selected_word']

                st.markdown(f"""
                <div style="background-color:#f0f2f6; padding:15px; border-radius:10px; border-left: 6px solid #e74c3c;">
                    <h4 style="margin:0; color:#2c3e50;">
                        Contextos para: <span style="color:#c0392b; font-size:1.3em;">{current_word}</span>
                    </h4>
                </div>
                """, unsafe_allow_html=True)
                
                if current_word:
                    mask = df['str_processed'].str.contains(current_word, case=False, na=False)
                    resul = df[mask]
                    
                    if len(resul) > 0:
                        st.dataframe(resul[[cat_cols[0], text_col]], use_container_width=True, hide_index=True)
                    else:
                        st.warning(f"No se encontraron coincidencias exactas para '{current_word}' con los filtros actuales.")

                st.markdown("<br><br>", unsafe_allow_html=True)

            # --- PESTAÑA 2: REDES ---
            with tab2:
                st.subheader("Red de Co-ocurrencia")
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
                    
                    fig_net, ax_net = plt.subplots(figsize=(12,8))
                    pos = nx.spring_layout(G, k=0.5, seed=42)
                    nx.draw(G, pos, with_labels=True, node_color='#aaddff', edge_color='#cccccc', node_size=1500, font_size=10, ax=ax_net)
                    st.pyplot(fig_net)
                except Exception as e:
                    st.warning(f"Se necesitan más datos para generar la red: {e}")
                st.markdown("<br><br>", unsafe_allow_html=True)

            # --- PESTAÑA 3: SENTIMIENTOS ---
            with tab3:
                c1, c2 = st.columns(2)
                with c1:
                    fig_h = px.histogram(df, x='polaridad', nbins=20, title="Distribución de Polaridad", color_discrete_sequence=['teal'])
                    st.plotly_chart(fig_h, use_container_width=True)
                with c2:
                    cat_sel = st.selectbox("Cruzar con variable:", cat_cols)
                    fig_b = px.box(df, x=cat_sel, y='polaridad', color=cat_sel, title="Polaridad por Categoría")
                    st.plotly_chart(fig_b, use_container_width=True)
                st.markdown("<br><br>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error procesando el archivo: {e}")
