import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
from wordcloud import WordCloud
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Lexicométrico", layout="wide")

# --- ESTILOS CSS (Márgenes y Fuentes) ---
st.markdown("""
    <style>
    .block-container {padding-top: 1rem; padding-bottom: 5rem;}
    .stDataFrame {font-size: 1.1rem;}
    /* Ocultar menú hamburguesa por defecto para limpieza */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- GESTIÓN DE ESTADO ---
if 'selected_word' not in st.session_state:
    st.session_state['selected_word'] = None

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
    # Mapeo simple de idioma para nltk
    lang_map = {'Español': 'spanish', 'Inglés': 'english'}
    stop_words = set(stopwords.words(lang_map.get(language, 'spanish')))
    stop_words.update(set(custom_stops))
    tokens = word_tokenize(str(text).lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words and len(word) >= min_len]
    return tokens

# --- INTERFAZ PRINCIPAL ---

st.title("📊 Lexicométrico")

# BARRA LATERAL
st.sidebar.header("1. Datos y Filtros")
# Nota: El texto "Drag and drop" es del navegador, pero el label lo ponemos en español
uploaded_file = st.sidebar.file_uploader("Cargar archivo CSV (Matriz de datos)", type=["csv"])
lang_opt = st.sidebar.selectbox("Idioma del texto", ["Español", "Inglés"])
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
            st.error("No hay palabras suficientes con los filtros actuales. Reduce la frecuencia mínima.")
        else:
            # --- PREPARACIÓN DE DATOS ---
            freq_dist = Counter(all_tokens)
            top_n = 40
            common_words = freq_dist.most_common(top_n)
            df_freq = pd.DataFrame(common_words, columns=['Término', 'Frecuencia'])
            
            # Clustering Semántico (KMeans)
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

            # Colores Consistentes
            palette = px.colors.qualitative.Bold 
            unique_groups = sorted(df_freq['Grupo'].unique())
            group_color_map = {grp: palette[i % len(palette)] for i, grp in enumerate(unique_groups)}
            # Diccionario rápido palabra -> color
            word_color_map = {row['Término']: group_color_map[row['Grupo']] for _, row in df_freq.iterrows()}

            # --- PESTAÑAS ---
            tab1, tab2, tab3 = st.tabs(["📊 Frecuencia & KWIC", "🕸️ Redes", "❤️ Sentimientos"])

            # --- PESTAÑA 1: VISUALIZACIÓN ---
            with tab1:
                col_left, col_right = st.columns([1, 1])
                
                # --- A. GRÁFICO DE BARRAS ---
                with col_left:
                    st.subheader("Glosario de términos más utilizados")
                    st.caption("Haz clic en una barra para filtrar el contexto.")
                    
                    fig_bar = px.bar(
                        df_freq, x='Frecuencia', y='Término', orientation='h', 
                        color='Grupo', text='Frecuencia', color_discrete_map=group_color_map
                    )
                    
                    fig_bar.update_layout(
                        yaxis=dict(
                            categoryorder='total ascending',
                            tickfont=dict(size=18, color='black', family="Arial Black")
                        ),
                        xaxis=dict(showticklabels=False),
                        showlegend=False, 
                        height=650,
                        margin=dict(l=0, r=0, t=0, b=0)
                    )
                    fig_bar.update_traces(
                        textposition='outside',
                        textfont_size=20,
                        cliponaxis=False,
                        width=0.7
                    )
                    
                    # Interacción Barras
                    event_bar = st.plotly_chart(fig_bar, use_container_width=True, on_select="rerun", key="bar_chart")
                    
                    # Lógica de Captura: Prioridad al último clic
                    if event_bar and event_bar['selection']['points']:
                        candidate_word = event_bar['selection']['points'][0]['y']
                        if candidate_word != st.session_state['selected_word']:
                            st.session_state['selected_word'] = candidate_word
                            # No usamos st.rerun() aquí para evitar bucles, el estado ya cambió

                # --- B. NUBE INTERACTIVA ---
                with col_right:
                    st.subheader("Nube semántica")
                    
                    # 1. Calcular posiciones (Backend WordCloud)
                    wc = WordCloud(width=600, height=650, max_words=top_n).generate_from_frequencies(dict(common_words))
                    
                    # 2. Convertir a Datos para Plotly
                    word_list = []
                    for (word, freq), font_size, position, orientation, color in wc.layout_:
                        grp = word_to_cluster.get(word, '0')
                        color_hex = group_color_map.get(grp, '#888888') # Recuperamos el color del grupo
                        
                        word_list.append({
                            'word': word, 
                            'x': position[1], 
                            'y': -position[0], # Invertir Y
                            'size': font_size, 
                            'freq': freq, 
                            'color': color_hex
                        })
                    
                    df_cloud = pd.DataFrame(word_list)
                    
                    # 3. Dibujar Scatter Plot (Texto Interactuable)
                    fig_cloud = go.Figure()
                    
                    fig_cloud.add_trace(go.Scatter(
                        x=df_cloud['x'],
                        y=df_cloud['y'],
                        mode='text',
                        text=df_cloud['word'],
                        textfont=dict(
                            size=df_cloud['size'] * 0.9, # Ajuste de escala visual
                            color=df_cloud['color'],      # <--- AQUI SE APLICA EL COLOR CORRECTO
                            family="Arial Black"
                        ),
                        hoverinfo='text',
                        hovertext=[f"{w}: {f}" for w, f in zip(df_cloud['word'], df_cloud['freq'])]
                    ))
                    
                    fig_cloud.update_layout(
                        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, visible=False),
                        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, visible=False),
                        hovermode='closest',
                        plot_bgcolor='white',
                        height=650,
                        margin=dict(l=0, r=0, t=0, b=0),
                        dragmode=False
                    )

                    # Interacción Nube
                    event_cloud = st.plotly_chart(fig_cloud, use_container_width=True, on_select="rerun", key="cloud_chart")
                    
                    # Lógica de Captura Nube
                    if event_cloud and event_cloud['selection']['points']:
                        idx = event_cloud['selection']['points'][0]['point_index']
                        # Verificar que el índice sea válido
                        if idx < len(df_cloud):
                            candidate_word_cloud = df_cloud.iloc[idx]['word']
                            if candidate_word_cloud != st.session_state['selected_word']:
                                st.session_state['selected_word'] = candidate_word_cloud

                # --- C. SECCIÓN KWIC ---
                st.markdown("---")
                st.markdown("### 📝 Análisis de Contexto (KWIC)")
                
                if st.session_state['selected_word']:
                    word = st.session_state['selected_word']
                    
                    # Panel visual de la palabra seleccionada
                    st.markdown(f"""
                    <div style="background-color:#e6f3ff; padding:15px; border-radius:10px; margin-bottom:20px; border-left: 5px solid #2980b9;">
                        <h4 style="margin:0; color:#2c3e50;">
                            Resultados para el término: <span style="font-size:1.3em; color:#d35400;">{word}</span>
                        </h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    mask = df['str_processed'].str.contains(word, case=False, na=False)
                    resul = df[mask]
                    
                    if len(resul) > 0:
                        st.dataframe(
                            resul[[cat_cols[0], text_col]], 
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        st.warning("No se encontraron coincidencias exactas.")
                else:
                    st.info("👈 Haz clic en una palabra del gráfico de barras o de la nube para ver aquí su contexto de uso.")

                # MARGEN INFERIOR
                st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)

            # --- PESTAÑA 2: REDES ---
            with tab2:
                st.subheader("Red de Co-ocurrencia")
                vectorizer_net = CountVectorizer(max_features=40, stop_words=stopwords.words('spanish' if lang_opt=='Español' else 'english'))
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
                    pos = nx.spring_layout(G, k=0.6)
                    nx.draw(G, pos, with_labels=True, node_color='#aaddff', edge_color='#cccccc', node_size=1200, font_size=11, ax=ax_net)
                    st.pyplot(fig_net)
                except Exception as e:
                    st.warning(f"No hay suficientes datos para generar la red: {e}")
                st.markdown("<br><br><br>", unsafe_allow_html=True)

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
                st.markdown("<br><br><br>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error procesando el archivo: {e}")
