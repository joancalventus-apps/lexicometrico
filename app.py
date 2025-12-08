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

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="LexicoMapper Pro", layout="wide")

# --- ESTILOS CSS ---
# Ajustes para márgenes y tamaño de fuente extra en tablas
st.markdown("""
    <style>
    .block-container {padding-top: 1rem; padding-bottom: 5rem;}
    .stDataFrame {font-size: 1.2rem;}
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
            # --- DATOS ---
            freq_dist = Counter(all_tokens)
            top_n = 40
            common_words = freq_dist.most_common(top_n)
            df_freq = pd.DataFrame(common_words, columns=['Término', 'Frecuencia'])
            
            # Clustering Semántico
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

            # Colores
            palette = px.colors.qualitative.Bold 
            unique_groups = sorted(df_freq['Grupo'].unique())
            group_color_map = {grp: palette[i % len(palette)] for i, grp in enumerate(unique_groups)}
            
            # --- PESTAÑAS ---
            tab1, tab2, tab3 = st.tabs(["📊 Frecuencia & KWIC", "🕸️ Redes", "❤️ Sentimientos"])

            # --- PESTAÑA 1 ---
            with tab1:
                col_left, col_right = st.columns([1, 1])
                
                # --- GRÁFICO DE BARRAS (IZQUIERDA) ---
                with col_left:
                    st.subheader("Top Términos")
                    st.caption("Clic en la barra para filtrar.")
                    
                    fig_bar = px.bar(
                        df_freq, x='Frecuencia', y='Término', orientation='h', 
                        color='Grupo', text='Frecuencia', color_discrete_map=group_color_map
                    )
                    
                    # AUMENTO MASIVO DE FUENTE
                    fig_bar.update_layout(
                        yaxis=dict(
                            categoryorder='total ascending',
                            tickfont=dict(size=18, color='black', family="Arial Black") # Eje Y negrita grande
                        ),
                        xaxis=dict(showticklabels=False),
                        showlegend=False, 
                        height=600,
                        margin=dict(l=0, r=0, t=0, b=0)
                    )
                    fig_bar.update_traces(
                        textposition='outside',
                        textfont_size=20, # Número de frecuencia gigante
                        cliponaxis=False,
                        width=0.7
                    )
                    
                    # Interacción Barras
                    event_bar = st.plotly_chart(fig_bar, use_container_width=True, on_select="rerun", key="bar_chart")
                    if event_bar and len(event_bar['selection']['points']) > 0:
                        st.session_state['selected_word'] = event_bar['selection']['points'][0]['y']

                # --- NUBE DE PALABRAS INTERACTIVA (DERECHA) ---
                with col_right:
                    st.subheader("Nube Semántica (Clicable)")
                    st.caption("Clic en una palabra de la nube para filtrar.")
                    
                    # 1. Calculamos posiciones usando WordCloud (sin dibujar)
                    wc = WordCloud(width=600, height=600, max_words=top_n).generate_from_frequencies(dict(common_words))
                    
                    # 2. Extraemos coordenadas y creamos un DataFrame para Plotly
                    word_list = []
                    # wc.layout_ contiene: ((word, freq), font_size, position, orientation, color)
                    for (word, freq), font_size, position, orientation, color in wc.layout_:
                        # Ajustamos coordenadas para centrar
                        x_pos = position[1]
                        y_pos = -position[0] # Invertir eje Y porque las imágenes van de arriba a abajo
                        
                        # Asignamos color según grupo semántico
                        grp = word_to_cluster.get(word, '0')
                        color_code = group_color_map.get(grp, '#888')
                        
                        word_list.append({
                            'word': word, 'x': x_pos, 'y': y_pos, 
                            'size': font_size, 'freq': freq, 'color': color_code
                        })
                    
                    df_cloud_plot = pd.DataFrame(word_list)
                    
                    # 3. Dibujamos con Plotly Scatter (Texto)
                    fig_cloud = go.Figure()
                    
                    # Añadimos las palabras como texto en coordenadas específicas
                    fig_cloud.add_trace(go.Scatter(
                        x=df_cloud_plot['x'],
                        y=df_cloud_plot['y'],
                        mode='text',
                        text=df_cloud_plot['word'],
                        textfont=dict(
                            size=df_cloud_plot['size'] * 0.8, # Factor de escala visual
                            color=df_cloud_plot['color'],
                            family="Arial Black"
                        ),
                        hoverinfo='text',
                        hovertext=[f"{w}: {f}" for w, f in zip(df_cloud_plot['word'], df_cloud_plot['freq'])]
                    ))
                    
                    fig_cloud.update_layout(
                        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
                        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
                        hovermode='closest',
                        plot_bgcolor='white',
                        height=600,
                        margin=dict(l=0, r=0, t=0, b=0),
                        dragmode=False # Desactivar zoom para evitar movimientos raros
                    )

                    # Interacción Nube
                    event_cloud = st.plotly_chart(fig_cloud, use_container_width=True, on_select="rerun", key="cloud_chart")
                    
                    # Detectar clic en la nube
                    if event_cloud and len(event_cloud['selection']['points']) > 0:
                        # En scatter trace, 'text' contiene la palabra
                        idx = event_cloud['selection']['points'][0]['point_index']
                        clicked_word_cloud = df_cloud_plot.iloc[idx]['word']
                        st.session_state['selected_word'] = clicked_word_cloud

                # --- SECCIÓN KWIC ---
                st.divider()
                st.markdown("### 📝 Análisis de Contexto (KWIC)")
                
                if st.session_state['selected_word']:
                    word = st.session_state['selected_word']
                    # Usamos HTML para resaltar grande la palabra seleccionada
                    st.markdown(f"""
                    <div style="background-color:#f0f2f6; padding:15px; border-radius:10px; margin-bottom:20px;">
                        <h3 style="margin:0; color:#31333F;">
                            Concordancias para: <span style="color:#ff4b4b; text-decoration:underline;">{word}</span>
                        </h3>
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
                    st.info("👈 Haz clic en una BARRA o en una PALABRA DE LA NUBE para ver los resultados aquí.")

                # ESPACIO FINAL EXTRA (MARGEN)
                st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)

            # --- PESTAÑA 2: REDES ---
            with tab2:
                st.subheader("Red de Co-ocurrencia")
                vectorizer_net = CountVectorizer(max_features=40, stop_words=stopwords.words(lang_opt))
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
                    st.warning(f"No hay suficientes datos: {e}")
                st.markdown("<br><br><br>", unsafe_allow_html=True)

            # --- PESTAÑA 3: SENTIMIENTOS ---
            with tab3:
                c1, c2 = st.columns(2)
                with c1:
                    fig_h = px.histogram(df, x='polaridad', nbins=20, title="Distribución", color_discrete_sequence=['teal'])
                    st.plotly_chart(fig_h, use_container_width=True)
                with c2:
                    cat_sel = st.selectbox("Variable Categórica:", cat_cols)
                    fig_b = px.box(df, x=cat_sel, y='polaridad', color=cat_sel)
                    st.plotly_chart(fig_b, use_container_width=True)
                st.markdown("<br><br><br>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error general: {e}")
