import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from textblob import TextBlob
from wordcloud import WordCloud
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

# Configuración de página
st.set_page_config(page_title="LexicoMapper", layout="wide")

# --- CONFIGURACIÓN NLTK (Corrección del error punkt_tab) ---
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')

# --- FUNCIONES AUXILIARES ---

def clean_text(text, language='spanish'):
    if pd.isna(text): return ""
    # Stopwords
    stop_words = set(stopwords.words(language))
    # Tokenización simple y limpieza
    tokens = word_tokenize(str(text).lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return tokens

def get_lexical_stats(tokens):
    n_tokens = len(tokens)
    n_types = len(set(tokens))
    # Riqueza léxica (Type-Token Ratio)
    ttr = (n_types / n_tokens) * 100 if n_tokens > 0 else 0
    # Hapax Legomena (palabras que aparecen una sola vez)
    freqs = Counter(tokens)
    hapax = sum(1 for word, count in freqs.items() if count == 1)
    hapax_ratio = (hapax / n_tokens) * 100 if n_tokens > 0 else 0
    return n_tokens, n_types, ttr, hapax, hapax_ratio

# --- INTERFAZ PRINCIPAL ---

st.title("📊 LexicoMapper: Análisis Lexicométrico y Cartográfico")
st.markdown("""
Esta herramienta realiza análisis estadístico textual sobre un corpus. 
Sube tu archivo CSV donde las primeras columnas sean variables categóricas y la última el texto libre.
""")

# 1. Carga de Datos
uploaded_file = st.sidebar.file_uploader("Cargar archivo CSV", type=["csv"])
lang_opt = st.sidebar.selectbox("Idioma del corpus", ["spanish", "english"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"Archivo cargado con éxito. Filas: {df.shape[0]}, Columnas: {df.shape[1]}")
        
        # Identificación de variables
        text_col = df.columns[-1] # Asumimos la última como texto
        cat_cols = df.columns[:-1].tolist() # El resto son categóricas
        
        st.write(f"**Variable Textual detectada:** {text_col}")
        st.write(f"**Variables Categóricas:** {', '.join(cat_cols)}")

        # Procesamiento
        with st.spinner('Procesando textos...'):
            df['tokens'] = df[text_col].apply(lambda x: clean_text(x, lang_opt))
            df['str_processed'] = df['tokens'].apply(lambda x: ' '.join(x))
            
            # Métricas por fila
            df['longitud'] = df['tokens'].apply(len)
            df['polaridad'] = df[text_col].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

        # --- PESTAÑAS DE ANÁLISIS ---
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Estadística Textual", 
            "☁️ Nube & Frecuencia", 
            "🕸️ Redes & Cartografía", 
            "❤️ Sentimientos",
            "🔍 KWIC & Concordancia"
        ])

        # 1. ESTADÍSTICA TEXTUAL
        with tab1:
            st.header("Estadísticas Globales")
            all_tokens = [token for sublist in df['tokens'] for token in sublist]
            total, types, ttr, hapax, hapax_ratio = get_lexical_stats(all_tokens)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Palabras (Tokens)", total)
            col2.metric("Vocabulario (Types)", types)
            col3.metric("Riqueza Léxica (TTR)", f"{ttr:.2f}%")
            col4.metric("Hapax Legomena", f"{hapax} ({hapax_ratio:.1f}%)")

            st.subheader("Distribución de Longitud de Respuestas")
            fig_hist = px.histogram(df, x='longitud', nbins=20, title="Frecuencia por longitud de respuesta")
            st.plotly_chart(fig_hist, use_container_width=True)

        # 2. NUBE Y FRECUENCIA
        with tab2:
            st.header("Análisis de Frecuencia")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.subheader("Nube de Palabras")
                wc = WordCloud(width=800, height=400, background_color='white').generate(' '.join(all_tokens))
                fig_wc, ax = plt.subplots()
                ax.imshow(wc, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig_wc)
            
            with col_b:
                st.subheader("Top 20 Términos")
                freq_dist = Counter(all_tokens)
                df_freq = pd.DataFrame(freq_dist.most_common(20), columns=['Término', 'Frecuencia'])
                fig_bar = px.bar(df_freq, x='Frecuencia', y='Término', orientation='h', title="Términos más frecuentes")
                fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_bar, use_container_width=True)

        # 3. REDES Y CARTOGRAFÍA (Mapas conceptuales)
        with tab3:
            st.header("Redes de Co-ocurrencia (Mapa Conceptual)")
            st.markdown("Este grafo muestra términos que aparecen juntos frecuentemente.")
            
            min_cooc = st.slider("Mínimo de co-ocurrencias para conexión", 2, 50, 5)
            top_n_nodes = st.slider("Número de palabras top a analizar", 10, 100, 30)
            
            # Construcción de matriz de co-ocurrencia
            vectorizer = CountVectorizer(max_features=top_n_nodes, stop_words=stopwords.words(lang_opt))
            X = vectorizer.fit_transform(df['str_processed'])
            co_occurrence_matrix = (X.T * X)
            co_occurrence_matrix.setdiag(0)
            
            names = vectorizer.get_feature_names_out()
            df_cooc = pd.DataFrame(co_occurrence_matrix.toarray(), index=names, columns=names)
            
            # Grafo con NetworkX
            G = nx.from_pandas_adjacency(df_cooc)
            
            # Filtrar por peso
            edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] < min_cooc]
            G.remove_edges_from(edges_to_remove)
            G.remove_nodes_from(list(nx.isolates(G)))
            
            if G.number_of_nodes() > 0:
                pos = nx.spring_layout(G, k=0.5)
                
                # Plotly para grafo interactivo
                edge_x = []
                edge_y = []
                for edge in G.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])

                edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')

                node_x = []
                node_y = []
                node_text = []
                node_adj = []
                for node in G.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    node_text.append(node)
                    node_adj.append(len(G.adj[node]))

                node_trace = go.Scatter(
                    x=node_x, y=node_y, mode='markers+text',
                    text=node_text, textposition="top center",
                    marker=dict(showscale=True, colorscale='YlGnBu', size=15, color=node_adj, line_width=2)
                )

                fig_net = go.Figure(data=[edge_trace, node_trace],
                                    layout=go.Layout(showlegend=False, hovermode='closest', margin=dict(b=0,l=0,r=0,t=0), xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
                st.plotly_chart(fig_net, use_container_width=True)
            else:
                st.warning("Aumenta el número de palabras o reduce el umbral de co-ocurrencia.")

            st.divider()
            st.subheader("Mapa de Calor: Variable Categórica vs Frecuencia de Términos")
            # Selección dinámica de variable
            cat_select = st.selectbox("Selecciona variable categórica para cruzar", cat_cols)
            
            # Correspondencia simplificada (Heatmap)
            # Creamos un df con las palabras top y la variable categórica
            top_words = df_freq['Término'].head(15).tolist()
            
            def count_top_words(txt_list):
                return {word: txt_list.count(word) for word in top_words}
            
            # Expandimos el conteo
            word_counts = df['tokens'].apply(count_top_words).apply(pd.Series).fillna(0)
            df_heatmap = pd.concat([df[cat_select], word_counts], axis=1)
            heatmap_data = df_heatmap.groupby(cat_select).sum()
            
            fig_heat = px.imshow(heatmap_data, text_auto=True, aspect="auto", color_continuous_scale='Viridis', title=f"Frecuencia de términos top por {cat_select}")
            st.plotly_chart(fig_heat, use_container_width=True)

        # 4. SENTIMIENTOS
        with tab4:
            st.header("Análisis de Sentimientos")
            st.info("Polaridad: -1 (Muy negativo) a +1 (Muy positivo). Basado en léxico (TextBlob).")
            
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                fig_sent = px.histogram(df, x='polaridad', nbins=20, title="Distribución de Polaridad", color_discrete_sequence=['indianred'])
                st.plotly_chart(fig_sent, use_container_width=True)
            
            with col_s2:
                cat_sent_select = st.selectbox("Comparar sentimiento por:", cat_cols, key='sent_cat')
                fig_box = px.box(df, x=cat_sent_select, y='polaridad', title=f"Sentimiento según {cat_sent_select}")
                st.plotly_chart(fig_box, use_container_width=True)

        # 5. KWIC
        with tab5:
            st.header("Key Word In Context (KWIC)")
            search_term = st.text_input("Buscar término:", "")
            
            if search_term:
                # Filtrar dataframe
                mask = df[text_col].str.contains(search_term, case=False, na=False)
                resul = df[mask]
                st.write(f"Se encontraron {len(resul)} coincidencias.")
                
                # Mostrar tabla simple con contexto
                st.dataframe(resul[[cat_cols[0], text_col]]) # Muestra primera var categórica y texto
            
    except Exception as e:
        st.error(f"Error al procesar el archivo: {e}")
        st.write("Asegúrate de que el archivo es CSV y tiene el formato correcto (Variables categóricas + 1 Variable texto al final).")

else:
    st.info("Esperando archivo CSV...")
