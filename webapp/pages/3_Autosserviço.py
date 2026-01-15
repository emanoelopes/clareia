import sys
from pathlib import Path

# Adicionar o diretório webapp ao path do Python
webapp_dir = Path(__file__).parent.parent
if str(webapp_dir) not in sys.path:
    sys.path.insert(0, str(webapp_dir))

import pandas as pd
import pygwalker as pyg
from pygwalker.api.streamlit import StreamlitRenderer
import streamlit as st
from src.openai_interpreter import criar_rodape_sidebar

st.set_page_config(
    page_title="Análise Exploratória - Autosserviço",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar com rodapé
with st.sidebar:
    st.markdown("### 📊 Sobre")
    st.info("""
    Esta página oferece uma ferramenta de análise interativa usando PygWalker, permitindo que você explore os dados de forma autônoma.
    """)
    st.markdown("---")
    # Rodapé com badges de status (igual ao da home)
    criar_rodape_sidebar()

st.markdown("""
# 🔍 Análise Exploratória Interativa

Esta página permite que você explore seus dados livremente usando o **PygWalker**, uma interface similar ao Tableau.

**Instruções:**
1. Arraste as variáveis da lista à esquerda para os eixos X e Y.
2. Use os filtros para refinar sua análise.
3. Clique em "Data" para visualizar a tabela bruta.
""")

# Determinar qual dataset usar
df_to_explore = None
source_name = ""

if 'user_data_uploaded' in st.session_state and st.session_state.user_data_uploaded is not None:
    df_to_explore = st.session_state.user_data_uploaded
    source_name = "Dados Enviados pelo Usuário (Home)"
elif 'df_uci' in st.session_state and st.session_state.df_uci is not None:
    df_to_explore = st.session_state['df_uci']
    source_name = "Dados UCI"
elif 'df_oulad' in st.session_state and st.session_state.df_oulad is not None: # Assumindo que OULAD salva assim
    df_to_explore = st.session_state['df_oulad']
    source_name = "Dados OULAD"

if df_to_explore is not None:
    st.success(f"✅ Analisando: **{source_name}**")
    
    # Inicializa o renderizador do Pygwalker com cache para performance
    @st.cache_resource
    def get_pyg_renderer(dataframe):
        return StreamlitRenderer(dataframe, spec="./gw_config.json", spec_io_mode="RW")
    
    try:
        renderer = get_pyg_renderer(df_to_explore)
        renderer.explorer()
    except Exception as e:
        st.error(f"Erro ao carregar Pygwalker: {e}")

else:
    st.warning("⚠️ Nenhum dado selecionado para análise.")
    st.info("""
    **Como carregar dados:**
    - Vá para a **Página Inicial** e faça upload do seu template.
    - Ou navegue pelas páginas **UCI** ou **OULAD** para carregar os datasets de exemplo.
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🏠 Ir para Home"):
            st.switch_page("home.py")
    with col2:
        if st.button("📊 Ir para UCI"):
            st.switch_page("pages/1_UCI.py")
    with col3:
         if st.button("🎓 Ir para OULAD"):
            st.switch_page("pages/2_OULAD.py")

