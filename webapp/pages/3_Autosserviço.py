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

st.title("📊 Análise Exploratória - Autosserviço (PygWalker)")
st.divider()

st.markdown("""
Esta página oferece uma ferramenta de análise interativa usando PygWalker, permitindo que você explore os dados de forma autônoma.
""")

if "df_uci" in st.session_state:
    df = st.session_state['df_uci']
    st.info("💡 Usando dados UCI. Para usar dados OULAD, navegue primeiro para a página OULAD.")
    walker = pyg.walk(df)
else:
    st.warning("⚠️ Nenhum dado disponível. Por favor, navegue para a página UCI ou OULAD primeiro para carregar os dados.")

