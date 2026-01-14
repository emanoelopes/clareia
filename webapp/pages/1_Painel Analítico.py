"""
Painel Analítico - Visão geral dos datasets UCI e OULAD
Página separada para análise consolidada dos datasets
"""

import sys
from pathlib import Path

# Adicionar o diretório webapp ao path do Python
webapp_dir = Path(__file__).parent.parent
if str(webapp_dir) not in sys.path:
    sys.path.insert(0, str(webapp_dir))

import streamlit as st
import pandas as pd
import numpy as np
from src.utilidades import (
    obter_metricas_principais_uci,
    obter_metricas_principais_oulad,
    exibir_cartoes_informativos
)
from src.openai_interpreter import criar_sidebar_padrao

# Configuração da página
st.set_page_config(
    page_title="Painel Analítico",
    page_icon="📊",
    layout="wide"
)

# Sidebar padrão
criar_sidebar_padrao()

# Título principal
st.title("📊 Painel Analítico")
st.markdown("Visão geral dos datasets UCI e OULAD")

# Métricas principais
st.markdown("## 📈 Métricas Principais")
exibir_cartoes_informativos()

# Gráficos comparativos
st.markdown("## 📊 Análise Comparativa")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📚 Dataset UCI")
    st.markdown("**Características:**")
    st.markdown("""
    - Escolas públicas portuguesas
    - Dados demográficos e comportamentais
    - Foco em educação tradicional
    - Features: faltas, notas, tempo de estudo, etc.
    """)

with col2:
    st.markdown("### 🌐 Dataset OULAD")
    st.markdown("**Características:**")
    st.markdown("""
    - Plataforma de aprendizado online
    - Dados de engajamento digital
    - Foco em educação online
    - Features: cliques, pontuações, atividades, etc.
    """)

# Estatísticas dos Datasets
st.markdown("## 📊 Estatísticas dos Datasets")

try:
    # Carregar dados para estatísticas
    from src.carregar_dados import carregar_uci_dados, carregar_oulad_dados
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📚 Dataset UCI")
        try:
            df_uci = carregar_uci_dados()
            if not df_uci.empty:
                st.markdown(f"**Total de registros:** {len(df_uci)}")
                st.markdown(f"**Features:** {len(df_uci.columns)}")
                st.markdown(f"**Período:** Dados históricos de escolas portuguesas")
                
                # Estatísticas básicas
                if 'G3' in df_uci.columns:
                    media_notas = df_uci['G3'].mean()
                    st.metric("Média das Notas Finais", f"{media_notas:.2f}")
            else:
                st.warning("Dataset UCI não disponível")
        except Exception as e:
            st.error(f"Erro ao carregar UCI: {e}")
    
    with col2:
        st.markdown("### 🌐 Dataset OULAD")
        try:
            df_oulad = carregar_oulad_dados()
            if not df_oulad.empty:
                st.markdown(f"**Total de registros:** {len(df_oulad)}")
                st.markdown(f"**Features:** {len(df_oulad.columns)}")
                st.markdown(f"**Período:** Dados de plataforma online")
                
                # Estatísticas básicas
                if 'final_result' in df_oulad.columns:
                    taxa_aprovacao = (df_oulad['final_result'] == 'Pass').mean() * 100
                    st.metric("Taxa de Aprovação", f"{taxa_aprovacao:.1f}%")
            else:
                st.warning("Dataset OULAD não disponível")
        except Exception as e:
            st.error(f"Erro ao carregar OULAD: {e}")
            
except Exception as e:
    st.error(f"Erro ao carregar estatísticas: {e}")

# Recomendações
st.markdown("## 💡 Recomendações de Uso")

st.markdown("""
### 🎯 Para Professores e Coordenadores:

1. **Use o Template Unificado** (página inicial)
   - Combina insights de educação tradicional e online
   - Inclui campo para nome do aluno
   - Análise personalizada para sua turma

2. **Analise os Resultados**
   - Gráficos com rótulos em português
   - Interpretações para gestores escolares
   - Sugestões de intervenção pedagógica

### 🔧 Para Desenvolvedores:

1. **Estrutura Modular**
   - `utilidades.py`: Lógica de negócio
   - `home.py`: Página inicial
   - `1_Painel Analítico.py`: Painel Analítico consolidado

2. **Extensibilidade**
   - Fácil adição de novos datasets
   - Novos tipos de análise
""")

# Rodapé
st.markdown("---")
st.markdown("### ℹ️ Sobre o Sistema")
st.caption("""
**Clareia - Sistema Inteligente de Análise de Dados Educacionais**

Mestrado em Tecnologia Educacional  
Programa de Pós-Graduação em Tecnologias Educacionais (PPGTE)  
Instituto UFC Virtual (IUVI)  
Universidade Federal do Ceará (UFC)

Versão 0.1.1 - 2026
""")
