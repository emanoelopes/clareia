"""
Landing Page Principal - Sistema de Análise Educacional
Página inicial com upload de template e análise completa
"""

import sys
from pathlib import Path

# Adicionar o diretório webapp ao path do Python
webapp_dir = Path(__file__).parent
if str(webapp_dir) not in sys.path:
    sys.path.insert(0, str(webapp_dir))

import streamlit as st
import pandas as pd
import numpy as np
from src.utilidades import (
    gerar_template_unificado, 
    validar_template_usuario, 
    realizar_analise_completa,
    exibir_resultados_com_ia,
    converter_template_para_excel
)
from src.openai_interpreter import criar_sidebar_landpage

# Configuração da página
st.set_page_config(
    page_title="Clareia - Sistema de Análise de Dados Educacionais",
    page_icon="📊",
    layout="wide"
)

# Sidebar minimalista
criar_sidebar_landpage()

# Título principal
st.title("📊 Clareia - Sistema de Análise de Dados Educacionais")

# Seção 1: Download do Template
st.markdown("## 📥 Passo 1: Baixe o Template")
st.markdown("""
O template inclui as **2 features mais importantes** identificadas em:
- **UCI**: Escolas públicas portuguesas (nota_2bim, faltas)
- **OULAD**: Plataforma de aprendizado online (pontuacao, regiao)
""")

col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("""
    **Como usar o Template Unificado:**
    1. Baixe o template Excel pré-gerado com as features mais importantes
    2. Preencha o template com seus dados (incluindo o nome do aluno)
    3. Use escala 0-10 para 'resultado_final' (padrão brasileiro)
    4. Faça upload do template preenchido para análise automática
    
    **Vantagens do Template Unificado:**
    - Combina insights de educação tradicional (UCI) e online (OULAD)
    - Inclui campo para nome do aluno para personalização
    - Análise mais abrangente com features diversificadas
    """)
with col2:
    st.info("""
    **Template inclui:**
    - Campo para nome do aluno
    - Top 2 features do UCI
    - Top 2 features do OULAD
    - Coluna de resultado final
    """)

# Botão para download do template pré-gerado
if st.button("📥 Baixar Template Unificado", type="primary"):
    import os
    import zipfile
    
    template_path = "template_unificado_features.xlsx"
    df_template = None
    
    # Tentar carregar do disco primeiro
    if os.path.exists(template_path):
        try:
            df_template = pd.read_excel(template_path, engine='openpyxl')
            st.session_state.template_downloaded = True
        except (zipfile.BadZipFile, Exception) as e:
            st.warning(f"⚠️ O template em disco está corrompido ou é inválido ({e}). Gerando um novo template dinamicamente...")
            df_template = gerar_template_unificado()
    else:
        st.info("ℹ️ Template não encontrado em disco. Gerando um novo template...")
        df_template = gerar_template_unificado()

    if df_template is not None and not df_template.empty:
        feature_cols = [col for col in df_template.columns if col not in ['nome_aluno', 'resultado_final']]
        st.success(f"✅ Template unificado preparado! Inclui {len(feature_cols)} features: {', '.join(feature_cols)}")
        
        st.markdown("**Preview do Template Unificado:**")
        st.dataframe(df_template.head(), use_container_width=True)
        
        # Converter para bytes para o botão de download
        excel_data = converter_template_para_excel(df_template)
        
        if excel_data:
            st.download_button(
                "⬇️ Baixar Template Excel",
                data=excel_data,
                file_name="template_analise_educacional.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key='download_unified_template'
            )
        else:
            st.error("❌ Falha ao converter o template para Excel.")
    else:
        st.error("❌ Não foi possível gerar ou carregar o template.")

# Seção 2: Upload e Análise
st.markdown("## 📤 Passo 2: Envie o Template Preenchido")

uploaded_file = st.file_uploader(
    "Envie a planilha preenchido:",
    type=['xlsx', 'csv'],
    help="Template com dados dos alunos preenchidos"
)

if uploaded_file:
    try:
        # Carregar dados
        if uploaded_file.name.endswith('.xlsx'):
            df_usuario = pd.read_excel(uploaded_file, engine='openpyxl')
        else:
            df_usuario = pd.read_csv(uploaded_file)
        
        # Validar template
        is_valid, msg = validar_template_usuario(df_usuario)
        
        if is_valid:
            st.success(f"✅ {msg}")
            st.session_state.user_data_uploaded = df_usuario
            
            st.markdown("**Preview dos Dados Carregados:**")
            st.dataframe(df_usuario.head(), use_container_width=True)
            
            if st.button("🔍 Executar Análise Completa", type="primary"):
                with st.spinner("Executando análise completa..."):
                    # Realizar análise
                    resultados = realizar_analise_completa(df_usuario)
                    
                    if resultados:
                        st.session_state.analise_resultados = resultados
                        st.success("✅ Análise concluída com sucesso!")
                        
                        # Os resultados serão exibidos na seção abaixo
                    else:
                        st.error("❌ Erro na análise. Verifique os dados e tente novamente.")
        else:
            st.error(f"❌ {msg}")
            
    except Exception as e:
        import zipfile
        if "BadZipFile" in str(e) or isinstance(e, zipfile.BadZipFile):
            st.error("❌ Erro: O arquivo enviado não é um arquivo Excel válido ou está corrompido. Certifique-se de que você baixou o template corretamente.")
        else:
            st.error(f"Erro ao processar arquivo: {e}")

# Seção 3: Resultados (se disponíveis)
if 'analise_resultados' in st.session_state and 'user_data_uploaded' in st.session_state:
    st.markdown("---")
    # Exibir resultados salvos (título já está na função exibir_resultados_com_ia)
    exibir_resultados_com_ia(st.session_state.analise_resultados, st.session_state.user_data_uploaded)

# Rodapé informativo
st.markdown("---")
st.markdown("### ℹ️ Sobre o Sistema")
st.caption("""
**Clareia - Sistema Inteligente de Análise de Dados Educacionais**

Mestrado em Tecnologia Educacional  
Programa de Pós-Graduação em Tecnologias Educacionais (PPGTE)  
Instituto UFC Virtual (IUVI)  
Universidade Federal do Ceará (UFC)

Versão 0.1.1 - 2025
""")