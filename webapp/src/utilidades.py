from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
try:
    from .carregar_dados import carregar_uci_dados, carregar_oulad_dados
except ImportError:
    # Fallback para quando executado diretamente
    from carregar_dados import carregar_uci_dados, carregar_oulad_dados

def leitura_oulad_data():
    """Função para leitura dos dados OULAD - mantida para compatibilidade"""
    datasets_path = Path(__file__).parent.parents[1] / 'datasets' / 'oulad_data'
    return datasets_path

@st.cache_data(ttl=3600)  # Cache por 1 hora
def carregar_dados_uci_cached():
    """Carrega dados UCI com cache"""
    try:
        df = carregar_uci_dados()
        if df is None or df.empty:
            st.error("⚠️ Dados UCI carregados estão vazios ou None")
        return df
    except Exception as e:
        st.error(f"❌ Erro ao carregar dados UCI: {e}")
        import traceback
        print(f"ERRO CARREGAR UCI CACHED: {traceback.format_exc()}")
        raise

@st.cache_data(ttl=3600)  # Cache por 1 hora
def carregar_dados_oulad_cached():
    """Carrega dados OULAD com cache"""
    try:
        df = carregar_oulad_dados()
        if df is None or df.empty:
            st.error("⚠️ Dados OULAD carregados estão vazios ou None")
        return df
    except Exception as e:
        st.error(f"❌ Erro ao carregar dados OULAD: {e}")
        import traceback
        print(f"ERRO CARREGAR OULAD CACHED: {traceback.format_exc()}")
        raise

def carregar_dados_dashboard():
    """Carrega os dados processados para o painel analítico com cache"""
    try:
        # Carregar dados UCI com cache
        df_uci = carregar_dados_uci_cached()
        st.session_state['df_uci'] = df_uci
    except Exception as e:
        st.warning(f"Erro ao carregar dados UCI: {e}")
        df_uci = pd.DataFrame()
        st.session_state['df_uci'] = df_uci
    
    try:
        # Carregar dados OULAD com cache
        df_oulad = carregar_dados_oulad_cached()
        st.session_state['df_oulad'] = df_oulad
    except Exception as e:
        st.warning(f"Erro ao carregar dados OULAD: {e}")
        df_oulad = pd.DataFrame()
        st.session_state['df_oulad'] = df_oulad
    
    return df_uci, df_oulad

def obter_metricas_principais_uci():
    """Retorna métricas principais do dataset UCI calculadas dinamicamente"""
    try:
        df_uci = carregar_dados_uci_cached()
        if df_uci is None or df_uci.empty:
            st.error("⚠️ Erro: Dados UCI não puderam ser carregados ou estão vazios.")
            return {
                'total_estudantes': 0,
                'media_nota_final': 0,
                'taxa_aprovacao': 0,
                'media_faltas': 0,
                'distribuicao_genero': {},
                'media_tempo_estudo': 0,
                'correlacao_g1_g3': 0,
                'correlacao_g2_g3': 0,
                'estudantes_alcool_baixo': 0,
                'estudantes_alcool_alto': 0
            }
        
        # Calcular métricas reais - contar estudantes únicos baseado em características demográficas
        # Usar combinação de colunas que identificam unicamente cada estudante
        colunas_id = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian']
        total_estudantes = df_uci[colunas_id].drop_duplicates().shape[0]
        media_nota_final = df_uci['G3'].mean() if 'G3' in df_uci.columns else 0
        taxa_aprovacao = (df_uci['G3'] >= 10).mean() * 100 if 'G3' in df_uci.columns else 0
        media_faltas = df_uci['absences'].mean() if 'absences' in df_uci.columns else 0
        
        # Distribuição de gênero
        if 'sex' in df_uci.columns:
            dist_genero = df_uci['sex'].value_counts(normalize=True) * 100
            distribuicao_genero = {k: round(v, 1) for k, v in dist_genero.to_dict().items()}
        else:
            distribuicao_genero = {}
        
        # Tempo de estudo médio - converter strings para números
        if 'studytime' in df_uci.columns:
            # Mapear strings para números para calcular média
            studytime_map = {'<2h': 1, '2-5h': 2, '5-10h': 3, '>10h': 4}
            studytime_numeric = df_uci['studytime'].map(studytime_map)
            media_tempo_estudo = studytime_numeric.mean()
        else:
            media_tempo_estudo = 0
        
        # Correlações
        correlacao_g1_g3 = df_uci[['G1', 'G3']].corr().iloc[0, 1] if all(col in df_uci.columns for col in ['G1', 'G3']) else 0
        correlacao_g2_g3 = df_uci[['G2', 'G3']].corr().iloc[0, 1] if all(col in df_uci.columns for col in ['G2', 'G3']) else 0
        
        # Consumo de álcool
        if 'Dalc' in df_uci.columns:
            alcool_baixo = (df_uci['Dalc'] <= 2).mean() * 100
            alcool_alto = (df_uci['Dalc'] >= 4).mean() * 100
        else:
            alcool_baixo = 0
            alcool_alto = 0
        
        return {
            'total_estudantes': total_estudantes,
            'media_nota_final': round(media_nota_final, 2),
            'taxa_aprovacao': round(taxa_aprovacao, 1),
            'media_faltas': round(media_faltas, 1),
            'distribuicao_genero': distribuicao_genero,
            'media_tempo_estudo': round(media_tempo_estudo, 1),
            'correlacao_g1_g3': round(correlacao_g1_g3, 2),
            'correlacao_g2_g3': round(correlacao_g2_g3, 2),
            'estudantes_alcool_baixo': round(alcool_baixo, 1),
            'estudantes_alcool_alto': round(alcool_alto, 1)
        }
    except Exception as e:
        import traceback
        error_msg = f"Erro ao calcular métricas UCI: {str(e)}"
        st.error(f"❌ {error_msg}")
        st.exception(e)  # Mostra o traceback completo
        print(f"ERRO DETALHADO UCI: {traceback.format_exc()}")
        return {
            'total_estudantes': 0,
            'media_nota_final': 0,
            'taxa_aprovacao': 0,
            'media_faltas': 0,
            'distribuicao_genero': {},
            'media_tempo_estudo': 0,
            'correlacao_g1_g3': 0,
            'correlacao_g2_g3': 0,
            'estudantes_alcool_baixo': 0,
            'estudantes_alcool_alto': 0
        }

def obter_metricas_principais_oulad():
    """Retorna métricas principais do dataset OULAD calculadas dinamicamente"""
    try:
        df_oulad = carregar_dados_oulad_cached()
        if df_oulad.empty:
            return {
                'total_estudantes': 0,
                'taxa_aprovacao': 0,
                'media_cliques': 0,
                'distribuicao_genero': {},
                'faixa_etaria_principal': 'N/A',
                'atividade_mais_comum': 'N/A',
                'regiao_principal': 'N/A',
                'estudantes_aprovados': 0,
                'estudantes_distincao': 0,
                'estudantes_reprovados': 0
            }
        
        # Calcular métricas reais
        # Usar nunique() para contar estudantes únicos, não registros
        if 'id_student' in df_oulad.columns:
            total_estudantes = df_oulad['id_student'].nunique()
        else:
            total_estudantes = len(df_oulad)  # Fallback se não houver coluna id_student
        
        # Calcular média de cliques - verificar tanto 'clicks' quanto 'sum_click'
        if 'clicks' in df_oulad.columns:
            media_cliques = df_oulad['clicks'].mean()
        elif 'sum_click' in df_oulad.columns:
            media_cliques = df_oulad['sum_click'].mean()
        else:
            media_cliques = 0
        
        # Taxa de aprovação
        if 'final_result' in df_oulad.columns:
            taxa_aprovacao = (df_oulad['final_result'] == 'Pass').mean() * 100
            estudantes_aprovados = taxa_aprovacao
            estudantes_distincao = (df_oulad['final_result'] == 'Distinction').mean() * 100
            estudantes_reprovados = (df_oulad['final_result'] == 'Fail').mean() * 100
        else:
            taxa_aprovacao = 0
            estudantes_aprovados = 0
            estudantes_distincao = 0
            estudantes_reprovados = 0
        
        # Distribuição de gênero
        if 'gender' in df_oulad.columns and 'id_student' in df_oulad.columns:
            dist_genero = df_oulad.groupby('gender', observed=False)['id_student'].nunique()
            total_estudantes = df_oulad['id_student'].nunique()
            dist_genero_pct = (dist_genero / total_estudantes * 100)
            distribuicao_genero = {k: round(v, 1) for k, v in dist_genero_pct.to_dict().items()}
        else:
            distribuicao_genero = {}
        
        # Faixa etária principal
        if 'age_band' in df_oulad.columns and 'id_student' in df_oulad.columns:
            # Encontrar a faixa etária com mais estudantes únicos
            idade_counts = df_oulad.groupby('age_band', observed=False)['id_student'].nunique()
            faixa_etaria_principal = idade_counts.idxmax() if not idade_counts.empty else 'N/A'
        else:
            faixa_etaria_principal = 'N/A'
        
        # Atividade mais comum
        if 'activity_type' in df_oulad.columns:
            atividade_mais_comum = df_oulad['activity_type'].mode().iloc[0] if not df_oulad['activity_type'].mode().empty else 'N/A'
        else:
            atividade_mais_comum = 'N/A'
        
        # Região principal
        if 'region' in df_oulad.columns and 'id_student' in df_oulad.columns:
            # Encontrar a região com mais estudantes únicos
            regiao_counts = df_oulad.groupby('region', observed=False)['id_student'].nunique()
            regiao_principal = regiao_counts.idxmax() if not regiao_counts.empty else 'N/A'
        else:
            regiao_principal = 'N/A'
        
        return {
            'total_estudantes': total_estudantes,
            'taxa_aprovacao': round(taxa_aprovacao, 1),
            'media_cliques': round(media_cliques, 2),
            'distribuicao_genero': distribuicao_genero,
            'faixa_etaria_principal': faixa_etaria_principal,
            'atividade_mais_comum': atividade_mais_comum,
            'regiao_principal': regiao_principal,
            'estudantes_aprovados': round(estudantes_aprovados, 1),
            'estudantes_distincao': round(estudantes_distincao, 1),
            'estudantes_reprovados': round(estudantes_reprovados, 1)
        }
    except Exception as e:
        import traceback
        error_msg = f"Erro ao calcular métricas OULAD: {str(e)}"
        st.error(f"❌ {error_msg}")
        st.exception(e)  # Mostra o traceback completo
        print(f"ERRO DETALHADO OULAD: {traceback.format_exc()}")
        return {
            'total_estudantes': 0,
            'taxa_aprovacao': 0,
            'media_cliques': 0,
            'distribuicao_genero': {},
            'faixa_etaria_principal': 'N/A',
            'atividade_mais_comum': 'N/A',
            'regiao_principal': 'N/A',
            'estudantes_aprovados': 0,
            'estudantes_distincao': 0,
            'estudantes_reprovados': 0
        }

def calcular_metricas_uci(df_uci):
    """Calcula métricas principais para o dataset UCI"""
    if df_uci.empty:
        return {}
    
    # Contar estudantes únicos baseado em características demográficas
    colunas_id = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian']
    total_estudantes_unicos = df_uci[colunas_id].drop_duplicates().shape[0]
    
    metricas = {
        'total_alunos': total_estudantes_unicos,
        'media_nota_final': df_uci['G3'].mean() if 'G3' in df_uci.columns else 0,
        'taxa_aprovacao': (df_uci['G3'] >= 10).mean() * 100 if 'G3' in df_uci.columns else 0,
        'media_faltas': df_uci['absences'].mean() if 'absences' in df_uci.columns else 0,
        'media_tempo_estudo': df_uci['studytime'].map({'<2h': 1, '2-5h': 2, '5-10h': 3, '>10h': 4}).mean() if 'studytime' in df_uci.columns else 0,
        'distribuicao_genero': df_uci['sex'].value_counts().to_dict() if 'sex' in df_uci.columns else {},
        'correlacao_notas': df_uci[['G1', 'G2', 'G3']].corr().to_dict() if all(col in df_uci.columns for col in ['G1', 'G2', 'G3']) else {}
    }
    return metricas

def calcular_metricas_oulad(df_oulad):
    """Calcula métricas principais para o dataset OULAD"""
    if df_oulad.empty:
        return {}
    
    metricas = {
        'total_estudantes': df_oulad['id_student'].nunique() if 'id_student' in df_oulad.columns else len(df_oulad),
        'media_cliques': (
            df_oulad['clicks'].mean() if 'clicks' in df_oulad.columns 
            else (df_oulad['sum_click'].mean() if 'sum_click' in df_oulad.columns else 0)
        ),
        'taxa_aprovacao': (df_oulad['final_result'] == 'Pass').mean() * 100 if 'final_result' in df_oulad.columns else 0,
        'distribuicao_genero': df_oulad.groupby('gender', observed=False)['id_student'].nunique().to_dict() if 'gender' in df_oulad.columns and 'id_student' in df_oulad.columns else {},
        'distribuicao_idade': df_oulad.groupby('age_band', observed=False)['id_student'].nunique().to_dict() if 'age_band' in df_oulad.columns and 'id_student' in df_oulad.columns else {},
        'atividade_mais_comum': df_oulad['activity_type'].mode().iloc[0] if 'activity_type' in df_oulad.columns else 'N/A',
        'regiao_mais_comum': df_oulad['region'].mode().iloc[0] if 'region' in df_oulad.columns else 'N/A'
    }
    return metricas

def gerar_metricas_consolidadas(df_uci, df_oulad):
    """Gera métricas consolidadas para o painel analítico"""
    metricas_uci = calcular_metricas_uci(df_uci)
    metricas_oulad = calcular_metricas_oulad(df_oulad)
    
    # Métricas consolidadas
    total_estudantes = metricas_uci.get('total_alunos', 0) + metricas_oulad.get('total_estudantes', 0)
    taxa_aprovacao_geral = np.mean([
        metricas_uci.get('taxa_aprovacao', 0),
        metricas_oulad.get('taxa_aprovacao', 0)
    ])
    
    return {
        'total_estudantes_geral': total_estudantes,
        'taxa_aprovacao_geral': taxa_aprovacao_geral,
        'metricas_uci': metricas_uci,
        'metricas_oulad': metricas_oulad
    }

def criar_sidebar_dashboard():
    """Cria a barra lateral do painel analítico"""
    with st.sidebar:
        st.markdown("### 📊 Painel Analítico")
        
        # Carregar métricas dinâmicas
        metricas_uci = obter_metricas_principais_uci()
        metricas_oulad = obter_metricas_principais_oulad()
        
        st.markdown("### 📚 Sobre os Datasets")
        st.markdown(f"""
        **📚 UCI Dataset:**
        - Escolas públicas portuguesas
        - {metricas_uci['total_estudantes']:,} estudantes
        - Dados demográficos e acadêmicos
        - Análise de fatores de sucesso
        """)
        
        st.markdown(f"""
        **🌐 OULAD Dataset:**
        - Plataforma de aprendizado online
        - {metricas_oulad['total_estudantes']:,} estudantes
        - Dados de engajamento digital
        - Análise de atividades online
        """)
        
        st.markdown("---")
        st.markdown("### 📈 Métricas Rápidas")
        
        # Métricas UCI
        st.metric(
            "🎓 UCI - Aprovação",
            f"{metricas_uci['taxa_aprovacao']:.1f}%",
            help="Taxa de aprovação nas escolas públicas"
        )
        
        st.metric(
            "📊 UCI - Média Notas",
            f"{metricas_uci['media_nota_final']:.1f}",
            help="Média das notas finais"
        )
        
        # Métricas OULAD
        st.metric(
            "🌐 OULAD - Aprovação",
            f"{metricas_oulad['taxa_aprovacao']:.1f}%",
            help="Taxa de aprovação na plataforma online"
        )
        
        st.metric(
            "🖱️ OULAD - Engajamento",
            f"{metricas_oulad['media_cliques']:.1f}",
            help="Média de cliques por estudante"
        )
        
        st.markdown("---")
        st.markdown("### 💡 Principais Insights")
        
        # Insights dinâmicos baseados nos dados reais
        insights_text = []
        
        if metricas_uci['correlacao_g1_g3'] > 0.7:
            insights_text.append(f"**Correlação forte** entre notas bimestrais e finais ({metricas_uci['correlacao_g1_g3']:.2f})")
        
        if metricas_uci['distribuicao_genero']:
            genero_maioria = max(metricas_uci['distribuicao_genero'], key=metricas_uci['distribuicao_genero'].get)
            insights_text.append(f"**Gênero predominante**: {genero_maioria} ({metricas_uci['distribuicao_genero'][genero_maioria]:.1f}%)")
        
        if metricas_uci['media_faltas'] > 0:
            insights_text.append(f"**Média de faltas**: {metricas_uci['media_faltas']:.1f} por estudante")
        
        if metricas_uci['media_tempo_estudo'] > 0:
            insights_text.append(f"**Tempo de estudo médio**: {metricas_uci['media_tempo_estudo']:.1f}h/semana")
        
        if metricas_oulad['atividade_mais_comum'] != 'N/A':
            insights_text.append(f"**Atividade mais comum**: {metricas_oulad['atividade_mais_comum']}")
        
        if insights_text:
            for insight in insights_text:
                st.markdown(f"- {insight}")
        else:
            st.markdown("""
            - **Correlação forte** entre notas bimestrais e finais
            - **Gênero influencia** desempenho acadêmico
            - **Faltas impactam** negativamente o desempenho
            - **Tempo de estudo** ideal: 5-10h/semana
            - **Atividades online** mais efetivas: outcontent, forumng
            """)
        
        st.markdown("---")
        st.markdown("### ℹ️ Informações")
        st.markdown("**Mestrado em Tecnologia Educacional - UFC**")
    
    return None, None  # Retorna None para manter compatibilidade

def exibir_cartoes_informativos():
    """Exibe cartões informativos com métricas principais"""
    metricas_uci = obter_metricas_principais_uci()
    metricas_oulad = obter_metricas_principais_oulad()
    
    # Cartões principais (título deve ser adicionado pela página que chama esta função)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "🎓 Total de Estudantes", 
            f"{metricas_uci['total_estudantes'] + metricas_oulad['total_estudantes']:,}",
            help="Soma dos estudantes dos datasets UCI e OULAD"
        )
    
    with col2:
        taxa_geral = (metricas_uci['taxa_aprovacao'] + metricas_oulad['taxa_aprovacao']) / 2
        st.metric(
            "✅ Taxa de Aprovação Geral", 
            f"{taxa_geral:.1f}%",
            help="Média das taxas de aprovação dos dois datasets"
        )
    
    with col3:
        st.metric(
            "📚 Média de Notas (UCI)", 
            f"{metricas_uci['media_nota_final']:.1f}",
            help="Média das notas finais no dataset UCI"
        )
    
    with col4:
        st.metric(
            "🖱️ Média de Cliques (OULAD)", 
            f"{metricas_oulad['media_cliques']:.1f}",
            help="Média de cliques por estudante no dataset OULAD"
        )

def exibir_cartoes_detalhados():
    """Exibe cartões detalhados para cada dataset"""
    metricas_uci = obter_metricas_principais_uci()
    metricas_oulad = obter_metricas_principais_oulad()
    
    # Cartões UCI
    st.markdown("### 📚 Dataset UCI - Escolas Públicas Portuguesas")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "👥 Total de Estudantes", 
            f"{metricas_uci['total_estudantes']:,}",
            help="Estudantes de escolas públicas portuguesas"
        )
    
    with col2:
        st.metric(
            "✅ Taxa de Aprovação", 
            f"{metricas_uci['taxa_aprovacao']:.1f}%",
            help="Percentual de estudantes aprovados"
        )
    
    with col3:
        st.metric(
            "📊 Média de Faltas", 
            f"{metricas_uci['media_faltas']:.1f}",
            help="Número médio de faltas por estudante"
        )
    
    with col4:
        st.metric(
            "⏰ Tempo de Estudo", 
            f"{metricas_uci['media_tempo_estudo']:.1f}h/semana",
            help="Tempo médio de estudo semanal"
        )
    
    # Cartões OULAD
    st.markdown("### 🌐 Dataset OULAD - Plataforma de Aprendizado Online")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "👥 Total de Estudantes", 
            f"{metricas_oulad['total_estudantes']:,}",
            help="Estudantes da plataforma online"
        )
    
    with col2:
        st.metric(
            "✅ Taxa de Aprovação", 
            f"{metricas_oulad['taxa_aprovacao']:.1f}%",
            help="Percentual de estudantes aprovados"
        )
    
    with col3:
        st.metric(
            "🏆 Distinção", 
            f"{metricas_oulad['estudantes_distincao']:.1f}%",
            help="Percentual de estudantes com distinção"
        )
    
    with col4:
        st.metric(
            "🖱️ Engajamento", 
            f"{metricas_oulad['media_cliques']:.1f} cliques",
            help="Média de cliques por estudante"
        )

def obter_insights_uci():
    """Retorna insights principais do dataset UCI baseados em dados reais"""
    metricas = obter_metricas_principais_uci()
    
    insights = []
    
    # Correlação forte
    if metricas['correlacao_g1_g3'] > 0.7 and metricas['correlacao_g2_g3'] > 0.7:
        insights.append(f"🎯 **Correlação Forte**: Notas do 1º e 2º bimestre têm correlação de {metricas['correlacao_g1_g3']:.2f} e {metricas['correlacao_g2_g3']:.2f} com a nota final")
    
    # Gênero
    if metricas['distribuicao_genero']:
        genero_maioria = max(metricas['distribuicao_genero'], key=metricas['distribuicao_genero'].get)
        genero_menor = min(metricas['distribuicao_genero'], key=metricas['distribuicao_genero'].get)
        insights.append(f"👥 **Gênero**: Estudantes do sexo {genero_maioria} representam {metricas['distribuicao_genero'][genero_maioria]:.1f}% vs {genero_menor} com {metricas['distribuicao_genero'][genero_menor]:.1f}%")
    
    # Consumo de álcool
    if metricas['estudantes_alcool_baixo'] > 0:
        insights.append(f"🍷 **Consumo de Álcool**: {metricas['estudantes_alcool_baixo']:.1f}% dos estudantes têm baixo consumo, com melhor desempenho acadêmico")
    
    # Tempo de estudo
    if metricas['media_tempo_estudo'] > 0:
        insights.append(f"📚 **Tempo de Estudo**: Média de {metricas['media_tempo_estudo']:.1f}h/semana por estudante")
    
    # Faltas
    if metricas['media_faltas'] > 0:
        insights.append(f"❌ **Faltas**: Média de {metricas['media_faltas']:.1f} faltas por estudante")
    
    # Taxa de aprovação
    if metricas['taxa_aprovacao'] > 0:
        insights.append(f"✅ **Aprovação**: Taxa de aprovação de {metricas['taxa_aprovacao']:.1f}%")
    
    # Média de notas
    if metricas['media_nota_final'] > 0:
        insights.append(f"📊 **Desempenho**: Média de notas finais de {metricas['media_nota_final']:.1f}")
    
    return {
        'titulo': '📚 Principais Insights - Dataset UCI',
        'insights': insights if insights else [
            "🎯 **Correlação Forte**: Notas do 1º e 2º bimestre têm correlação forte com a nota final",
            "👥 **Gênero**: Distribuição equilibrada entre gêneros",
            "📚 **Tempo de Estudo**: Fator importante para o desempenho acadêmico",
            "❌ **Faltas**: Impactam negativamente o desempenho",
            "👨‍👩‍👧‍👦 **Família**: Escolaridade dos pais influencia o desempenho dos filhos"
        ]
    }

def obter_insights_oulad():
    """Retorna insights principais do dataset OULAD baseados em dados reais"""
    metricas = obter_metricas_principais_oulad()
    
    insights = []
    
    # Demografia
    if metricas['distribuicao_genero']:
        genero_maioria = max(metricas['distribuicao_genero'], key=metricas['distribuicao_genero'].get)
        insights.append(f"👥 **Demografia**: {metricas['distribuicao_genero'][genero_maioria]:.1f}% são do sexo {genero_maioria}")
    
    if metricas['faixa_etaria_principal'] != 'N/A':
        insights.append(f"👥 **Faixa Etária**: Faixa etária predominante de {metricas['faixa_etaria_principal']}")
    
    # Desempenho
    if metricas['taxa_aprovacao'] > 0:
        insights.append(f"🏆 **Alto Desempenho**: {metricas['taxa_aprovacao']:.1f}% de aprovação")
    
    if metricas['estudantes_distincao'] > 0:
        insights.append(f"🏆 **Distinção**: {metricas['estudantes_distincao']:.1f}% obtendo distinção")
    
    # Engajamento
    if metricas['media_cliques'] > 0:
        insights.append(f"🖱️ **Engajamento**: Média de {metricas['media_cliques']:.1f} cliques por estudante, indicando engajamento moderado")
    
    # Atividades
    if metricas['atividade_mais_comum'] != 'N/A':
        insights.append(f"📚 **Atividades**: '{metricas['atividade_mais_comum']}' é a atividade mais realizada")
    
    # Região
    if metricas['regiao_principal'] != 'N/A':
        insights.append(f"🌍 **Região**: {metricas['regiao_principal']} concentra a maior parte dos estudantes")
    
    # Distribuição de resultados
    if metricas['estudantes_reprovados'] > 0:
        insights.append(f"📊 **Distribuição**: Aprovação supera largamente outras categorias (reprovação: {metricas['estudantes_reprovados']:.1f}%)")
    
    # Total de estudantes
    if metricas['total_estudantes'] > 0:
        insights.append(f"👥 **Total**: {metricas['total_estudantes']:,} estudantes analisados")
    
    return {
        'titulo': '🌐 Principais Insights - Dataset OULAD',
        'insights': insights if insights else [
            "👥 **Demografia**: Distribuição equilibrada entre gêneros",
            "🏆 **Alto Desempenho**: Boa taxa de aprovação geral",
            "🖱️ **Engajamento**: Nível moderado de engajamento na plataforma",
            "📚 **Atividades**: Diversas atividades disponíveis",
            "🌍 **Região**: Distribuição geográfica variada",
            "📊 **Distribuição**: Resultados positivos predominam"
        ]
    }

# =============================================================================
# FUNÇÕES DE TREINAMENTO SOB DEMANDA
# =============================================================================

def treinar_modelo_uci_on_demand():
    """Treina modelo UCI sob demanda com progresso e salvamento automático"""
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import train_test_split
        import pickle
        
        # Indicador de progresso (compatível com e sem Streamlit)
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
            use_streamlit = True
        except:
            progress_bar = None
            status_text = None
            use_streamlit = False
            print("🔄 Carregando dados UCI...")
        
        if use_streamlit:
            status_text.text("🔄 Carregando dados UCI...")
            progress_bar.progress(20)
        else:
            print("🔄 Carregando dados UCI...")
        
        # Carregar dados UCI
        df_uci = carregar_uci_dados()
        
        if use_streamlit:
            status_text.text("🔄 Preparando dados...")
            progress_bar.progress(40)
        else:
            print("🔄 Preparando dados...")
        
        # Preparar dados como na página 1_UCI.py
        Y = df_uci['G3']
        X = df_uci.drop('G3', axis=1)
        
        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        if use_streamlit:
            status_text.text("🔄 Treinando modelo RandomForest...")
            progress_bar.progress(60)
        else:
            print("🔄 Treinando modelo RandomForest...")
        
        # Identificar colunas categóricas
        categorical_features = X_train.select_dtypes(include=['object']).columns
        
        # Criar preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ],
            remainder='passthrough'
        )
        
        # Criar pipeline
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        
        # Treinar modelo
        y_train_encoded = y_train.astype(float)
        model.fit(X_train, y_train_encoded)
        
        if use_streamlit:
            status_text.text("🔄 Salvando modelo...")
            progress_bar.progress(80)
        else:
            print("🔄 Salvando modelo...")
        
        # Salvar modelo
        with open('uci.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        if use_streamlit:
            status_text.text("✅ Modelo UCI treinado e salvo!")
            progress_bar.progress(100)
            
            # Limpar indicadores
            progress_bar.empty()
            status_text.empty()
        else:
            print("✅ Modelo UCI treinado e salvo!")
        
        return model
        
    except Exception as e:
        try:
            st.error(f"Erro ao treinar modelo UCI: {e}")
        except:
            print(f"❌ Erro ao treinar modelo UCI: {e}")
        return None

def treinar_modelo_oulad_on_demand():
    """Treina modelo OULAD sob demanda com progresso e salvamento automático"""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import OneHotEncoder, LabelEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import train_test_split
        from sklearn.impute import SimpleImputer
        import pickle
        import numpy as np
        
        # Indicador de progresso (compatível com e sem Streamlit)
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
            use_streamlit = True
        except:
            progress_bar = None
            status_text = None
            use_streamlit = False
            print("🔄 Carregando dados OULAD...")
        
        if use_streamlit:
            status_text.text("🔄 Carregando dados OULAD...")
            progress_bar.progress(10)
        else:
            print("🔄 Carregando dados OULAD...")
        
        # Carregar dados OULAD
        df_oulad = carregar_oulad_dados()
        
        # Usar amostra para treinamento mais rápido
        if len(df_oulad) > 50000:
            df_oulad = df_oulad.sample(n=50000, random_state=42)
            if use_streamlit:
                st.info("📊 Usando amostra de 50k registros para treinamento mais rápido")
            else:
                print("📊 Usando amostra de 50k registros para treinamento mais rápido")
        
        if use_streamlit:
            status_text.text("🔄 Preparando dados...")
            progress_bar.progress(30)
        else:
            print("🔄 Preparando dados...")
        
        # Preparar dados como na página 2_OULAD.py
        Y = df_oulad['final_result']
        X = df_oulad.loc[:, df_oulad.columns != 'final_result']
        
        # Remover colunas irrelevantes
        X = X.drop(['id_student', 'id_site', 'id_assessment', 'code_module', 'code_presentation', 'code_module_y', 'code_module_x'], axis=1, errors='ignore')
        
        if use_streamlit:
            status_text.text("🔄 Dividindo dados...")
            progress_bar.progress(50)
        else:
            print("🔄 Dividindo dados...")
        
        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        if use_streamlit:
            status_text.text("🔄 Limpando dados de treino...")
            progress_bar.progress(60)
        else:
            print("🔄 Limpando dados de treino...")
        
        # Limpar dados de treino
        nan_rows_train = y_train.isnull()
        X_train_cleaned = X_train[~nan_rows_train].copy()
        y_train_cleaned = y_train[~nan_rows_train].copy()
        
        
        if use_streamlit:
            status_text.text("🔄 Treinando modelo RandomForest...")
            progress_bar.progress(70)
        else:
            print("🔄 Treinando modelo RandomForest...")
        
        # Identificar colunas categóricas e numéricas
        categorical_cols = X_train_cleaned.select_dtypes(include='object').columns
        numerical_cols = X_train_cleaned.select_dtypes(include=[np.number]).columns
        
        # Verificar se há colunas que não são nem categóricas nem numéricas
        all_cols = set(X_train_cleaned.columns)
        processed_cols = set(categorical_cols) | set(numerical_cols)
        remaining_cols = all_cols - processed_cols
        
        if len(remaining_cols) > 0:
            # Adicionar colunas restantes como categóricas
            categorical_cols = list(categorical_cols) + list(remaining_cols)
        
        # Criar preprocessor
        transformers = []
        
        # Adicionar transformador numérico apenas se houver colunas numéricas
        if len(numerical_cols) > 0:
            transformers.append(('num', SimpleImputer(strategy='mean'), numerical_cols))
        
        # Adicionar transformador categórico apenas se houver colunas categóricas
        if len(categorical_cols) > 0:
            transformers.append(('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_cols))
        
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='passthrough'
        )
        
        # Criar pipeline
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=50, n_jobs=2, max_depth=4, random_state=42))
        ])
        
        # Treinar modelo
        model.fit(X_train_cleaned, y_train_cleaned)
        
        if use_streamlit:
            status_text.text("🔄 Salvando modelo...")
            progress_bar.progress(90)
        else:
            print("🔄 Salvando modelo...")
        
        # Salvar modelo
        with open('oulad.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        if use_streamlit:
            status_text.text("✅ Modelo OULAD treinado e salvo!")
            progress_bar.progress(100)
            
            # Limpar indicadores
            progress_bar.empty()
            status_text.empty()
        else:
            print("✅ Modelo OULAD treinado e salvo!")
        
        return model
        
    except Exception as e:
        try:
            st.error(f"Erro ao treinar modelo OULAD: {e}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
        except:
            print(f"❌ Erro ao treinar modelo OULAD: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
        return None

@st.cache_resource(ttl=7200)  # Cache por 2 horas
def carregar_modelo_uci():
    """Carrega o modelo UCI com cache ou treina sob demanda"""
    try:
        # Tentar diferentes caminhos para o arquivo pickle
        possible_paths = [
            '../uci.pkl',
            '../../uci.pkl',
            Path(__file__).parent.parents[1] / "uci.pkl",
            'uci.pkl'
        ]
        
        model = None
        for path in possible_paths:
            p = Path(path)
            if p.is_file():
                try:
                    with p.open("rb") as f:
                        model = pickle.load(f)
                    break
                except Exception as e:
                    continue
        
        if model is None:
            st.info("📦 Modelo UCI não encontrado. Treinando modelo automaticamente...")
            return treinar_modelo_uci_on_demand()
        
        return model
    except Exception as e:
        st.warning(f"Erro ao carregar modelo UCI: {e}")
        return None

@st.cache_resource(ttl=7200)  # Cache por 2 horas
def carregar_modelo_oulad():
    """Carrega o modelo OULAD com cache ou treina sob demanda"""
    try:
        # Tentar diferentes caminhos para o arquivo pickle
        possible_paths = [
            '../oulad.pkl',
            '../../oulad.pkl',
            Path(__file__).parent.parents[1] / "oulad.pkl",
            'oulad.pkl'
        ]
        
        model = None
        for path in possible_paths:
            p = Path(path)
            if p.is_file():
                try:
                    with p.open("rb") as f:
                        model = pickle.load(f)
                    break
                except Exception as e:
                    continue
        
        if model is None:
            st.info("📦 Modelo OULAD não encontrado. Treinando modelo automaticamente (pode levar alguns minutos)...")
            return treinar_modelo_oulad_on_demand()
        
        return model
    except Exception as e:
        st.warning(f"Erro ao carregar modelo OULAD: {e}")
        return None

@st.cache_data(ttl=3600)  # Cache por 1 hora (UCI é menor)
def calcular_feature_importance_uci():
    """Calcula feature importance real para UCI com otimizações"""
    try:
        from sklearn.inspection import permutation_importance
        from sklearn.model_selection import train_test_split
        
        # Indicador de progresso
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🔄 Carregando dados UCI...")
        progress_bar.progress(20)
        
        # Carregar dados UCI
        df_uci = carregar_uci_dados()
        
        status_text.text("🔄 Preparando dados...")
        progress_bar.progress(40)
        
        # Preparar dados como nas páginas individuais
        Y = df_uci['G3']
        X = df_uci.drop('G3', axis=1)
        
        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        status_text.text("🔄 Carregando modelo...")
        progress_bar.progress(60)
        
        # Carregar modelo treinado
        model = carregar_modelo_uci()
        if model is None:
            progress_bar.empty()
            status_text.empty()
            return pd.DataFrame()
        
        # Garantir que todas as colunas tenham tipos corretos
        # Converter colunas não numéricas para object (categóricas)
        for col in X_test.columns:
            if X_test[col].dtype not in [np.number, 'object']:
                # Tentar converter para numérico primeiro
                try:
                    X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
                    # Se ainda não for numérico, converter para string/object
                    if X_test[col].dtype not in [np.number]:
                        X_test[col] = X_test[col].astype(str).astype('object')
                except:
                    X_test[col] = X_test[col].astype(str).astype('object')
        
        # Garantir que colunas numéricas não tenham valores infinitos
        numeric_cols = X_test.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0)
            X_test[col] = X_test[col].replace([np.inf, -np.inf], 0)
        
        # Garantir que colunas categóricas sejam do tipo object
        categorical_cols = X_test.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            X_test[col] = X_test[col].astype(str).replace('nan', np.nan).astype('object')
        
        status_text.text("🔄 Calculando feature importance...")
        progress_bar.progress(80)
        
        # OTIMIZAÇÃO: Usar todos os cores disponíveis
        try:
            result = permutation_importance(
                model, X_test, y_test, 
                n_repeats=10,  # Manter 10 para UCI (é pequeno)
                random_state=42, 
                n_jobs=-1  # Usar todos os cores disponíveis
            )
            sorted_idx = result.importances_mean.argsort()
            
            status_text.text("✅ Finalizando...")
            progress_bar.progress(95)
            
            # Criar DataFrame com resultados reais
            features = X_test.columns[sorted_idx]
            importance = result.importances_mean[sorted_idx]
        except Exception as e:
            # Se permutation_importance falhar, retornar DataFrame vazio
            st.warning(f"Erro ao calcular feature importance: {e}")
            progress_bar.empty()
            status_text.empty()
            return pd.DataFrame()
        
        df_result = pd.DataFrame({
            'feature': features,
            'importance': importance
        }).sort_values('importance', ascending=True)
        
        # Limpar indicadores
        progress_bar.empty()
        status_text.empty()
        
        return df_result
        
    except Exception as e:
        st.warning(f"Erro ao calcular feature importance UCI: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=7200)  # Cache por 2 horas (OULAD é pesado)
def calcular_feature_importance_oulad():
    """Calcula feature importance real para OULAD com otimizações"""
    try:
        from sklearn.inspection import permutation_importance
        from sklearn.model_selection import train_test_split
        
        # Indicador de progresso
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🔄 Carregando dados OULAD...")
        progress_bar.progress(10)
        
        # Carregar dados OULAD
        df_oulad = carregar_oulad_dados()
        
        # AMOSTRAGEM: Usar apenas 50k registros para OULAD (muito mais rápido)
        if len(df_oulad) > 50000:
            df_oulad = df_oulad.sample(n=50000, random_state=42)
            st.info("📊 Usando amostra de 50k registros para análise mais rápida")
        
        status_text.text("🔄 Preparando dados...")
        progress_bar.progress(30)
        
        # Preparar dados como nas páginas individuais
        Y = df_oulad['final_result']
        X = df_oulad.loc[:, df_oulad.columns != 'final_result']
        
        # Remover colunas irrelevantes
        X = X.drop(['id_student', 'id_site', 'id_assessment', 'code_module', 'code_presentation', 'code_module_y', 'code_module_x'], axis=1, errors='ignore')
        
        status_text.text("🔄 Carregando modelo...")
        progress_bar.progress(40)
        
        # Carregar modelo treinado primeiro para saber quais colunas ele espera
        model = carregar_modelo_oulad()
        if model is None:
            progress_bar.empty()
            status_text.empty()
            return pd.DataFrame()
        
        status_text.text("🔄 Dividindo dados...")
        progress_bar.progress(50)
        
        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        status_text.text("🔄 Limpando dados de teste...")
        progress_bar.progress(70)
        
        # Limpar dados de teste
        nan_rows_test = y_test.isnull()
        X_test_cleaned = X_test[~nan_rows_test].copy()
        y_test_cleaned = y_test[~nan_rows_test].copy()
        
        # Garantir que todas as colunas tenham tipos corretos
        # Primeiro, identificar colunas numéricas e categóricas
        numeric_cols = X_test_cleaned.select_dtypes(include=[np.number]).columns
        categorical_cols = X_test_cleaned.select_dtypes(include=['object', 'category']).columns
        
        # Garantir que colunas numéricas não tenham valores infinitos ou NaN problemáticos
        for col in numeric_cols:
            X_test_cleaned[col] = pd.to_numeric(X_test_cleaned[col], errors='coerce').fillna(0)
            X_test_cleaned[col] = X_test_cleaned[col].replace([np.inf, -np.inf], 0)
        
        # Garantir que TODAS as colunas categóricas sejam explicitamente convertidas para string
        # Isso é crítico para evitar erros de conversão
        # Especialmente importante para colunas como 'activity_type' que contêm valores como 'homepage'
        for col in categorical_cols:
            # Converter todos os valores para string explicitamente
            X_test_cleaned[col] = X_test_cleaned[col].apply(
                lambda x: str(x) if pd.notna(x) and x != '' else 'missing'
            )
            # Garantir que seja do tipo object (categórica)
            X_test_cleaned[col] = X_test_cleaned[col].astype('object')
        
        # Garantir que colunas conhecidas como categóricas sejam explicitamente tratadas
        known_categorical_cols = ['activity_type', 'gender', 'region', 'highest_education', 
                                  'imd_band', 'age_band', 'disability', 'final_result',
                                  'assessment_type', 'code_module', 'code_presentation']
        for col in known_categorical_cols:
            if col in X_test_cleaned.columns:
                # Forçar conversão para string e depois object
                X_test_cleaned[col] = X_test_cleaned[col].apply(
                    lambda x: str(x) if pd.notna(x) and x != '' else 'missing'
                ).astype('object')
        
        # Tratar colunas que não são nem numéricas nem categóricas
        other_cols = set(X_test_cleaned.columns) - set(numeric_cols) - set(categorical_cols)
        for col in other_cols:
            # Tentar converter para numérico primeiro
            try:
                X_test_cleaned[col] = pd.to_numeric(X_test_cleaned[col], errors='coerce')
                if X_test_cleaned[col].isna().all():
                    # Se todas as conversões falharam, tratar como categórica
                    X_test_cleaned[col] = X_test_cleaned[col].astype(str).astype('object')
                else:
                    # Preencher NaN com 0
                    X_test_cleaned[col] = X_test_cleaned[col].fillna(0)
            except:
                # Se falhar, tratar como categórica
                X_test_cleaned[col] = X_test_cleaned[col].astype(str).astype('object')
        
        status_text.text("🔄 Testando modelo com dados de exemplo...")
        progress_bar.progress(80)
        
        # Testar se o modelo consegue processar os dados antes de calcular feature importance
        # Identificar e remover colunas problemáticas
        problematic_cols = []
        test_sample = X_test_cleaned.head(1).copy()
        
        for col in test_sample.columns:
            try:
                # Tentar fazer predição apenas com esta coluna para identificar problemas
                test_df = pd.DataFrame({col: test_sample[col]})
                # Não vamos testar coluna por coluna, mas sim identificar tipos problemáticos
                if test_sample[col].dtype == 'object':
                    # Verificar se há valores que não são strings válidas
                    non_string_values = test_sample[col].apply(lambda x: not isinstance(x, (str, type(None))))
                    if non_string_values.any():
                        problematic_cols.append(col)
            except:
                problematic_cols.append(col)
        
        # Remover colunas problemáticas se houver
        if problematic_cols:
            st.warning(f"⚠️ Removendo colunas problemáticas: {problematic_cols}")
            X_test_cleaned = X_test_cleaned.drop(columns=problematic_cols, errors='ignore')
        
        # Testar se o modelo consegue processar os dados
        try:
            # Tentar fazer uma predição de exemplo para verificar se os dados estão corretos
            _ = model.predict(X_test_cleaned.head(1))
        except Exception as test_error:
            st.warning(f"⚠️ Erro ao testar modelo com dados: {test_error}")
            st.info("💡 Tentando ajustar tipos de dados...")
            
            # Tentar converter todas as colunas categóricas explicitamente
            for col in X_test_cleaned.columns:
                if X_test_cleaned[col].dtype == 'object':
                    # Garantir que todos os valores sejam strings válidas
                    X_test_cleaned[col] = X_test_cleaned[col].apply(lambda x: str(x) if pd.notna(x) else 'missing')
                    X_test_cleaned[col] = X_test_cleaned[col].astype('object')
            
            # Tentar novamente
            try:
                _ = model.predict(X_test_cleaned.head(1))
            except Exception as test_error2:
                # Se ainda falhar, tentar identificar a coluna específica do erro
                error_msg = str(test_error2)
                if "'homepage'" in error_msg or "homepage" in error_msg:
                    # Remover coluna 'homepage' ou similar se existir
                    homepage_cols = [col for col in X_test_cleaned.columns if 'homepage' in col.lower() or 'activity_type' in col.lower()]
                    if homepage_cols:
                        st.warning(f"⚠️ Removendo colunas relacionadas a 'homepage': {homepage_cols}")
                        X_test_cleaned = X_test_cleaned.drop(columns=homepage_cols, errors='ignore')
                        try:
                            _ = model.predict(X_test_cleaned.head(1))
                        except:
                            st.error(f"❌ Não foi possível processar os dados mesmo após remover colunas problemáticas: {test_error2}")
                            progress_bar.empty()
                            status_text.empty()
                            return pd.DataFrame()
                    else:
                        st.error(f"❌ Não foi possível processar os dados mesmo após ajustes: {test_error2}")
                        progress_bar.empty()
                        status_text.empty()
                        return pd.DataFrame()
                else:
                    st.error(f"❌ Não foi possível processar os dados mesmo após ajustes: {test_error2}")
                    progress_bar.empty()
                    status_text.empty()
                    return pd.DataFrame()
        
        status_text.text("🔄 Calculando feature importance...")
        progress_bar.progress(85)
        
        # OTIMIZAÇÃO: Menos repetições e mais jobs
        # O modelo Pipeline fará o preprocessing automaticamente
        try:
            # Verificar se o modelo é um Pipeline antes de chamar permutation_importance
            if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
                # O modelo é um Pipeline, pode passar dados brutos
                result = permutation_importance(
                    model, X_test_cleaned, y_test_cleaned, 
                    n_repeats=5,  # Reduzido de 10 para 5
                    random_state=42, 
                    n_jobs=-1  # Usar todos os cores disponíveis
                )
            else:
                # Modelo não é Pipeline, precisa pré-processar manualmente
                raise ValueError("Modelo não é um Pipeline com preprocessor")
            sorted_idx = result.importances_mean.argsort()
            
            status_text.text("✅ Finalizando...")
            progress_bar.progress(95)
            
            # Criar DataFrame com resultados reais
            # Nota: após o preprocessing, as features podem ter mudado (OneHotEncoder cria múltiplas colunas)
            # Vamos usar os nomes das colunas originais
            features = X_test_cleaned.columns[sorted_idx]
            importance = result.importances_mean[sorted_idx]
        except Exception as e:
            # Se permutation_importance falhar, retornar DataFrame vazio
            st.warning(f"Erro ao calcular feature importance: {e}")
            progress_bar.empty()
            status_text.empty()
            return pd.DataFrame()
        
        df_result = pd.DataFrame({
            'feature': features,
            'importance': importance
        }).sort_values('importance', ascending=True)
        
        # Limpar indicadores
        progress_bar.empty()
        status_text.empty()
        
        return df_result
        
    except Exception as e:
        st.warning(f"Erro ao calcular feature importance OULAD: {e}")
        return pd.DataFrame()

def criar_grafico_feature_importance_uci():
    """Cria gráfico de feature importance para UCI"""
    df_importance = calcular_feature_importance_uci()
    if df_importance.empty:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(df_importance['feature'], df_importance['importance'], color='skyblue')
    ax.set_title('Importância das Features - Dataset UCI', fontsize=14, fontweight='bold')
    ax.set_xlabel('Importância')
    ax.set_ylabel('Features')
    
    # Adicionar valores nas barras
    for i, (bar, importance) in enumerate(zip(bars, df_importance['importance'])):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{importance:.3f}', va='center', fontsize=10)
    
    plt.tight_layout()
    return fig

def criar_grafico_feature_importance_oulad():
    """Cria gráfico de feature importance para OULAD"""
    df_importance = calcular_feature_importance_oulad()
    if df_importance.empty:
        return None
    
    # Tradução amigável das variáveis para exibição
    feature_translation = {
        'date_unregistration': 'Data de cancelamento',
        'date_registration': 'Data de registro',
        'age_band': 'Faixa etária',
        'studied_credits': 'Créditos cursados',
        'studied_credits_x': 'Créditos cursados',
        'studied_credits_y': 'Créditos cursados',
        'score': 'Nota',
        'score_x': 'Nota',
        'score_y': 'Nota',
        'activity_type': 'Tipo de atividade',
        'clicks': 'Cliques',
        'gender': 'Gênero',
        'region': 'Região',
        'disability': 'Deficiência',
        'highest_education': 'Escolaridade',
        'imd_band': 'Faixa IMD',
        'num_of_prev_attempts': 'Tentativas anteriores',
        'module_presentation_length': 'Duração do módulo',
        'cancelou': 'Cancelou',
    }
    df_importance['feature_pt'] = df_importance['feature'].map(feature_translation).fillna(df_importance['feature'])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(df_importance['feature_pt'], df_importance['importance'], color='lightcoral')
    ax.set_title('Importância das Features - Dataset OULAD', fontsize=14, fontweight='bold')
    ax.set_xlabel('Importância')
    ax.set_ylabel('Variáveis')
    
    # Adicionar valores nas barras
    for i, (bar, importance) in enumerate(zip(bars, df_importance['importance'])):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{importance:.3f}', va='center', fontsize=10)
    
    plt.tight_layout()
    return fig

def criar_secao_pygwalker():
    """Cria seção opcional para PyGWalker com seleção de dataset"""
    st.markdown("---")
    st.markdown("### 🔍 Análise Interativa com PyGWalker")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        dataset_selecionado = st.selectbox(
            "Selecione o dataset para análise:",
            ["UCI", "OULAD"],
            help="Escolha qual dataset analisar interativamente"
        )
    
    with col2:
        usar_pygwalker_uci = st.checkbox(
            "Ativar PyGWalker UCI", 
            value=False,
            help="Permite análise interativa dos dados UCI"
        )

        usar_pygwalker_oulad = st.checkbox(
            "Ativar PyGWalker OULAD", 
            value=False,
            help="Permite análise interativa dos dados OULAD"
        )
    if usar_pygwalker_uci:
        try:
            import pygwalker as pyg
            from pygwalker.api.streamlit import StreamlitRenderer
            
            # Carregar dados baseado na seleção
            if dataset_selecionado == "UCI":
                if 'df_uci' in st.session_state and not st.session_state['df_uci'].empty:
                    st.info("📊 Carregando PyGWalker com dados UCI...")
                    df = st.session_state['df_uci']
                else:
                    st.info("📊 Carregando dados UCI do arquivo...")
                    df = carregar_uci_dados()
            else:  # OULAD
                if 'df_oulad' in st.session_state and not st.session_state['df_oulad'].empty:
                    st.info("📊 Carregando PyGWalker com dados OULAD...")
                    df = st.session_state['df_oulad']
                else:
                    st.info("📊 Carregando dados OULAD do arquivo...")
                    df = carregar_oulad_dados()
            
            # Verificar se os dados foram carregados
            if df is not None and not df.empty:
                # Criar renderer do PyGWalker
                renderer = StreamlitRenderer(df, spec="./gw0.json", debug=False)
                renderer.render_explore()
            else:
                st.warning(f"⚠️ Nenhum dado disponível para {dataset_selecionado}. Verifique se os arquivos de dados existem.")
                
        except ImportError:
            st.error("❌ PyGWalker não está instalado. Execute: `pip install pygwalker`")
        except Exception as e:
            st.error(f"❌ Erro ao carregar PyGWalker: {e}")
    else:
        st.info(f"💡 Marque a opção acima para ativar a análise interativa com PyGWalker para o dataset {dataset_selecionado}")

    if usar_pygwalker_oulad:
        try:
            import pygwalker as pyg
            from pygwalker.api.streamlit import StreamlitRenderer
            
            # Verificar se há dados disponíveis
            if 'df_oulad' in st.session_state and not st.session_state['df_oulad'].empty:
                st.info("📊 Carregando PyGWalker com dados OULAD...")
                df = st.session_state['df_oulad']
                
                # Criar renderer do PyGWalker
                renderer = StreamlitRenderer(df, spec="./gw0.json", debug=False)
                renderer.render_explore()
                
            else:
                st.warning("⚠️ Nenhum dado disponível para análise interativa. Navegue para as páginas de análise primeiro.")
            
        except ImportError:
            st.error("❌ PyGWalker não está instalado. Execute: `pip install pygwalker`")
        except Exception as e:
            st.error(f"❌ Erro ao carregar PyGWalker: {e}")
        else:
            st.info("💡 Marque a opção acima para ativar a análise interativa com PyGWalker")

# =============================================================================
# FUNÇÕES PARA TEMPLATE DE FEATURE IMPORTANCE
# =============================================================================

def traduzir_nome_feature(feature: str, dataset_origem: str) -> str:
    """Traduz nomes de features para português brasileiro"""
    # Mapeamento de traduções para features comuns
    traducoes = {
        # UCI features
        'failures': 'reprovacoes',
        'absences': 'faltas',
        'G1': 'nota_1bim',
        'G2': 'nota_2bim',
        'G3': 'nota_final',
        'studytime': 'tempo_estudo',
        'goout': 'saidas',
        'Dalc': 'alcool_dia',
        'Walc': 'alcool_fds',
        'freetime': 'tempo_livre',
        'health': 'saude',
        'age': 'idade',
        'sex': 'sexo',
        'school': 'escola',
        'address': 'endereco',
        'famsize': 'tamanho_familia',
        'Pstatus': 'status_pais',
        'Medu': 'educacao_mae',
        'Fedu': 'educacao_pai',
        'Mjob': 'trabalho_mae',
        'Fjob': 'trabalho_pai',
        'reason': 'motivo_escola',
        'guardian': 'responsavel',
        'traveltime': 'tempo_viagem',
        
        # OULAD features
        'sum_click': 'cliques',
        'score': 'pontuacao',
        'studied_credits': 'creditos_estudados',
        'num_of_prev_attempts': 'tentativas_anteriores',
        'date': 'data',
        'date_submitted': 'data_submissao',
        'clicks': 'cliques',
        'final_result': 'resultado_final',
        'gender': 'genero',
        'region': 'regiao',
        'highest_education': 'educacao_superior',
        'imd_band': 'banda_imd',
        'age_band': 'faixa_etaria',
        'disability': 'deficiencia',
        'activity_type': 'tipo_atividade',
        'assessment_type': 'tipo_avaliacao',
        'weight': 'peso',
        'module_presentation_length': 'duracao_modulo',
    }
    
    # Tentar traduzir, se não existir mapeamento, retornar o original em lowercase
    return traducoes.get(feature, feature.lower().replace('_', '_'))

def gerar_template_unificado() -> pd.DataFrame:
    """Gera template unificado com TOP 2 features de UCI e OULAD"""
    try:
        # Get TOP 2 features from UCI (não 3!)
        df_importance_uci = calcular_feature_importance_uci()
        top_features_uci = df_importance_uci.nlargest(2, 'importance')['feature'].tolist() if not df_importance_uci.empty else []
        
        # Get TOP features from OULAD, excluding temporal features
        df_importance_oulad = calcular_feature_importance_oulad()
        if not df_importance_oulad.empty:
            # Exclude temporal features that don't make sense for regular school periods
            temporal_features = ['date_registration', 'date_unregistration']
            df_filtered = df_importance_oulad[~df_importance_oulad['feature'].isin(temporal_features)]
            top_features_oulad = df_filtered.nlargest(2, 'importance')['feature'].tolist()
        else:
            top_features_oulad = []
        
        # Build template with name field first
        template_data = {'nome_aluno': [''] * 10}
        
        # Store mapping for validation later
        feature_mapping = {}
        
        # Add UCI features translated to Portuguese, without prefix
        for feature in top_features_uci:
            translated = traduzir_nome_feature(feature, 'uci')
            # If translation conflicts, add _uci suffix
            if translated in template_data or translated in feature_mapping:
                translated = f"{translated}_uci"
            template_data[translated] = [np.nan] * 10
            feature_mapping[translated] = ('uci', feature)
        
        # Add OULAD features translated to Portuguese, without prefix
        for feature in top_features_oulad:
            translated = traduzir_nome_feature(feature, 'oulad')
            # If translation conflicts, add _oulad suffix
            if translated in template_data or translated in feature_mapping:
                translated = f"{translated}_oulad"
            template_data[translated] = [np.nan] * 10
            feature_mapping[translated] = ('oulad', feature)
        
        # Add result column
        template_data['resultado_final'] = [np.nan] * 10
        
        df_template = pd.DataFrame(template_data)
        
        # Add example row with placeholder values
        df_template.loc[0, 'nome_aluno'] = 'João Silva'
        
        # Add example values based on typical feature ranges
        for col in df_template.columns:
            if col == 'nome_aluno':
                continue
            elif col == 'resultado_final':
                df_template.loc[0, col] = 7.5  # Exemplo de nota 0-10 (padrão brasileiro)
            elif 'reprovacoes' in col or 'faltas' in col or 'tentativas' in col:
                df_template.loc[0, col] = 0
            elif 'nota' in col or 'pontuacao' in col:
                df_template.loc[0, col] = 10.0
            elif 'cliques' in col or 'creditos' in col:
                df_template.loc[0, col] = 50
            elif 'tempo' in col:
                df_template.loc[0, col] = 2
            else:
                df_template.loc[0, col] = 'Exemplo'
        
        # Store mapping as metadata (could be saved separately if needed)
        df_template.attrs['feature_mapping'] = feature_mapping
        
        return df_template
        
    except Exception as e:
        st.error(f"Erro ao gerar template unificado: {e}")
        return pd.DataFrame()

def gerar_template_features(dataset_tipo: str) -> pd.DataFrame:
    """Gera template com as 2 features mais importantes do dataset selecionado"""
    try:
        # Obter feature importance baseado no dataset
        if dataset_tipo.lower() == 'uci':
            df_importance = calcular_feature_importance_uci()
        elif dataset_tipo.lower() == 'oulad':
            df_importance = calcular_feature_importance_oulad()
        else:
            raise ValueError(f"Dataset tipo '{dataset_tipo}' não reconhecido. Use 'uci' ou 'oulad'")
        
        if df_importance.empty:
            st.warning(f"Não foi possível obter feature importance para {dataset_tipo}")
            return pd.DataFrame()
        
        # Pegar as 2 features mais importantes (maior importance)
        top_features = df_importance.nlargest(2, 'importance')['feature'].tolist()
        
        # Criar template DataFrame
        template_data = {
            feature: [np.nan] * 10 for feature in top_features
        }
        template_data['resultado_final'] = [np.nan] * 10
        
        df_template = pd.DataFrame(template_data)
        
        # Adicionar algumas linhas de exemplo com valores placeholder
        if dataset_tipo.lower() == 'uci':
            # Exemplos baseados no dataset UCI
            if len(top_features) >= 1:
                df_template.loc[0, top_features[0]] = 1.0  # Exemplo de valor numérico
            if len(top_features) >= 2:
                df_template.loc[0, top_features[1]] = 2.0
            df_template.loc[0, 'resultado_final'] = 10.0  # Exemplo de nota
        else:  # OULAD
            # Exemplos baseados no dataset OULAD
            if len(top_features) >= 1:
                df_template.loc[0, top_features[0]] = 'Pass'  # Exemplo de categoria
            if len(top_features) >= 2:
                df_template.loc[0, top_features[1]] = 1.0
            df_template.loc[0, 'resultado_final'] = 'Pass'
        
        return df_template
        
    except Exception as e:
        st.error(f"Erro ao gerar template: {e}")
        return pd.DataFrame()

def converter_template_para_excel(df_template: pd.DataFrame) -> bytes:
    """Converte DataFrame template para formato Excel (bytes)"""
    try:
        import io
        
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df_template.to_excel(writer, sheet_name='Template', index=False)
        buffer.seek(0)
        return buffer.getvalue()
        
    except Exception as e:
        st.error(f"Erro ao converter template para Excel: {e}")
        return b''

def validar_template_usuario(df_usuario: pd.DataFrame, df_template: pd.DataFrame = None) -> tuple[bool, str]:
    """Valida se o template preenchido pelo usuário está correto"""
    try:
        # Verificar se tem a coluna resultado_final
        if 'resultado_final' not in df_usuario.columns:
            return False, "Coluna 'resultado_final' não encontrada no arquivo"
        
        # Verificar se resultado_final está na escala 0-10 (padrão brasileiro)
        resultado_values = df_usuario['resultado_final'].dropna()
        if len(resultado_values) > 0:
            if not pd.api.types.is_numeric_dtype(resultado_values):
                return False, "Coluna 'resultado_final' deve conter valores numéricos (0-10)"
            if (resultado_values < 0).any() or (resultado_values > 10).any():
                return False, "Valores em 'resultado_final' devem estar entre 0 e 10"
        
        # Verificar se tem a coluna nome_aluno (para template unificado)
        if 'nome_aluno' not in df_usuario.columns:
            return False, "Coluna 'nome_aluno' não encontrada no arquivo"
        
        # Se df_template foi fornecido, verificar features específicas
        if df_template is not None:
            expected_features = [col for col in df_template.columns if col not in ['resultado_final', 'nome_aluno']]
            missing_features = [col for col in expected_features if col not in df_usuario.columns]
            
            if missing_features:
                return False, f"Colunas de features esperadas não encontradas: {missing_features}"
        
        # Verificar se tem dados (não está vazio)
        if df_usuario.empty:
            return False, "Arquivo está vazio"
        
        # Verificar se tem pelo menos algumas linhas com dados válidos
        non_empty_rows = df_usuario.dropna(how='all').shape[0]
        if non_empty_rows < 3:
            return False, "Arquivo deve ter pelo menos 3 linhas com dados válidos"
        
        # Verificar se tem pelo menos algumas features além de nome e resultado
        feature_cols = [col for col in df_usuario.columns if col not in ['nome_aluno', 'resultado_final']]
        if len(feature_cols) < 2:
            return False, "Template deve ter pelo menos 2 features além de nome_aluno e resultado_final"
        
        return True, "Template válido"
        
    except Exception as e:
        return False, f"Erro na validação: {e}"

def realizar_eda_automatica(df_usuario: pd.DataFrame) -> dict:
    """Realiza EDA automática no dataset do usuário"""
    try:
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        from sklearn.preprocessing import OneHotEncoder, LabelEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report
        from sklearn.inspection import permutation_importance
        import numpy as np
        
        # Preparar dados - remover nome_aluno se existir
        target_col = 'resultado_final'
        y = df_usuario[target_col]
        X = df_usuario.drop([target_col, 'nome_aluno'], axis=1, errors='ignore')
        
        # Detectar tipo de problema (regressão vs classificação)
        # Sempre tratar como regressão se for numérico (escala 0-10)
        is_regression = pd.api.types.is_numeric_dtype(y)
        
        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Preparar preprocessamento
        categorical_features = X_train.select_dtypes(include=['object']).columns
        numerical_features = X_train.select_dtypes(include=[np.number]).columns
        
        # Criar preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ]
        )
        
        # Treinar modelo apropriado
        if is_regression:
            model = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
            ])
            y_train_encoded = y_train.astype(float)
            y_test_encoded = y_test.astype(float)
        else:
            model = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
            ])
            # Para classificação, usar LabelEncoder se necessário
            if not pd.api.types.is_numeric_dtype(y_train):
                le = LabelEncoder()
                y_train_encoded = le.fit_transform(y_train)
                y_test_encoded = le.transform(y_test)
            else:
                y_train_encoded = y_train
                y_test_encoded = y_test
        
        # Treinar modelo
        model.fit(X_train, y_train_encoded)
        
        # Fazer predições
        predictions = model.predict(X_test)
        
        # Calcular métricas
        if is_regression:
            mae = mean_absolute_error(y_test_encoded, predictions)
            rmse = np.sqrt(mean_squared_error(y_test_encoded, predictions))
            r2 = r2_score(y_test_encoded, predictions)
            metrics = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'type': 'regression'
            }
        else:
            accuracy = accuracy_score(y_test_encoded, predictions)
            metrics = {
                'accuracy': accuracy,
                'type': 'classification',
                'classification_report': classification_report(y_test_encoded, predictions, output_dict=True)
            }
        
        # Calcular feature importance
        try:
            # Usar permutation importance
            result = permutation_importance(
                model, X_test, y_test_encoded, 
                n_repeats=5, random_state=42, n_jobs=-1
            )
            
            feature_importance = pd.DataFrame({
                'feature': X_test.columns,
                'importance': result.importances_mean
            }).sort_values('importance', ascending=False)
        except:
            # Fallback para feature_importances_ do modelo
            if hasattr(model.named_steps[list(model.named_steps.keys())[-1]], 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': X_test.columns,
                    'importance': model.named_steps[list(model.named_steps.keys())[-1]].feature_importances_
                }).sort_values('importance', ascending=False)
            else:
                feature_importance = pd.DataFrame()
        
        # Estatísticas descritivas
        stats = {
            'shape': df_usuario.shape,
            'missing_values': df_usuario.isnull().sum().to_dict(),
            'dtypes': df_usuario.dtypes.to_dict(),
            'numeric_summary': df_usuario.select_dtypes(include=[np.number]).describe().to_dict() if not df_usuario.select_dtypes(include=[np.number]).empty else {},
            'categorical_summary': df_usuario.select_dtypes(include=['object']).describe().to_dict() if not df_usuario.select_dtypes(include=['object']).empty else {}
        }
        
        return {
            'model': model,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'predictions': predictions,
            'y_test': y_test_encoded,
            'stats': stats,
            'is_regression': is_regression
        }
        
    except Exception as e:
        st.error(f"Erro na EDA automática: {e}")
        return {}

def realizar_analise_completa(df_usuario: pd.DataFrame) -> dict:
    """
    Executa análise completa dos dados do usuário
    Similar às análises feitas em UCI e OULAD
    """
    try:
        resultados = {
            'eda': realizar_eda_automatica(df_usuario),
            'graficos': {},
            'metricas': {}
        }
        
        # Estatísticas descritivas
        resultados['metricas']['descritivas'] = df_usuario.describe()
        
        # Correlações
        numeric_cols = df_usuario.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            resultados['metricas']['correlacao'] = df_usuario[numeric_cols].corr()
        
        # Distribuições
        resultados['graficos']['distribuicoes'] = criar_graficos_distribuicao(df_usuario)
        
        # Gráfico radar (será criado com seleção de aluno)
        resultados['graficos']['radar'] = criar_grafico_radar_aluno(df_usuario)
        
        return resultados
        
    except Exception as e:
        st.error(f"Erro na análise completa: {e}")
        return {}

def criar_graficos_distribuicao(df_usuario: pd.DataFrame) -> dict:
    """Cria gráficos de distribuição para análise educacional"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        graficos = {}
        
        # Gráfico de distribuição de resultados
        if 'resultado_final' in df_usuario.columns:
            fig, ax = plt.subplots(figsize=(12, 8))  # Aumentado o tamanho
            
            # Criar bins para notas numéricas (escala 0-10)
            df_traduzido = df_usuario.copy()
            df_traduzido['faixa_nota'] = pd.cut(
                df_traduzido['resultado_final'], 
                bins=[0, 5, 7, 10], 
                labels=['Insuficiente (0-5)', 'Regular (5-7)', 'Bom (7-10)'],
                include_lowest=True
            )
            
            # Contar valores por faixa
            contagem_faixas = df_traduzido['faixa_nota'].value_counts()
            
            # Criar gráfico de barras
            bars = ax.bar(contagem_faixas.index, contagem_faixas.values, 
                         color=['#dc3545', '#ffc107', '#28a745'], alpha=0.8, edgecolor='black', linewidth=1)
            
            # Configurar título e labels
            ax.set_title('Distribuição de Resultados da Turma', fontsize=18, fontweight='bold', pad=20)
            ax.set_xlabel('Faixa de Nota', fontsize=14, fontweight='bold')
            ax.set_ylabel('Quantidade de Alunos', fontsize=14, fontweight='bold')
            ax.tick_params(axis='x', rotation=45, labelsize=12)
            ax.tick_params(axis='y', labelsize=12)
            
            # Adicionar valores nas barras
            for i, (bar, valor) in enumerate(zip(bars, contagem_faixas.values)):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                       str(valor), ha='center', va='bottom', fontweight='bold', fontsize=12)
            
            # Adicionar grid para melhor visualização
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_axisbelow(True)
            
            # Ajustar layout
            plt.tight_layout()
            graficos['distribuicao_resultados'] = fig
        
        return graficos
        
    except Exception as e:
        st.error(f"Erro ao criar gráficos de distribuição: {e}")
        return {}

def criar_grafico_radar_aluno(df_usuario: pd.DataFrame, nome_aluno: str = None) -> dict:
    """Cria gráfico radar comparando aluno individual com média da turma"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from math import pi
        
        graficos = {}
        
        if 'resultado_final' in df_usuario.columns and 'nome_aluno' in df_usuario.columns:
            # Se não especificado, usar o primeiro aluno como exemplo
            if nome_aluno is None:
                nome_aluno = df_usuario['nome_aluno'].iloc[0]
            
            # Verificar se o aluno existe
            aluno_data = df_usuario[df_usuario['nome_aluno'] == nome_aluno]
            if aluno_data.empty:
                st.warning(f"Aluno '{nome_aluno}' não encontrado. Usando primeiro aluno como exemplo.")
                nome_aluno = df_usuario['nome_aluno'].iloc[0]
                aluno_data = df_usuario.iloc[[0]]
            
            # Obter dados do aluno selecionado
            aluno_row = aluno_data.iloc[0]
            
            # Calcular médias da turma (excluindo o aluno selecionado)
            turma_media = df_usuario[df_usuario['nome_aluno'] != nome_aluno].mean(numeric_only=True)
            
            # Selecionar colunas numéricas para o radar (incluindo resultado_final)
            colunas_numericas = [col for col in df_usuario.select_dtypes(include=[np.number]).columns 
                               if col in aluno_row.index]
            
            if len(colunas_numericas) >= 3:  # Mínimo 3 dimensões para radar
                # Preparar dados para o radar
                valores_aluno = [aluno_row[col] for col in colunas_numericas]
                valores_turma = [turma_media[col] for col in colunas_numericas]
                
                # Normalizar valores para escala 0-10 (assumindo que as features já estão nessa escala)
                # Se não estiverem, fazer normalização simples
                max_val = max(max(valores_aluno), max(valores_turma))
                if max_val > 10:
                    valores_aluno = [v/max_val*10 for v in valores_aluno]
                    valores_turma = [v/max_val*10 for v in valores_turma]
                
                # Criar gráfico radar
                fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
                
                # Ângulos para cada dimensão
                angles = [n / float(len(colunas_numericas)) * 2 * pi for n in range(len(colunas_numericas))]
                angles += angles[:1]  # Fechar o círculo
                
                # Adicionar valores para fechar o círculo
                valores_aluno += valores_aluno[:1]
                valores_turma += valores_turma[:1]
                
                # Plotar dados
                ax.plot(angles, valores_aluno, 'o-', linewidth=2, label=f'{nome_aluno}', color='#2E86AB', markersize=8)
                ax.fill(angles, valores_aluno, alpha=0.25, color='#2E86AB')
                
                ax.plot(angles, valores_turma, 'o-', linewidth=2, label='Média da Turma', color='#A23B72', markersize=8)
                ax.fill(angles, valores_turma, alpha=0.25, color='#A23B72')
                
                # Configurar eixos com traduções
                traducao_rotulos = {
                    'nota_2bim': 'Nota 2º Bimestre',
                    'faltas': 'Faltas',
                    'pontuacao': 'Pontuação',
                    'resultado_final': 'Nota Final'
                }
                
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels([traducao_rotulos.get(col, col.replace('_', ' ').title()) for col in colunas_numericas])
                ax.set_ylim(0, 10)
                ax.set_yticks([2, 4, 6, 8, 10])
                ax.set_yticklabels(['2', '4', '6', '8', '10'])
                ax.grid(True)
                
                # Título e legenda
                ax.set_title(f'Comparação: {nome_aluno} vs Média da Turma', size=16, fontweight='bold', pad=20)
                ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
                
                # Adicionar valores nas pontas
                for i, (angle, valor_aluno, valor_turma) in enumerate(zip(angles[:-1], valores_aluno[:-1], valores_turma[:-1])):
                    ax.text(angle, valor_aluno + 0.5, f'{valor_aluno:.1f}', ha='center', va='center', fontweight='bold', color='#2E86AB')
                    ax.text(angle, valor_turma - 0.5, f'{valor_turma:.1f}', ha='center', va='center', fontweight='bold', color='#A23B72')
                
                plt.tight_layout()
                graficos['radar_comparacao_aluno'] = fig
            else:
                st.warning("Não há colunas numéricas suficientes para criar o gráfico radar.")
        
        return graficos
        
    except Exception as e:
        st.error(f"Erro ao criar gráfico radar: {e}")
        return {}

def exibir_resultados_com_ia(resultados: dict, df_usuario: pd.DataFrame):
    """Exibe resultados com interpretação automática"""
    
    st.markdown("## 📊 Resultados da Análise")
    
    # 1. Métricas Gerais
    st.markdown("### 📈 Métricas Gerais")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total de Alunos", len(df_usuario))
    with col2:
        taxa_aprovacao = (df_usuario['resultado_final'] >= 5.0).mean() * 100  # Aprovação >= 5.0
        st.metric("Taxa de Aprovação", f"{taxa_aprovacao:.1f}%")
    with col3:
        # Calcular média das faltas (se a coluna existir)
        if 'faltas' in df_usuario.columns:
            media_faltas = df_usuario['faltas'].mean()
            st.metric("Média das Faltas", f"{media_faltas:.1f}")
        else:
            st.metric("Média das Faltas", "N/A")
    with col4:
        # Calcular média das notas finais
        media_notas = df_usuario['resultado_final'].mean()
        st.metric("Média das Notas Finais", f"{media_notas:.1f}")
    
    # 2. Gráfico de Distribuição + Interpretação IA
    st.markdown("### 📊 Distribuição de Resultados")
    if 'distribuicoes' in resultados['graficos'] and 'distribuicao_resultados' in resultados['graficos']['distribuicoes']:
        fig_dist = resultados['graficos']['distribuicoes']['distribuicao_resultados']
        st.pyplot(fig_dist)
        
        # Interpretação automática
        contexto = {
            'total_alunos': len(df_usuario),
            'aprovados': (df_usuario['resultado_final'] >= 5.0).sum(),
            'reprovados': (df_usuario['resultado_final'] < 5.0).sum(),
            'media_geral': df_usuario['resultado_final'].mean()
        }
        
        from .openai_interpreter import gerar_interpretacao_traduzida
        interpretacao = gerar_interpretacao_traduzida('distribuicao_resultados', contexto)
        st.info(f"💡 **Interpretação**: {interpretacao}")
    
    # Histograma de Distribuição das Notas Finais
    st.markdown("### 📊 Histograma de Distribuição das Notas Finais")
    if 'resultado_final' in df_usuario.columns:
        fig_hist, ax_hist = plt.subplots(figsize=(12, 6))
        
        # Criar histograma com KDE
        sns.histplot(df_usuario['resultado_final'], bins=20, kde=True, ax=ax_hist, 
                    color='#3498db', alpha=0.7, edgecolor='black', linewidth=1)
        
        # Adicionar linha vertical para média
        media_notas = df_usuario['resultado_final'].mean()
        ax_hist.axvline(media_notas, color='red', linestyle='--', linewidth=2, 
                       label=f'Média: {media_notas:.2f}')
        
        # Adicionar linha vertical para mediana
        mediana_notas = df_usuario['resultado_final'].median()
        ax_hist.axvline(mediana_notas, color='orange', linestyle='--', linewidth=2, 
                       label=f'Mediana: {mediana_notas:.2f}')
        
        # Adicionar linha vertical para nota de corte (5.0)
        ax_hist.axvline(5.0, color='green', linestyle='-', linewidth=2, 
                       label='Nota de Corte (5.0)')
        
        # Configurar gráfico
        ax_hist.set_title('Distribuição das Notas Finais - Histograma', 
                         fontsize=16, fontweight='bold', pad=20)
        ax_hist.set_xlabel('Nota Final (0-10)', fontsize=14, fontweight='bold')
        ax_hist.set_ylabel('Frequência', fontsize=14, fontweight='bold')
        ax_hist.legend(fontsize=12)
        ax_hist.grid(True, alpha=0.3)
        ax_hist.set_xlim(0, 10)
        
        # Adicionar estatísticas no gráfico
        stats_text = f"""
📊 ESTATÍSTICAS:
• Média: {media_notas:.2f}
• Mediana: {mediana_notas:.2f}
• Desvio Padrão: {df_usuario['resultado_final'].std():.2f}
• Mínimo: {df_usuario['resultado_final'].min():.2f}
• Máximo: {df_usuario['resultado_final'].max():.2f}
• Aprovados (≥5.0): {((df_usuario['resultado_final'] >= 5.0).sum() / len(df_usuario) * 100):.1f}%
        """
        
        # Adicionar caixa de estatísticas
        ax_hist.text(0.02, 0.98, stats_text, transform=ax_hist.transAxes, 
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        st.pyplot(fig_hist)
        
        # Interpretação do histograma
        contexto_hist = {
            'media': media_notas,
            'mediana': mediana_notas,
            'desvio_padrao': df_usuario['resultado_final'].std(),
            'aprovados_pct': (df_usuario['resultado_final'] >= 5.0).mean() * 100,
            'distribuicao': 'normal' if abs(media_notas - mediana_notas) < 0.5 else 'assimétrica'
        }
        
        # Interpretação automática
        from .openai_interpreter import gerar_interpretacao_traduzida
        interpretacao_hist = gerar_interpretacao_traduzida('histograma_notas', contexto_hist)
        # Adicionar informações específicas do histograma
        interpretacao_hist += f"\n\nA média de {media_notas:.2f} e mediana de {mediana_notas:.2f} indicam o desempenho central. {((df_usuario['resultado_final'] >= 5.0).sum() / len(df_usuario) * 100):.1f}% dos alunos foram aprovados."
        st.info(f"💡 **Interpretação**: {interpretacao_hist}")
    
    # 3. Gráficos de Distribuição Numérica
    st.markdown("### 📊 Distribuições Numéricas")
    
    # Criar gráficos de distribuição
    graficos_distribuicao = criar_graficos_distribuicao_numerica(df_usuario)
    
    if graficos_distribuicao:
        col1, col2 = st.columns(2)
        
        with col1:
            if 'distribuicao_faltas' in graficos_distribuicao:
                st.pyplot(graficos_distribuicao['distribuicao_faltas'])
                
                # Interpretação das faltas
                from .openai_interpreter import gerar_interpretacao_traduzida
                contexto_faltas = {
                    'media_faltas': df_usuario['faltas'].mean() if 'faltas' in df_usuario.columns else 0,
                    'total_alunos': len(df_usuario)
                }
                interpretacao = gerar_interpretacao_traduzida('distribuicao_faltas', contexto_faltas)
                st.info(f"💡 **Interpretação**: {interpretacao}")
        
        with col2:
            if 'distribuicao_nota_2bim' in graficos_distribuicao:
                st.pyplot(graficos_distribuicao['distribuicao_nota_2bim'])
                
                # Interpretação da nota do 2º bimestre
                from .openai_interpreter import gerar_interpretacao_traduzida
                contexto_nota = {
                    'media_nota_2bim': df_usuario['nota_2bim'].mean() if 'nota_2bim' in df_usuario.columns else 0,
                    'total_alunos': len(df_usuario)
                }
                interpretacao = gerar_interpretacao_traduzida('distribuicao_nota_2bim', contexto_nota)
                st.info(f"💡 **Interpretação**: {interpretacao}")
    
    # 4. Gráfico de Linhas - Análise por Região
    st.markdown("### 📊 Análise por Região - Média das Notas Finais")
    grafico_linhas = criar_grafico_barras_empilhadas(df_usuario)
    if grafico_linhas:
        st.pyplot(grafico_linhas)
        
        # Interpretação do gráfico de linhas
        from .openai_interpreter import gerar_interpretacao_traduzida
        contexto_linhas = {
            'regioes': df_usuario['regiao'].unique().tolist() if 'regiao' in df_usuario.columns else [],
            'total_alunos': len(df_usuario),
            'media_geral': df_usuario['resultado_final'].mean()
        }
        interpretacao = gerar_interpretacao_traduzida('grafico_linhas_regiao', contexto_linhas)
        st.info(f"💡 **Interpretação**: {interpretacao}")
    
    # 5. Gráfico Radar - Comparação Individual
    st.markdown("### 🎯 Análise Individual - Gráfico Radar")
    
    # Campo de busca para seleção do aluno
    if 'nome_aluno' in df_usuario.columns:
        nomes_alunos = sorted(df_usuario['nome_aluno'].unique().tolist())  # Ordem alfabética
        nome_selecionado = st.selectbox(
            "Selecione o aluno para análise:",
            options=nomes_alunos,
            index=0,
            help="Escolha um aluno para comparar com a média da turma",
            key="selectbox_aluno_radar"  # Chave única para evitar duplicação
        )
        
        # Criar gráfico radar para o aluno selecionado
        grafico_radar = criar_grafico_radar_aluno(df_usuario, nome_selecionado)
        
        if 'radar_comparacao_aluno' in grafico_radar:
            st.pyplot(grafico_radar['radar_comparacao_aluno'])
            
            # Interpretação do gráfico radar
            from .openai_interpreter import gerar_interpretacao_traduzida
            contexto_radar = {
                'nome_aluno': nome_selecionado,
                'total_alunos': len(df_usuario),
                'media_turma': df_usuario['resultado_final'].mean()
            }
            interpretacao = gerar_interpretacao_traduzida('radar_comparacao', contexto_radar)
            # Adicionar informações específicas do aluno
            interpretacao = interpretacao.replace("do aluno selecionado", f"de {nome_selecionado}")
            st.info(f"💡 **Interpretação**: {interpretacao}")
        else:
            st.warning("Não foi possível criar o gráfico radar para este aluno.")
    else:
        st.warning("Coluna 'nome_aluno' não encontrada nos dados.")
    
    # 5. Tabela de Dados
    st.markdown("### 📋 Dados Completos da Turma")
    st.dataframe(df_usuario, use_container_width=True)

def criar_grafico_correlacao_traduzido(corr_matrix: pd.DataFrame):
    """Cria heatmap de correlação com rótulos traduzidos"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Traduzir rótulos das colunas
    traducao_colunas = {
        'faltas': 'Faltas',
        'nota_2bim': 'Nota 2º Bimestre',
        'cliques_plataforma': 'Cliques na Plataforma',
        'pontuacao_atividades': 'Pontuação nas Atividades'
    }
    
    # Renomear colunas
    corr_traduzida = corr_matrix.rename(columns=traducao_colunas, index=traducao_colunas)
    
    # Criar gráfico
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_traduzida, 
        annot=True, 
        cmap='RdYlBu_r', 
        center=0,
        square=True,
        fmt='.2f',
        cbar_kws={'label': 'Força da Relação'}
    )
    
    ax.set_title('Relação entre Fatores de Desempenho', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    return fig

def encontrar_top_correlacoes(corr_matrix: pd.DataFrame) -> dict:
    """Encontra as top correlações para interpretação"""
    try:
        # Remover diagonal (correlação de 1.0 com si mesmo)
        corr_no_diag = corr_matrix.where(~np.eye(len(corr_matrix), dtype=bool))
        
        # Encontrar correlações mais altas
        top_corr = corr_no_diag.stack().nlargest(3)
        
        return {
            'top_correlacoes': top_corr.to_dict(),
            'num_features': len(corr_matrix.columns)
        }
    except:
        return {'top_correlacoes': {}, 'num_features': 0}

def criar_graficos_distribuicao_numerica(df_usuario: pd.DataFrame) -> dict:
    """Cria gráficos de distribuição otimizados para análise educacional"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        import pandas as pd
        
        graficos = {}
        
        # 1. GRÁFICO DE FALTAS - Linha (Faltas vs Nota Final) + Violin Plot
        if 'faltas' in df_usuario.columns and 'resultado_final' in df_usuario.columns:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Gráfico de Linha - Relação entre faltas e nota final
            # Agrupar por número de faltas e calcular média das notas
            df_agrupado = df_usuario.groupby('faltas', observed=False)['resultado_final'].agg(['mean', 'count']).reset_index()
            df_agrupado = df_agrupado[df_agrupado['count'] >= 1]  # Pelo menos 1 aluno
            
            # Plotar linha principal
            ax1.plot(df_agrupado['faltas'], df_agrupado['mean'], 
                    marker='o', linewidth=3, markersize=8, 
                    color='#e74c3c', alpha=0.8, label='Média das Notas')
            
            # Adicionar pontos individuais (transparentes para mostrar densidade)
            ax1.scatter(df_usuario['faltas'], df_usuario['resultado_final'], 
                       alpha=0.3, s=30, color='#3498db', label='Alunos individuais')
            
            # Linha de tendência
            z = np.polyfit(df_usuario['faltas'], df_usuario['resultado_final'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(df_usuario['faltas'].min(), df_usuario['faltas'].max(), 100)
            ax1.plot(x_trend, p(x_trend), '--', color='#2c3e50', linewidth=2, alpha=0.7,
                    label=f'Tendência (R²={np.corrcoef(df_usuario["faltas"], df_usuario["resultado_final"])[0,1]**2:.2f})')
            
            ax1.set_title('Relação: Faltas vs Nota Final', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Número de Faltas', fontsize=12)
            ax1.set_ylabel('Nota Final', fontsize=12)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 10)
            
            # Adicionar valores nos pontos principais
            for i, row in df_agrupado.iterrows():
                if row['count'] > 1:  # Só mostrar valores onde há múltiplos alunos
                    ax1.annotate(f'{row["mean"]:.1f}\n(n={int(row["count"])})', 
                               (row['faltas'], row['mean']), 
                               textcoords="offset points", 
                               xytext=(0,15), 
                               ha='center', 
                               fontsize=9, 
                               fontweight='bold',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
            # Violin Plot - Mostra densidade da distribuição
            violin_parts = ax2.violinplot([df_usuario['faltas']], positions=[1], 
                                         showmeans=True, showmedians=True)
            violin_parts['bodies'][0].set_facecolor('#ff6b6b')
            violin_parts['bodies'][0].set_alpha(0.7)
            ax2.set_title('Densidade de Faltas - Violin Plot', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Número de Faltas', fontsize=12)
            ax2.set_xticks([1])
            ax2.set_xticklabels(['Faltas'])
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            graficos['distribuicao_faltas'] = fig
        
        # 2. GRÁFICO DE NOTA 2º BIMESTRE - Scatter Plot + Regressão
        if 'nota_2bim' in df_usuario.columns and 'resultado_final' in df_usuario.columns:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Scatter Plot - Relação entre nota 2º bimestre e nota final
            scatter = ax1.scatter(df_usuario['nota_2bim'], df_usuario['resultado_final'], 
                                alpha=0.6, c='#4ecdc4', s=50, edgecolors='black', linewidth=0.5)
            
            # Linha de regressão
            z = np.polyfit(df_usuario['nota_2bim'], df_usuario['resultado_final'], 1)
            p = np.poly1d(z)
            ax1.plot(df_usuario['nota_2bim'], p(df_usuario['nota_2bim']), 
                    "r--", alpha=0.8, linewidth=2, label=f'Tendência (R²={np.corrcoef(df_usuario["nota_2bim"], df_usuario["resultado_final"])[0,1]**2:.2f})')
            
            ax1.set_title('Relação: Nota 2º Bimestre vs Nota Final', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Nota do 2º Bimestre', fontsize=12)
            ax1.set_ylabel('Nota Final', fontsize=12)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Gráfico de Pizza com Insights Estatísticos
            # Criar categorias para notas do 2º bimestre
            df_usuario['categoria_2bim'] = pd.cut(
                df_usuario['nota_2bim'], 
                bins=[0, 5, 7, 10], 
                labels=['Insuficiente (0-5)', 'Regular (5-7)', 'Bom (7-10)'],
                include_lowest=True
            )
            
            contagem_categorias = df_usuario['categoria_2bim'].value_counts()
            cores_categorias = ['#e74c3c', '#f39c12', '#2ecc71']  # Vermelho, laranja, verde
            
            # Calcular percentuais e estatísticas
            total_alunos = len(df_usuario)
            percentuais = (contagem_categorias / total_alunos * 100).round(1)
            
            # Criar gráfico de pizza
            wedges, texts, autotexts = ax2.pie(contagem_categorias.values, 
                                              labels=contagem_categorias.index,
                                              colors=cores_categorias,
                                              autopct='%1.1f%%',
                                              startangle=90,
                                              explode=(0.05, 0.05, 0.05),  # Separar fatias
                                              textprops={'fontsize': 10, 'fontweight': 'bold'})
            
            # Melhorar aparência dos textos
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(11)
            
            ax2.set_title('Distribuição por Categoria - 2º Bimestre\n(Com Insights Estatísticos)', 
                         fontsize=14, fontweight='bold', pad=20)
            
            # Adicionar caixa de estatísticas
            stats_text = f"""
📊 ESTATÍSTICAS:
• Total de Alunos: {total_alunos}
• Insuficiente: {contagem_categorias.get('Insuficiente (0-5)', 0)} ({percentuais.get('Insuficiente (0-5)', 0):.1f}%)
• Regular: {contagem_categorias.get('Regular (5-7)', 0)} ({percentuais.get('Regular (5-7)', 0):.1f}%)
• Bom: {contagem_categorias.get('Bom (7-10)', 0)} ({percentuais.get('Bom (7-10)', 0):.1f}%)

🎯 INSIGHTS:
• Taxa de Aprovação: {((contagem_categorias.get('Regular (5-7)', 0) + contagem_categorias.get('Bom (7-10)', 0)) / total_alunos * 100):.1f}%
• Necessita Intervenção: {contagem_categorias.get('Insuficiente (0-5)', 0)} alunos
            """
            
            # Adicionar texto de estatísticas ao lado do gráfico
            ax2.text(1.3, 0.5, stats_text, transform=ax2.transAxes, 
                    fontsize=9, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
            
            plt.tight_layout()
            graficos['distribuicao_nota_2bim'] = fig
        
        return graficos
        
    except Exception as e:
        st.error(f"Erro ao criar gráficos de distribuição numérica: {e}")
        return {}

def criar_grafico_barras_empilhadas(df_usuario: pd.DataFrame):
    """Cria gráfico de linhas mostrando média das notas finais por região e categoria de faltas"""
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
        
        # Verificar se as colunas necessárias existem
        if 'regiao' not in df_usuario.columns or 'resultado_final' not in df_usuario.columns:
            return None
        
        # Preparar dados
        df_plot = df_usuario.copy()
        
        # Criar categorias para faltas
        if 'faltas' in df_usuario.columns:
            df_plot['categoria_faltas'] = pd.cut(
                df_plot['faltas'], 
                bins=[0, 2, 5, 10], 
                labels=['Baixas (0-2)', 'Médias (2-5)', 'Altas (5+)'],
                include_lowest=True
            )
        else:
            df_plot['categoria_faltas'] = 'N/A'
        
        # Calcular média das notas finais por região e categoria de faltas
        media_por_categoria = df_plot.groupby(['regiao', 'categoria_faltas'], observed=False)['resultado_final'].mean().unstack(fill_value=0)
        
        # Criar gráfico de linhas
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Cores e estilos para as linhas
        cores = ['#2ecc71', '#f39c12', '#e74c3c']  # Verde, laranja, vermelho
        estilos = ['-', '--', '-.']  # Linha sólida, tracejada, ponto-traço
        marcadores = ['o', 's', '^']  # Círculo, quadrado, triângulo
        
        # Plotar linhas para cada categoria de faltas
        for i, categoria in enumerate(media_por_categoria.columns):
            if categoria in media_por_categoria.columns and not media_por_categoria[categoria].isna().all():
                ax.plot(media_por_categoria.index, media_por_categoria[categoria], 
                       color=cores[i % len(cores)], 
                       linestyle=estilos[i % len(estilos)],
                       marker=marcadores[i % len(marcadores)],
                       linewidth=3, markersize=8, alpha=0.8,
                       label=f'{categoria} (Média: {media_por_categoria[categoria].mean():.1f})')
        
        # Configurar gráfico
        ax.set_title('Média das Notas Finais por Região e Categoria de Faltas', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Região', fontsize=14, fontweight='bold')
        ax.set_ylabel('Média das Notas Finais (0-10)', fontsize=14, fontweight='bold')
        ax.legend(title='Categoria de Faltas', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Configurar eixo Y para escala 0-10
        ax.set_ylim(0, 10)
        ax.set_yticks(range(0, 11, 2))
        
        # Rotacionar labels do eixo x se necessário
        plt.xticks(rotation=45, ha='right')
        
        # Adicionar valores nos pontos
        for categoria in media_por_categoria.columns:
            if categoria in media_por_categoria.columns:
                for i, regiao in enumerate(media_por_categoria.index):
                    valor = media_por_categoria.loc[regiao, categoria]
                    if not pd.isna(valor) and valor > 0:
                        ax.annotate(f'{valor:.1f}', 
                                 (i, valor), 
                                 textcoords="offset points", 
                                 xytext=(0,10), 
                                 ha='center', 
                                 fontsize=9, 
                                 fontweight='bold',
                                 bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        st.error(f"Erro ao criar gráfico de linhas: {e}")
        return None
