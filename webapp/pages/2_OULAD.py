from pathlib import Path
import streamlit as st
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pickle
from src.openai_interpreter import criar_rodape_sidebar


from src.utilidades import carregar_dados_oulad_cached

st.set_page_config(
    page_title="Análise Exploratória dos Dados - OULAD",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Carregar dados usando a função centralizada e resiliente
try:
    merged_df = carregar_dados_oulad_cached()
    st.session_state['merged_df'] = merged_df
except Exception as e:
    st.error(f"❌ Erro crítico: Não foi possível carregar os dados OULAD. {e}")
    st.stop()

# Sidebar com rodapé
with st.sidebar:
    st.markdown("### 📊 Informações")
    st.info("""
    Esta página apresenta uma análise exploratória dos dados do OULAD (Open University Learning Analytics Dataset).
    """)
    st.markdown("---")
    # Rodapé com badges de status (igual ao da home)
    criar_rodape_sidebar()

st.write('# Análise Exploratória de Dados (EDA) - OULAD')

'''
Esta página apresenta uma Análise Exploratória dos Dados do OULAD (Open University Learning Analytics Dataset), com foco em entender o perfil dos estudantes, suas atividades na plataforma e fatores que influenciam o desempenho acadêmico . Através de visualizações, são identificados padrões relevantes, como a predominância de estudantes do gênero masculino e a distribuição etária dos estudantes.
'''

st.markdown("## Descrição estatísticas das colunas numéricas:")
st.dataframe(merged_df.select_dtypes('number').describe().T.round(2))

'''
A grande diferença entre a mediana (≈2) e a média (≈4.65) do número de cliques indica que a maioria dos estudantes tem engajamento moderado, mas uma pequena parcela é extremamente ativa, elevando a média geral.

O número de tentativas anteriores é zero para a vasta maioria dos estudantes (quartis e valor máximo são 0), sugerindo que o conjunto de dados está focado na performance na primeira tentativa.
'''


st.write('## Distribuição das notas finais dos estudantes')
plt.figure(figsize=(10, 6))
# Calcular nota média por estudante único
if 'score' in merged_df.columns and 'id_student' in merged_df.columns:
    notas_por_estudante = merged_df.groupby('id_student')['score'].mean()
    sns.histplot(notas_por_estudante, bins=30, kde=True)
    plt.title('Distribuição de Notas Finais dos Estudantes (Únicos)')
    plt.xlabel('Nota Final Média')
    plt.ylabel('Número de Estudantes Únicos')
else:
    sns.histplot(merged_df['score'], bins=30, kde=True)
    plt.title('Distribuição de Notas Finais dos Estudantes')
    plt.xlabel('Nota Final')
    plt.ylabel('Frequência')
st.pyplot(plt)
plt.clf()

'''
Com base no histograma, a maioria dos estudantes obteve notas finais elevadas, concentrando-se principalmente na faixa de 70 a 90. Há uma distribuição que parece ser bimodal ou multimodal, com picos notáveis e uma frequência menor de notas mais baixas.
'''

st.write('## Distribuição de Atividades por Tipo')
plt.figure(figsize=(10, 6))
# Dicionário de tradução dos tipos de atividades
traducao_atividades = {
    'outcontent': 'Conteúdo Externo',
    'forumng': 'Fórum NG',
    'subpage': 'Subpágina',
    'resource': 'Recurso',
    'url': 'URL',
    'homepage': 'Página Inicial',
    'quiz': 'Quiz',
    'ouwiki': 'Wiki da Open University',
    'dataplus': 'DataPlus',
    'glossary': 'Glossário',
    'htmlactivity': 'Atividade HTML',
    'questionnaire': 'Questionário',
    'page': 'Página',
    'folder': 'Pasta',
    '   llaborate': 'Atividades Colaborativas',
    'dualpane': 'Painel Duplo',
    'repeatactivity': 'Atividade Repetida',
    'sharedsubpage': 'Subpágina Compartilhada'
}

# Contar atividades únicas por tipo (não estudantes únicos, pois é sobre atividades)
atividade_counts = merged_df['activity_type'].value_counts()
# Traduzir os índices (tipos de atividades) - criar novo Series com índices traduzidos
atividades_traduzidas = [traducao_atividades.get(x, x) for x in atividade_counts.index]
atividade_counts_traduzido = pd.Series(atividade_counts.values, index=atividades_traduzidas)
sns.barplot(x=atividade_counts_traduzido.index, y=atividade_counts_traduzido.values)
plt.title('Distribuição de Atividades por Tipo')
plt.xlabel('Tipo de Atividade')
plt.ylabel('Número de Atividades')
plt.xticks(rotation=45)
st.pyplot(plt)
plt.clf()

'''
A atividade mais realizada é a 'Conteúdo Externo' com quase o dobro de execuções em relação à segunda posição que é 'Fórum NG'. A distribuição é acentuadamente desigual, com poucas atividades (como "Fórum NG" e "Subpágina") tendo uso moderado.
'''


st.markdown('## Explorando valores categóricos')
## Explorando valores categóricos
st.dataframe(merged_df.select_dtypes('object').describe().T)

"""
Por meio da análise dos dados categóricos, os estudantes são, na sua maioria, do gênero masculino, até 35 anos, que realizaram a atividade do tipo fórum na plataforma e foram aprovados.
"""

col1, col2 = st.columns(2)

with col1:
    st.write('## Distribuição de Estudantes por Idade')
    # Contar estudantes únicos por faixa etária
    idade_counts = merged_df.groupby('age_band')['id_student'].nunique()
    fig_idade, ax_idade = plt.subplots(figsize=(6, 4))
    sns.barplot(x=idade_counts.index, y=idade_counts.values, ax=ax_idade)
    ax_idade.set_title('Distribuição de Estudantes por Idade')
    ax_idade.set_xlabel('Faixa Etária')
    ax_idade.set_ylabel('Número de Estudantes Únicos')
    ax_idade.tick_params(axis='x', rotation=45)
    st.pyplot(fig_idade)

    '''
    Este histograma revela que a maioria dos estudantes se encontra na faixa etária de 35 a 55 anos e a faixa etária dentro do grupo 0-35 é o segundo maior contingente, enquanto estudantes com mais de 55 anos são a minoria.
    '''

with col2:
    st.write('## Distribuição de Estudantes por Gênero')
    # Contar estudantes únicos por gênero
    genero_counts = merged_df.groupby('gender')['id_student'].nunique()
    fig_genero, ax_genero = plt.subplots(figsize=(6, 4))
    sns.barplot(x=genero_counts.index, y=genero_counts.values, ax=ax_genero)
    ax_genero.set_title('Distribuição de Estudantes por Gênero')
    ax_genero.set_xlabel('Gênero')
    ax_genero.set_ylabel('Número de Estudantes Únicos')
    st.pyplot(fig_genero)

    # Espaço extra para alinhar com o texto do gráfico ao lado
    st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)

    '''
    A diferença na quantidade entre os gêneros masculino e feminino é algo em torno de 60% 
    '''

st.write('## Distribuição de Estudantes por Região')
plt.figure(figsize=(10, 6))
# Dicionário de tradução das regiões
traducao_regioes = {
    'East Anglian Region': 'Região de East Anglia',
    'East Midlands Region': 'Região dos Midlands Orientais',
    'Ireland': 'Irlanda',
    'London Region': 'Região de Londres',
    'North Region': 'Região Norte',
    'North East Region': 'Região Nordeste',
    'North Western Region': 'Região Noroeste',
    'North West Region': 'Região Noroeste',
    'Scotland': 'Escócia',
    'South East Region': 'Região Sudeste',
    'South Region': 'Região Sul',
    'South West Region': 'Região Sudoeste',
    'Wales': 'País de Gales',
    'West Midlands Region': 'Região dos Midlands Ocidentais',
    'Yorkshire and The Humber Region': 'Região de Yorkshire e Humber',
    'Yorkshire and the Humber Region': 'Região de Yorkshire e Humber'
}

# Contar estudantes únicos por região
regiao_counts = merged_df.groupby('region')['id_student'].nunique().sort_values(ascending=False)
# Traduzir os índices (regiões) - criar novo Series com índices traduzidos
regioes_traduzidas = [traducao_regioes.get(x, x) for x in regiao_counts.index]
regiao_counts_traduzido = pd.Series(regiao_counts.values, index=regioes_traduzidas)
sns.barplot(x=regiao_counts_traduzido.index, y=regiao_counts_traduzido.values)
plt.title('Distribuição de Estudantes por Região')
plt.xlabel('Região')
plt.ylabel('Número de Estudantes Únicos')
plt.xticks(rotation=45)
st.pyplot(plt)
plt.clf()

"""
As regiões do sudeste sul detêm a maior concentração de estudantes, pode ter relação com a presença de importantes universidades na região: Universidade de Cambridge, Universidade de Essex, Universidade de Artes de Norwich, entre outras.
A distribuição é relativamente decrescente e sem discrepâncias abruptas.
"""

st.write('## Distribuição dos Estudantes por Resultado Final')
# Tamanho reduzido para evitar ocupar toda a largura
fig_resultado, ax_resultado = plt.subplots(figsize=(6, 4))
# Dicionário de tradução dos resultados finais
traducao_resultados = {
    'Pass': 'Aprovado',
    'Distinction': 'Aprovação com Mérito',
    'Withdrawn': 'Desistente',
    'Fail': 'Reprovado'
}

# Contar estudantes únicos por resultado final
resultado_counts = merged_df.groupby('final_result')['id_student'].nunique().sort_values(ascending=False)
# Traduzir os índices (resultados) - criar novo Series com índices traduzidos
resultados_traduzidos = [traducao_resultados.get(x, x) for x in resultado_counts.index]
resultado_counts_traduzido = pd.Series(resultado_counts.values, index=resultados_traduzidos)
sns.barplot(x=resultado_counts_traduzido.index, y=resultado_counts_traduzido.values, ax=ax_resultado)
ax_resultado.set_title('Distribuição dos Estudantes por Resultado Final')
ax_resultado.set_xlabel('Resultado Final')
ax_resultado.set_ylabel('Número de Estudantes Únicos')
st.pyplot(fig_resultado)

'''
A grande maioria dos estudantes obteve o resultado "Aprovado", superando vastamente as outras categorias. Os resultados de "Aprovação com Mérito", "Desistente" e "Reprovado" representam uma proporção muito menor do total de alunos, indicando uma alta taxa de sucesso geral.
'''

st.markdown('## Analisando  a importância das classes (feature importance)')

st.markdown("Preparação dos dados para modelos de ML...")
Y = merged_df['final_result']
X = merged_df.loc[:, merged_df.columns != 'final_result']

st.markdown('Removendo as classes irrelevantes ou com alta cardinalidade...')
X = X.drop(['id_student', 'id_site', 'id_assessment', 'code_module', 'code_presentation', 'code_module_y', 'code_module_x'], axis=1)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

@st.cache_data(ttl=7200)  # Cache por 2 horas
def treinar_modelo_oulad(X_train, y_train):
    """Treina o modelo OULAD com cache"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    import pandas as pd
    
    # Drop rows with NaN in y_train
    nan_rows_train = y_train.isnull()
    X_train_cleaned = X_train[~nan_rows_train].copy()
    y_train_cleaned = y_train[~nan_rows_train].copy()
    
    # Identify categorical and numerical columns
    categorical_cols = X_train_cleaned.select_dtypes(include=['object', 'category']).columns
    numerical_cols = X_train_cleaned.select_dtypes(include=np.number).columns
    
    # Create a column transformer to apply different preprocessing steps to different column types
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='mean'), numerical_cols),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_cols)
        ],
        remainder='passthrough' # Keep other columns (numeric) as they are
    )
    
    # Create a pipeline that first preprocesses the data and then trains the model
    ml_model = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', RandomForestClassifier(n_estimators=50, n_jobs=2, max_depth=4, random_state=42))])
    
    # Train the model
    ml_model.fit(X_train_cleaned, y_train_cleaned)
    return ml_model

ml_model = treinar_modelo_oulad(X_train, y_train)

st.markdown("Modelo treinado com sucesso!")
st.markdown("Avaliando do modelo...")

predictions = ml_model.predict(X_test)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Drop rows with NaN in y_test
nan_rows_test = y_test.isnull()
X_test_cleaned = X_test[~nan_rows_test].copy()
y_test_cleaned = y_test[~nan_rows_test].copy()
predictions_cleaned = ml_model.predict(X_test_cleaned)

# Exibir métricas do modelo
st.markdown("### Métricas de Avaliação do Modelo")

# Calcular métricas individuais
accuracy = accuracy_score(y_test_cleaned, predictions_cleaned)
precision = precision_score(y_test_cleaned, predictions_cleaned, average='weighted', zero_division=0)
recall = recall_score(y_test_cleaned, predictions_cleaned, average='weighted', zero_division=0)
f1 = f1_score(y_test_cleaned, predictions_cleaned, average='weighted', zero_division=0)

# Criar tabela com as métricas
metricas_df = pd.DataFrame({
    'Métrica': ['Acurácia', 'Precisão (weighted)', 'Recall (weighted)', 'F1-Score (weighted)'],
    'Valor': [accuracy, precision, recall, f1]
})
metricas_df['Valor'] = metricas_df['Valor'].round(4)
st.dataframe(metricas_df, use_container_width=True, hide_index=True)

from sklearn.inspection import permutation_importance

result = permutation_importance(ml_model, X_test_cleaned, y_test_cleaned, n_repeats=10, random_state=42, n_jobs=2)
sorted_idx = result.importances_mean.argsort()

# Pegar apenas as top 5 features mais importantes (ordenadas da mais importante para a menos importante)
top_5_idx = sorted_idx[-5:][::-1]  # Reverter para ter a mais importante primeiro
top_5_features = X_test_cleaned.columns[top_5_idx]
top_5_importances = result.importances_mean[top_5_idx]

# Traduzir nomes das variáveis para exibição
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
top_5_features_pt = [feature_translation.get(f, f) for f in top_5_features]

# Criar gráfico de barras
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(range(len(top_5_features)), top_5_importances)
ax.set_yticks(range(len(top_5_features)))
ax.set_yticklabels(top_5_features_pt)
ax.set_xlabel('Importância por Permutação')
ax.set_title('Top 5 Variáveis Mais Importantes (OULAD)')
ax.invert_yaxis()  # Mostrar a feature mais importante no topo
fig.tight_layout()
st.pyplot(fig)
st.markdown(
    "Histograma de importância das variáveis (método de permutação). "
    "Valores mais altos indicam maior impacto na previsão do resultado final."
)
plt.clf()

st.markdown("## Conclusão")
st.markdown("Nesta análise exploratória dos dados do OULAD, conseguimos entender melhor o perfil dos estudantes, suas atividades na plataforma e os fatores que influenciam seu desempenho acadêmico. Através da visualização dos dados, identificamos padrões interessantes, como a predominância de estudantes do gênero masculino e a distribuição etária dos participantes. Além disso, o treinamento do modelo de aprendizado de máquina nos permitiu avaliar a importância das diferentes características dos dados, destacando quais fatores têm maior impacto no resultado final dos estudantes. Essas informações são valiosas para instituições educacionais que buscam melhorar a experiência de aprendizagem e o suporte oferecido aos alunos. Futuras análises podem aprofundar ainda mais esses insights, explorando outras variáveis e utilizando técnicas avançadas de modelagem preditiva.")

with open('oulad.pkl', 'wb') as f:
    pickle.dump(ml_model, f)
    f.close()