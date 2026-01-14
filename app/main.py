
from .data import create_data
from sklearn.model_selection import train_test_split


from .prerequisite_issues import identify_prerequisite_issues
import streamlit as st
import pandas as pd

def main():
    st.title("Clareia - Análise de Dados Educacionais")
    
    # Criando o DataFrame
    df, pre_reqs = create_data()

    # Identificando os pré-requisitos que os alunos precisam melhorar
    recommendations, metrics_summary = identify_prerequisite_issues(df, pre_reqs)

    # Exibir os resultados
    st.subheader("Recomendações por Aluno")
    for aluno, recs in recommendations.items():
        st.write(f"**{aluno}:**")
        for prereq, importance in recs[:3]:  # Mostrar apenas os 3 mais importantes
            st.write(f"- {prereq}: {importance:.3f}")

    st.subheader("Resumo das Métricas dos Modelos")
    for subject, metrics in metrics_summary.items():
        st.write(f"**{subject}:**")
        for model_name, model_metrics in metrics.items():
            st.write(f"- {model_name}: R² = {model_metrics['R²']:.3f}, MAE = {model_metrics['MAE']:.3f}")

    # Exporta o DataFrame para um arquivo CSV 
    df.to_csv('output1.csv', index=False, encoding='utf-8')
    st.success("Arquivo CSV 'output1.csv' criado com sucesso.")

    # Visualizações
    st.subheader("Visualizações dos Dados")
    
    # Gráfico de barras das notas médias por disciplina
    st.write("**Notas Médias por Disciplina:**")
    mean_scores = df.drop('Aluno', axis=1).mean()
    st.bar_chart(mean_scores)
    
    # Tabela com os dados
    st.subheader("Dados dos Alunos")
    st.dataframe(df)

if __name__ == "__main__":
    main()

