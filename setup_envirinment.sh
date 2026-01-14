#!/bin/bash
# Script para configurar o ambiente em qualquer máquina

echo "🚀 Configurando o sistema Clareia..."

# Verificar Python 3.9+
python3 --version
if [ $? -ne 0 ]; then
    echo "❌ Python 3.9+ é necessário"
    exit 1
fi

# Criar ambiente virtual
echo "📦 Criando ambiente virtual..."
python3 -m venv venv

# Ativar ambiente virtual
echo "🔧 Ativando ambiente virtual..."
source venv/bin/activate

# Atualizar pip
echo "⬆️ Atualizando pip..."
pip install --upgrade pip

# Instalar dependências
echo "📚 Instalando dependências..."
pip install -r requirements.txt

# Verificar instalação
echo "✅ Verificando instalação..."
python -c "import streamlit, pandas, numpy, sklearn; print('Todas as dependências instaladas!')"

echo "🎉 Ambiente configurado com sucesso!"
echo "Para executar: source venv/bin/activate && streamlit run webapp/home_1.py"