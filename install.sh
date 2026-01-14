#!/bin/bash
# Script para instalar dependências em diferentes sistemas

echo "Instalando dependências do Clareia..."

# Verificar se Python 3.9+ está instalado
python3 --version

# Criar ambiente virtual se não existir
if [ ! -d "venv" ]; then
    echo "Criando ambiente virtual..."
    python3 -m venv venv
fi

# Ativar ambiente virtual
source venv/bin/activate

# Atualizar pip
pip install --upgrade pip

# Instalar dependências
pip install -r requirements.txt

echo "Instalação concluída!"
echo "Para executar o projeto:"
echo "1. source venv/bin/activate"
echo "2. streamlit run webapp/home_1.py"