# Guia de Configuração do Clareia

## Pré-requisitos
- Python 3.9 ou superior
- Git

## Instalação Rápida
```bash
# Clonar o repositório
git clone <url-do-repositorio>
cd sida

# Executar script de setup
chmod +x setup_environment.sh
./setup_environment.sh

# Ativar ambiente virtual
source venv/bin/activate

# Executar aplicação
streamlit run webapp/home_1.py
```

## Estrutura do Projeto
- `webapp/` - Aplicação Streamlit principal
- `datasets/` - Datasets UCI e OULAD
- `data/` - Dados processados
- `app/` - Configurações da aplicação Flask
- `docs/` - Documentação técnica

## Comandos Úteis

### Execução Local
- Executar: `streamlit run webapp/home_1.py`
- Testes: `pytest`

### Docker (Recomendado)
```bash
# Usando Docker Compose (mais fácil)
docker-compose up -d

# Ou usando Docker diretamente
docker build -t sida . && docker run -p 8501:8501 sida
```

### Docker Compose - Opções Avançadas
```bash
# Executar em modo desenvolvimento (com hot-reload)
docker-compose --profile dev up -d

# Parar todos os serviços
docker-compose down

# Reconstruir e executar
docker-compose up --build -d

# Ver logs
docker-compose logs -f sida
```