import pandas as pd
from pathlib import Path
import pickle
import os

def is_lfs_pointer(file_path):
    """Verifica se um arquivo é um ponteiro do Git LFS"""
    try:
        if not os.path.exists(file_path):
            return False
            
        # Pointers are small
        if os.path.getsize(file_path) > 1024:
            return False
            
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            first_line = f.readline()
            return first_line.startswith('version https://git-lfs.github.com/spec/v1')
    except:
        return False

def carregar_uci_dados(pickle_path: str = "../uci_dataframe.pkl") -> pd.DataFrame:
    """Carrega dados UCI processados do arquivo pickle ou carrega dados brutos se não encontrar"""
    # Tentar diferentes caminhos para o arquivo pickle
    possible_paths = [
        pickle_path,
        f"../{pickle_path}",
        f"../../{pickle_path}",
        Path(__file__).parent.parents[1] / "uci_dataframe.pkl",
        Path(__file__).parent.parents[1] / "uci.pkl",
        Path.cwd() / "uci_dataframe.pkl",
        Path.cwd() / "uci.pkl"
    ]
    
    df = None
    for path in possible_paths:
        p = Path(path)
        if p.is_file() and p.stat().st_size > 1000:  # Verificar se arquivo não está vazio (mais de 1KB)
            try:
                with p.open("rb") as f:
                    df = pickle.load(f)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    print(f"✅ Dados UCI carregados do pickle: {p} ({df.shape})")
                    break
                else:
                    print(f"⚠️ Pickle {p} está vazio ou inválido")
                    df = None
            except Exception as e:
                print(f"⚠️ Erro ao carregar pickle {p}: {e}")
                continue
    
    # Se não encontrou o pickle válido, tentar carregar dados brutos
    if df is None or df.empty:
        print("🔄 Arquivo pickle não encontrado ou inválido. Carregando dados brutos UCI...")
        try:
            df = carregar_dados_uci_raw()
            print(f"✅ Dados UCI brutos carregados: {df.shape}")
            
            # Tentar salvar o pickle para próximas execuções
            try:
                pickle_path_final = Path(__file__).parent.parents[1] / "uci_dataframe.pkl"
                with open(pickle_path_final, 'wb') as f:
                    pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"💾 Dados salvos em pickle: {pickle_path_final}")
            except Exception as e:
                print(f"⚠️ Não foi possível salvar pickle: {e}")
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            print(f"ERRO AO CARREGAR DADOS UCI: {error_detail}")
            raise FileNotFoundError(f"Não foi possível carregar dados UCI (pickle ou brutos): {e}\n\nDetalhes: {error_detail}")
    
    return df

def carregar_oulad_dados(pickle_path: str = "../oulad_data.pkl") -> pd.DataFrame:
    """Carrega dados OULAD - tenta carregar do pickle otimizado, se não conseguir carrega dados brutos"""
    # Primeiro, tentar carregar dados processados do pickle otimizado
    possible_paths = [
        pickle_path,
        f"../{pickle_path}",
        f"../../{pickle_path}",
        Path(__file__).parent.parents[1] / "oulad_data.pkl"
    ]
    
    df = None
    for path in possible_paths:
        p = Path(path)
        if p.is_file():
            try:
                print(f"🔄 Carregando dados OULAD do pickle: {p}")
                with p.open("rb") as f:
                    content = pickle.load(f)
                if isinstance(content, pd.DataFrame):
                    df = content
                    print(f"✅ Dados OULAD carregados: {df.shape}")
                    break
            except Exception as e:
                print(f"⚠️ Erro ao carregar pickle {p}: {e}")
                continue
    
    if df is None:
        try:
            print("🔄 Carregando dados OULAD brutos...")
            dataframes_oulad = carregar_dados_oulad_raw()
            df = processar_dados_oulad(dataframes_oulad)
            
            # Salvar pickle otimizado para próximas execuções
            print("💾 Salvando dados processados em pickle...")
            pickle_path_final = Path(__file__).parent.parents[1] / "oulad_data.pkl"
            with open(pickle_path_final, 'wb') as f:
                pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"✅ Dados salvos em: {pickle_path_final}")
            
        except Exception as e:
            raise FileNotFoundError(f"Não foi possível carregar dados OULAD: {e}")
    
    return df

def carregar_dados_uci_raw():
    """Carrega dados UCI brutos dos arquivos CSV"""
    datasets_path = Path(__file__).parent.parents[1] / 'datasets' / 'uci_data'
    
    # Português
    por_path = os.path.join(datasets_path, 'student-por.csv')
    if is_lfs_pointer(por_path):
        raise ValueError(f"O arquivo {por_path} é um ponteiro Git LFS. Execute 'git lfs pull' localmente.")
    por = pd.read_csv(por_path, sep=';')
    
    # Matemática
    mat_path = os.path.join(datasets_path, 'student-mat.csv')
    if is_lfs_pointer(mat_path):
        raise ValueError(f"O arquivo {mat_path} é um ponteiro Git LFS. Execute 'git lfs pull' localmente.")
    mat = pd.read_csv(mat_path, sep=';')
    
    # Adicionando coluna com o conjunto de dados de origem
    mat['origem'] = 'mat'
    por['origem'] = 'por'
    
    # Concatenando os dataframes
    df = pd.concat([mat, por])
    
    # Transformando valores e tipos de dados
    df['traveltime'] = df['traveltime'].map({1: '<15m', 2: '15-30m', 3: '30-1h', 4: '>1h'})
    df['studytime'] = df['studytime'].map({1: '<2h', 2: '2-5h', 3: '5-10h', 4: '>10h'})
    df[['Medu','Fedu','famrel','goout','Dalc','Walc','health']] = \
        df[['Medu','Fedu','famrel','goout','Dalc','Walc','health']].astype('object')
    
    return df

def carregar_dados_oulad_raw():
    """Carrega dados OULAD brutos dos arquivos CSV com otimizações"""
    datasets_path = Path(__file__).parent.parents[1] / 'datasets' / 'oulad_data'
    
    dataframes_oulad = {}
    
    # Configurações otimizadas para cada arquivo
    file_configs = {
        'studentVle': {'nrows': 50000, 'dtype': {'id_student': 'int32', 'id_site': 'int32', 'date': 'int32', 'sum_click': 'int16'}},
        'studentAssessment': {'dtype': {'id_student': 'int32', 'id_assessment': 'int32', 'score': 'float32', 'date_submitted': 'int32'}},
        'studentInfo': {'dtype': {'id_student': 'int32', 'code_module': 'category', 'code_presentation': 'category', 'gender': 'category', 'region': 'category', 'highest_education': 'category', 'imd_band': 'category', 'age_band': 'category', 'num_of_prev_attempts': 'int8', 'studied_credits': 'int16', 'disability': 'category', 'final_result': 'category'}},
        'studentRegistration': {'dtype': {'id_student': 'int32', 'code_presentation': 'category', 'date_registration': 'float32', 'date_unregistration': 'float32'}},
        'assessments': {'dtype': {'id_assessment': 'int32', 'code_module': 'category', 'code_presentation': 'category', 'assessment_type': 'category', 'date': 'float32', 'weight': 'float32'}},
        'courses': {'dtype': {'code_module': 'category', 'code_presentation': 'category', 'module_presentation_length': 'int16'}},
        'vle': {'dtype': {'id_site': 'int32', 'code_module': 'category', 'code_presentation': 'category', 'activity_type': 'category', 'week_from': 'float32', 'week_to': 'float32'}}
    }
    
    for filename in os.listdir(datasets_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(datasets_path, filename)
            df_name = os.path.splitext(filename)[0]
            try:
                config = file_configs.get(df_name, {})
                if is_lfs_pointer(file_path):
                    print(f"⚠️ {df_name} é um ponteiro Git LFS. Execute 'git lfs pull'.")
                    continue
                    
                dataframes_oulad[df_name] = pd.read_csv(
                    file_path, 
                    sep=',', 
                    encoding='ISO-8859-1',
                    **config
                )
                print(f"✅ Carregado {df_name}: {dataframes_oulad[df_name].shape}")
            except Exception as e:
                print(f"❌ Erro ao carregar o arquivo '{filename}': {e}")
    
    return dataframes_oulad

def processar_dados_oulad(dataframes_oulad):
    """Processa os dados OULAD para análise com otimizações"""
    print("🔄 Processando dados OULAD...")
    
    # Usar dados completos mas com otimizações de memória
    df_assessments = dataframes_oulad['assessments'].copy()
    df_courses = dataframes_oulad['courses'].copy()
    df_vle = dataframes_oulad['vle'].copy()
    df_studentinfo = dataframes_oulad['studentInfo'].copy()
    df_studentregistration = dataframes_oulad['studentRegistration'].copy()
    df_studentassessment = dataframes_oulad['studentAssessment'].copy()
    df_studentvle = dataframes_oulad['studentVle'].copy()
    
    print(f"📊 Dados carregados - studentVle: {df_studentvle.shape}")
    
    # Processar dados de forma mais eficiente
    new_vle = df_vle.drop(['week_from','week_to'], axis=1)
    
    # Imputação otimizada com os valores mais frequentes por região
    if 'imd_band' in df_studentinfo.columns:
        mode_by_region = df_studentinfo.groupby('region')['imd_band'].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown')
        df_studentinfo['imd_band'] = df_studentinfo['imd_band'].fillna(df_studentinfo['region'].map(mode_by_region))
    
    # Imputação otimizada de valores ausentes
    df_student_registration_copy = df_studentregistration.copy()
    
    # Criar variável binária indicando se o estudante cancelou o registro
    df_student_registration_copy['cancelou'] = df_student_registration_copy['date_unregistration'].notna().astype(int)
    
    # Preencher date_unregistration com valor alto quando ausente (para diferenciar de valores reais)
    # Usar max + 1000 para garantir que seja claramente distinto de qualquer data real
    max_date_unregistration = df_student_registration_copy['date_unregistration'].max()
    if pd.notna(max_date_unregistration):
        valor_nao_cancelou = max_date_unregistration + 1000
    else:
        # Se todos os valores forem NaN, usar um valor padrão alto
        valor_nao_cancelou = 999999
    df_student_registration_copy['date_unregistration'] = df_student_registration_copy['date_unregistration'].fillna(valor_nao_cancelou)
    
    # Preencher date_registration com a média quando ausente
    mean_date_registration = df_student_registration_copy['date_registration'].mean()
    df_student_registration_copy['date_registration'] = df_student_registration_copy['date_registration'].fillna(mean_date_registration)
    
    print("🔄 Fazendo joins dos dados...")
    
    # Junção dos dados de forma mais eficiente
    vle_activities = pd.merge(df_studentvle, new_vle, on=['code_module','code_presentation','id_site'], how='inner')
    print(f"📊 Após merge VLE: {vle_activities.shape}")
    
    assessments_activities = pd.merge(df_studentassessment, df_assessments, on='id_assessment', how='inner')
    print(f"📊 Após merge assessments: {assessments_activities.shape}")
    
    studentinfo_activities = pd.merge(vle_activities, df_studentinfo, on=['code_module','code_presentation','id_student'], how='inner')
    print(f"📊 Após merge student info: {studentinfo_activities.shape}")
    
    merged_df = pd.merge(studentinfo_activities, assessments_activities, on=['code_module','code_presentation','id_student'], how='inner')
    print(f"📊 Após merge assessments: {merged_df.shape}")
    
    # Merge com outros dataframes
    merged_df = pd.merge(merged_df, df_courses, on=['code_presentation'], how='inner')
    merged_df = pd.merge(merged_df, df_student_registration_copy, on=['code_presentation','id_student'], how='inner')
    
    print(f"📊 Dataset final: {merged_df.shape}")
    
    # Imputação otimizada de valores ausentes
    print("🔄 Imputando valores ausentes...")
    numeric_cols = merged_df.select_dtypes(include=['number']).columns
    categorical_cols = merged_df.select_dtypes(include='object').columns
    
    # Usar fillna com valores específicos para melhor performance
    for col in numeric_cols:
        if merged_df[col].isnull().any():
            merged_df[col] = merged_df[col].fillna(merged_df[col].median())
    
    for col in categorical_cols:
        if merged_df[col].isnull().any():
            merged_df[col] = merged_df[col].fillna(merged_df[col].mode().iloc[0] if not merged_df[col].mode().empty else 'Unknown')
    
    # Otimizar tipos de dados para economizar memória
    print("🔄 Otimizando tipos de dados...")
    for col in merged_df.select_dtypes(include=['int64']).columns:
        if merged_df[col].max() < 2147483647:  # int32 max
            merged_df[col] = merged_df[col].astype('int32')
        elif merged_df[col].max() < 32767:  # int16 max
            merged_df[col] = merged_df[col].astype('int16')
        elif merged_df[col].max() < 255:  # int8 max
            merged_df[col] = merged_df[col].astype('int8')
    
    for col in merged_df.select_dtypes(include=['float64']).columns:
        merged_df[col] = merged_df[col].astype('float32')
    
    print(f"✅ Processamento concluído! Dataset final: {merged_df.shape}")
    print(f"💾 Uso de memória: {merged_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return merged_df