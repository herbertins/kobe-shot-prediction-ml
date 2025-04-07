from sklearn.model_selection import train_test_split
import pandas as pd
import mlflow

def filter_and_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Colunas selecionadas
    cols = [
        'lat',
        'lon',
        'minutes_remaining',
        'period',
        'playoffs',
        'shot_distance',
        'shot_made_flag'
    ]
    
    # Seleção e limpeza
    df_filtered = df[cols].dropna()
    
    return df_filtered

def split_train_test(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X = df.drop('shot_made_flag', axis=1)
    y = df['shot_made_flag']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y
    )

    # Reconstituir DataFrames com alvo incluso
    df_train = X_train.copy()
    df_train['shot_made_flag'] = y_train

    df_test = X_test.copy()
    df_test['shot_made_flag'] = y_test
    
    with mlflow.start_run(run_name="PreparacaoDados", nested=True):
        mlflow.log_param("test_size_percent", 0.2 * 100)
        mlflow.log_metric("train_rows", df_train.shape[0])
        mlflow.log_metric("test_rows", df_test.shape[0])
        mlflow.log_metric("total_rows", df.shape[0])

    return df_train, df_test