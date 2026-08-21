"""
Неделя 1: инструменты для разведочного анализа (EDA).

explore() и churn_rate_by() уже написаны — применяйте их, переписывать не надо.
Ваша задача на неделе 1 — функция missing_months() внизу файла.
"""
import pandas as pd


def explore(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """ГОТОВО. Быстрый профиль таблицы: типы, пропуски, уникальные значения."""
    print(f"===== {name} =====")
    print(f"Размер: {df.shape[0]} строк, {df.shape[1]} столбцов")

    summary = pd.DataFrame({
        "тип": df.dtypes,
        "пропусков": df.isna().sum(),
        "пропусков_%": (df.isna().mean() * 100).round(1),
        "уникальных": df.nunique(),
    })
    print(summary)

    numeric = df.select_dtypes("number")
    if not numeric.empty:
        print("\n--- Числовые столбцы ---")
        print(numeric.describe().T.round(2))

    return summary


def churn_rate_by(df: pd.DataFrame, target_col: str, group_col: str) -> pd.DataFrame:
    """ГОТОВО. Доля оттока в разбивке по группе (сегмент/продукт/регион).

    Колонки target_col в данных нет — вы создаёте её сами, см. тетрадь 5.3, шаг 5.
    """
    return (
        df.groupby(group_col)[target_col]
        .agg(["mean", "count"])
        .rename(columns={"mean": "доля_оттока", "count": "клиентов"})
        .sort_values("доля_оттока", ascending=False)
    )


def missing_months(usage: pd.DataFrame) -> pd.DataFrame:
    """
    Для каждого клиента: первый месяц, последний месяц, сколько месяцев
    фактически есть и сколько должно быть между первым и последним.
    
    Returns
    -------
    DataFrame с индексом client_id и столбцами:
    first_month, last_month, n_months_actual, n_months_expected, has_gap
    """
    #создаём копию, чтобы не менять исходные данные
    usage_copy = usage.copy()
    
    #переводим строку '2024-01' в pd.Period для арифметики с месяцами
    usage_copy['month_period'] = pd.to_datetime(usage_copy['month']).dt.to_period('M')
    
    #группируем по client_id и считаем:
    #первый месяц (min)
    #последний месяц (max)
    #сколько месяцев фактически есть (nunique)
    result = usage_copy.groupby('client_id').agg(
        first_month=('month_period', 'min'),
        last_month=('month_period', 'max'),
        n_months_actual=('month_period', 'nunique')
    ).reset_index()
    
    #считаем, сколько месяцев ДОЛЖНО быть между первым и последним
    #например: от 2024-01 до 2024-06 = 6 месяцев (а не 5!)
    #формула: (последний - первый).n + 1
    result['n_months_expected'] = (
        (result['last_month'] - result['first_month']).apply(lambda x: x.n) + 1
    )
    
    #если фактических месяцев меньше, чем ожидаемых → есть дыра
    result['has_gap'] = result['n_months_actual'] < result['n_months_expected']
    
    #устанавливаем client_id как индекс (так удобнее смотреть)
    result = result.set_index('client_id')
    
    return result