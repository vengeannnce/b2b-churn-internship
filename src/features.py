"""
Неделя 2-3: сборка обучающей выборки и построение признаков.
"""
import pandas as pd
import numpy as np

#Константы
OBSERVATION_MONTHS = 6
GAP_MONTHS = 2
FORECAST_MONTHS = 3

SNAPSHOTS = [
    '2024-07', '2024-10', '2025-01', '2025-04',
    '2025-07', '2025-10', '2026-01'
]


def window_bounds(snapshot: str) -> tuple:
    """Возвращает (obs_start, obs_end, fc_start, fc_end)."""
    snap = pd.Period(snapshot, freq='M')
    obs_start = str(snap - (OBSERVATION_MONTHS - 1))
    obs_end = str(snap)
    fc_start = str(snap + GAP_MONTHS + 1)
    fc_end = str(snap + GAP_MONTHS + FORECAST_MONTHS)
    return obs_start, obs_end, fc_start, fc_end


def make_features(obs: pd.DataFrame, obs_end: str) -> pd.DataFrame:
    """
    Расширенная версия: 5 групп признаков.
    Индекс результата — client_id.
    
    Группы:
    1. Платежи: revenue_mean, revenue_min, revenue_max, revenue_last, revenue_last_to_mean
    2. Динамика: revenue_std, revenue_trend, months_declining, traffic_last_to_mean
    3. Объём отношений: n_sim_last, n_sim_trend
    4. Сигналы боли: tickets_sum, tickets_last, debt_last, debt_months
    """
    #Агрегаты по числовым колонкам
    agg_dict = {}
    if 'revenue' in obs.columns:
        agg_dict['revenue'] = ['mean', 'min', 'max', 'std', 'count']
    if 'traffic' in obs.columns:
        agg_dict['traffic'] = ['mean', 'last']
    if 'n_sim' in obs.columns:
        agg_dict['n_sim'] = ['last', 'mean']
    if 'tickets' in obs.columns:
        agg_dict['tickets'] = ['sum', 'last']
    if 'debt' in obs.columns:
        agg_dict['debt'] = ['mean', 'last']

    features = obs.groupby('client_id').agg(agg_dict)
    features.columns = [f"{col}_{stat}" for col, stat in features.columns]

    #Отношения последнего месяца к среднему ===
    if 'revenue_mean' in features.columns:
        last_rev = obs[obs['month'] == obs_end].set_index('client_id')['revenue']
        features['revenue_last'] = last_rev
        features['revenue_last_to_mean'] = features['revenue_last'] / (features['revenue_mean'] + 1e-9)

    if 'traffic_mean' in features.columns and 'traffic_last' in features.columns:
        features['traffic_last_to_mean'] = features['traffic_last'] / (features['traffic_mean'] + 1e-9)

    #Динамика: тренды (используем готовые функции) ===
    if 'revenue' in obs.columns:
        # Сортируем по месяцу, чтобы тренд считался корректно
        obs_sorted = obs.sort_values(['client_id', 'month'])
        features['revenue_trend'] = obs_sorted.groupby('client_id')['revenue'].apply(trend_slope)
        features['months_declining'] = obs_sorted.groupby('client_id')['revenue'].apply(months_declining)
        
        # Нормализуем тренд: делим на средний уровень (иначе крупный бизнес доминирует)
        features['revenue_trend_norm'] = features['revenue_trend'] / (features['revenue_mean'] + 1e-9)

    #Объём отношений: тренд n_sim ===
    if 'n_sim' in obs.columns:
        obs_sorted = obs.sort_values(['client_id', 'month'])
        features['n_sim_trend'] = obs_sorted.groupby('client_id')['n_sim'].apply(trend_slope)

    #Сигналы боли: debt_months (сколько месяцев с долгом > 0) ===
    if 'debt' in obs.columns:
        features['debt_months'] = (obs['debt'] > 0).groupby(obs['client_id']).sum()

    return features


def make_target(fut: pd.DataFrame, alive, rev_mean: pd.Series) -> pd.Series:
    """Таргет: 1 если нет строк в будущем ИЛИ revenue < 20% от rev_mean."""
    if len(fut) > 0:
        fut_mean = fut.groupby('client_id')['revenue'].mean()
    else:
        fut_mean = pd.Series(dtype=float)
    
    fut_mean = fut_mean.reindex(alive)
    rev_mean_aligned = rev_mean.reindex(alive)
    
    target = pd.Series(0, index=alive)
    no_data = fut_mean.isna()
    target[no_data] = 1
    low_revenue = (fut_mean / (rev_mean_aligned + 1e-9)) < 0.2
    target[low_revenue] = 1
    
    return target


def build_snapshot(usage: pd.DataFrame, clients: pd.DataFrame, snapshot: str) -> pd.DataFrame:
    """Собирает обучающую выборку для одного snapshot."""
    #Границы окон
    obs_start, obs_end, fc_start, fc_end = window_bounds(snapshot)
    
    #Срез окна наблюдения
    obs = usage[(usage['month'] >= obs_start) & (usage['month'] <= obs_end)].copy()
    
    #Скоуп: только живые на конец окна
    alive = obs.loc[obs['month'] == obs_end, 'client_id'].unique()
    obs = obs[obs['client_id'].isin(alive)]
    
    #Признаки — ТОЛЬКО из obs
    X = make_features(obs, obs_end)
    
    #Таргет — ТОЛЬКО из окна прогноза
    fut = usage[(usage['month'] >= fc_start) & (usage['month'] <= fc_end)]
    y = make_target(fut, alive, X['revenue_mean'])
    
    #Сборка
    result = X.reset_index()
    result['target'] = y.values
    result['snapshot_date'] = snapshot
    
    result = result.merge(
        clients[['client_id', 'segment', 'product', 'region']],
        on='client_id',
        how='left'
    )
    
    assert result['segment'].isna().sum() == 0, 'потеряли сегмент при merge'
    
    return result


# --------------------------------------------------------------------------
# ГОТОВО — переписывать не нужно
# --------------------------------------------------------------------------

def trend_slope(s: pd.Series) -> float:
    """Наклон линейного тренда: >0 растёт, <0 падает."""
    y = s.to_numpy(dtype=float)
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y))
    return float(np.polyfit(x, y, 1)[0])


def months_declining(s: pd.Series) -> int:
    """Сколько последних месяцев подряд значение снижалось."""
    diffs = s.diff().to_numpy()[1:]
    cnt = 0
    for d in diffs[::-1]:
        if d < 0:
            cnt += 1
        else:
            break
    return cnt