"""
Неделя 2-3: сборка обучающей выборки и построение признаков.

Правило №1 проекта: признаки строятся ТОЛЬКО из данных до точки отсчёта
(snapshot), таргет — ТОЛЬКО из данных после.

Что писать когда:
  неделя 2 — window_bounds, make_features (минимум, 4 признака), make_target, build_snapshot
  неделя 3 — расширить make_features до пяти групп признаков

Сигнатуры функций не меняйте — по ним ревьюер сравнивает решения.
"""
import pandas as pd
import numpy as np

# Константы
OBSERVATION_MONTHS = 6   # окно наблюдения
GAP_MONTHS = 2           # слепой зазор
FORECAST_MONTHS = 3      # окно прогноза

SNAPSHOTS = [
    '2024-07', '2024-10', '2025-01', '2025-04',
    '2025-07', '2025-10', '2026-01'
]


def window_bounds(snapshot: str) -> tuple:
    """
    Возвращает (obs_start, obs_end, fc_start, fc_end) для заданного snapshot.
    """
    snap = pd.Period(snapshot, freq='M')
    
    obs_start = str(snap - (OBSERVATION_MONTHS - 1))
    obs_end = str(snap)
    fc_start = str(snap + GAP_MONTHS + 1)
    fc_end = str(snap + GAP_MONTHS + FORECAST_MONTHS)
    
    return obs_start, obs_end, fc_start, fc_end


def make_features(obs: pd.DataFrame, obs_end) -> pd.DataFrame:
    """
    Минимальная версия: 4 признака.
    Индекс результата — client_id.
    """
    features = obs.groupby('client_id').agg(
        rev_mean=('revenue', 'mean'),
        n_months=('revenue', 'count')
    )
    
    # revenue последнего месяца окна
    last_month = obs[obs['month'] == obs_end].set_index('client_id')['revenue']
    features['rev_last'] = last_month
    
    # Отношение последнего к среднему (защита от деления на ноль)
    features['rev_last_to_mean'] = features['rev_last'] / (features['rev_mean'] + 1e-9)
    
    return features


def make_target(fut: pd.DataFrame, alive, rev_mean: pd.Series) -> pd.Series:
    """
    Таргет: 1 если нет строк в будущем ИЛИ revenue < 20% от rev_mean.
    """
    if len(fut) > 0:
        fut_mean = fut.groupby('client_id')['revenue'].mean()
    else:
        fut_mean = pd.Series(dtype=float)
    
    # Выравниваем по списку alive
    fut_mean = fut_mean.reindex(alive)
    rev_mean_aligned = rev_mean.reindex(alive)
    
    target = pd.Series(0, index=alive)
    
    # Условие 1: нет данных в будущем
    no_data = fut_mean.isna()
    target[no_data] = 1
    
    # Условие 2: revenue упал ниже 20%
    low_revenue = (fut_mean / (rev_mean_aligned + 1e-9)) < 0.2
    target[low_revenue] = 1
    
    return target


def build_snapshot(usage: pd.DataFrame, clients: pd.DataFrame, snapshot: str) -> pd.DataFrame:
    """
    Собирает обучающую выборку для одного snapshot.
    """
    # 1. Границы окон
    obs_start, obs_end, fc_start, fc_end = window_bounds(snapshot)
    
    # 2. Срез окна наблюдения
    obs = usage[(usage['month'] >= obs_start) & (usage['month'] <= obs_end)].copy()
    
    # 3. Скоуп: только живые на конец окна
    alive = obs.loc[obs['month'] == obs_end, 'client_id'].unique()
    obs = obs[obs['client_id'].isin(alive)]
    
    # 4. Признаки — ТОЛЬКО из obs
    X = make_features(obs, obs_end)
    
    # 5. Таргет — ТОЛЬКО из окна прогноза
    fut = usage[(usage['month'] >= fc_start) & (usage['month'] <= fc_end)]
    y = make_target(fut, alive, X['rev_mean'])
    
    # 6. Сборка
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
# ГОТОВО — переписывать не нужно, просто применяйте
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