"""
Неделя 3-4: метрики качества и валидация.

Отток — редкое событие, поэтому accuracy не подходит. Используем метрики
ранжирования: ROC-AUC, PR-AUC, Precision@k, Lift@k.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def precision_at_k(y_true, y_score, k: float = 0.1) -> float:
    """Доля реально ушедших среди топ-k% клиентов по прогнозу риска."""
    n = max(1, int(len(y_score) * k))
    idx = np.argsort(-np.asarray(y_score))[:n]
    return float(np.asarray(y_true)[idx].mean())


def lift_at_k(y_true, y_score, k: float = 0.1) -> float:
    """Во сколько раз топ-k% лучше случайного выбора."""
    return precision_at_k(y_true, y_score, k) / (np.mean(y_true) + 1e-12)


def evaluate(y_true, y_score, label: str = "") -> dict:
    """Печатает и возвращает основные метрики качества."""
    metrics = {
        "label": label,
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "precision_at_10": precision_at_k(y_true, y_score, 0.10),
        "lift_at_10": lift_at_k(y_true, y_score, 0.10),
        "churn_rate": float(np.mean(y_true)),
    }
    print(f"--- {label} ---")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"Precision@10%: {metrics['precision_at_10']:.4f}")
    print(f"Lift@10%: {metrics['lift_at_10']:.2f}x")
    print(f"Доля оттока: {metrics['churn_rate']:.4f}")
    return metrics


def bootstrap_auc_diff(y_true, s1, s2, n_iter: int = 1000, seed: int = 42) -> dict:
    """Bootstrap-сравнение двух подходов: доля выборок, где подход 1 лучше.

    Используйте на неделе 4 для честного сравнения подходов A/B/C — разница
    в 0.5-1 п.п. AUC может быть просто шумом.
    """
    rng = np.random.default_rng(seed)
    y_true, s1, s2 = map(np.asarray, (y_true, s1, s2))
    wins, diffs = 0, []
    for _ in range(n_iter):
        idx = rng.integers(0, len(y_true), len(y_true))
        if y_true[idx].min() == y_true[idx].max():
            continue  # выборка без обоих классов — пропускаем
        d = roc_auc_score(y_true[idx], s1[idx]) - roc_auc_score(y_true[idx], s2[idx])
        diffs.append(d)
        wins += d > 0
    if not diffs:
        raise ValueError("Ни одна bootstrap-выборка не содержала оба класса")
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    result = {
        "win_rate": wins / len(diffs),
        "mean_diff": float(np.mean(diffs)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "significant": bool(lo > 0 or hi < 0),
    }
    print(f"Средняя разница AUC = {result['mean_diff']:+.4f}, "
          f"95% интервал [{lo:+.4f}, {hi:+.4f}]")
    print("Различие значимо" if result["significant"]
          else "Различие НЕ значимо: интервал накрывает ноль")
    return result


# --------------------------------------------------------------------------
# Неделя 3: разбиение по времени
# --------------------------------------------------------------------------

# Схема с зазором. Между окнами прогноза train/valid и окном наблюдения теста
# не должно быть ни одного общего месяца — иначе метрика на тесте завышена.
# Снапшоты 2025-04, 2025-07, 2025-10 выброшены намеренно. См. тетрадь 7.2.
TRAIN_SNAPSHOTS = ['2024-07', '2024-10']
VALID_SNAPSHOTS = ['2025-01']
TEST_SNAPSHOTS = ['2026-01']


def split_by_time(dataset):
    """train, valid, test — три DataFrame

    Схема из раздела 7.2 рабочей тетради:
    - train: точка отсчета 2024-07, 2024-10
    - valid: точка отсчета 2025-01 (для early stopping)
    - test: точка отсчтеа 2026-01
    - зазор (не используем): 2025-04, 2025-07, 2025-10
    """
    train = dataset[dataset['snapshot_date'].isin(TRAIN_SNAPSHOTS)].copy()
    valid = dataset[dataset['snapshot_date'].isin(VALID_SNAPSHOTS)].copy()
    test = dataset[dataset['snapshot_date'].isin(TEST_SNAPSHOTS)].copy()
    
    return train, valid, test