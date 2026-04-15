# =============================================================================
# Лабораторна робота №3
# Дисципліна: Методи інтелектуальної обробки даних
# Тема: Лінійна регресія та попередня обробка даних
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_log_error,
    max_error,
    d2_absolute_error_score
)

# ======================== НАЛАШТУВАННЯ ========================
# Тут використовується датасет "California Housing" зі sklearn
# (понад 20 000 рядків, 8 числових ознак) — не потребує завантаження з Kaggle
# Якщо у вас є власний CSV — замініть блок нижче на pd.read_csv("your_file.csv")
# ==============================================================

from sklearn.datasets import fetch_california_housing

# =============================================================================
# ЗАВДАННЯ 1. ПІДГОТОВКА НАБОРУ ДАНИХ
# =============================================================================

# ─────────────────────────────────────────────────────────────
# 1.1 Завантаження набору даних
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("ЗАВДАННЯ 1. ПІДГОТОВКА НАБОРУ ДАНИХ")
print("=" * 60)

housing = fetch_california_housing(as_frame=True)
df = housing.frame.copy()

print("\n▶ Перші 5 рядків датасету:")
print(df.head())
print(f"\nРозмір датасету: {df.shape[0]} рядків, {df.shape[1]} колонок")

# ─────────────────────────────────────────────────────────────
# 1.2 Базова підготовка даних
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("1.2 Базова підготовка даних")
print("─" * 60)

# Перевіримо типи
print("\n▶ Типи даних:")
print(df.dtypes)

# Вибір ознак та цільової змінної
# Цільова змінна: MedHouseVal (медіанна вартість житла)
# Ознаки: 8 числових колонок
FEATURE_COLS = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
                'Population', 'AveOccup', 'Latitude', 'Longitude']
TARGET_COL   = 'MedHouseVal'

df = df[FEATURE_COLS + [TARGET_COL]]

# Обробка пропусків
print(f"\n▶ Пропущені значення:\n{df.isnull().sum()}")
df.dropna(inplace=True)

# Фільтрація аномалій (IQR-метод для кожної колонки)
def remove_outliers_iqr(dataframe, cols, factor=3.0):
    mask = pd.Series([True] * len(dataframe), index=dataframe.index)
    for col in cols:
        Q1 = dataframe[col].quantile(0.25)
        Q3 = dataframe[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        mask &= (dataframe[col] >= lower) & (dataframe[col] <= upper)
    return dataframe[mask]

all_cols = FEATURE_COLS + [TARGET_COL]
df_clean = remove_outliers_iqr(df, all_cols)
print(f"\n▶ Розмір після видалення аномалій: {df_clean.shape[0]} рядків")

# ─────────────────────────────────────────────────────────────
# 1.3 Знаходження залежностей
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("1.3 Знаходження залежностей")
print("─" * 60)

# Кореляційна матриця
corr_matrix = df_clean.corr()

# Ознаки відсортовані за кореляцією з цільовою змінною
target_corr = corr_matrix[TARGET_COL].drop(TARGET_COL).abs().sort_values(ascending=False)
sorted_features = target_corr.index.tolist()
BEST_FEATURE = sorted_features[0]  # найбільш пов'язана ознака

print(f"\n▶ Кореляція ознак з '{TARGET_COL}' (за спаданням):")
print(corr_matrix[TARGET_COL].drop(TARGET_COL).reindex(sorted_features))
print(f"\n▶ Найбільш корельована ознака: {BEST_FEATURE}")

# Теплова карта (ознаки відсортовано за кореляцією з таргетом)
order = sorted_features + [TARGET_COL]
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    corr_matrix.loc[order, order],
    annot=True, fmt=".2f", cmap="coolwarm",
    linewidths=0.5, ax=ax
)
ax.set_title("Теплова карта кореляцій\n(ознаки відсортовано за зв'язком з цільовою змінною)")
plt.tight_layout()
plt.savefig("heatmap.png", dpi=120)
plt.show()

# Видалення сильно корелюючих ознак між собою (|r| > 0.85)
CORR_THRESHOLD = 0.85
drop_features = set()
feature_corr_matrix = corr_matrix.loc[FEATURE_COLS, FEATURE_COLS]
for i, f1 in enumerate(FEATURE_COLS):
    for f2 in FEATURE_COLS[i+1:]:
        if abs(feature_corr_matrix.loc[f1, f2]) > CORR_THRESHOLD:
            # Залишаємо ту, що більше корелює з таргетом
            keep = f1 if abs(corr_matrix.loc[f1, TARGET_COL]) >= abs(corr_matrix.loc[f2, TARGET_COL]) else f2
            drop = f2 if keep == f1 else f1
            drop_features.add(drop)
            print(f"  Висока кореляція між '{f1}' і '{f2}' ({feature_corr_matrix.loc[f1,f2]:.2f}) → видаляємо '{drop}'")

selected_features = [f for f in sorted_features if f not in drop_features]
print(f"\n▶ Фінальні ознаки для моделей: {selected_features}")

X = df_clean[selected_features]
y = df_clean[TARGET_COL]

# Поділ на тренувальну та тестову вибірку (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n▶ Тренувальна вибірка: {X_train.shape[0]} зразків")
print(f"▶ Тестова вибірка:     {X_test.shape[0]} зразків")


# =============================================================================
# ДОПОМІЖНА ФУНКЦІЯ ДЛЯ ОБЧИСЛЕННЯ МЕТРИК
# =============================================================================

def compute_metrics(model, X_tr, X_te, y_tr, y_te, label=""):
    """Повертає dict з усіма метриками для train і test."""
    y_tr_pred = model.predict(X_tr)
    y_te_pred = model.predict(X_te)

    # MSLE вимагає, щоб і таргет, і прогноз були > -1
    # Після PowerTransformer значення стандартизовані і можуть бути < -1
    def safe_msle(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        if np.any(y_true <= -1) or np.any(y_pred <= -1):
            return float('nan')  # MSLE не застосовний для таких значень
        y_pred_clip = np.clip(y_pred, 0, None)
        return round(mean_squared_log_error(y_true, y_pred_clip), 4)

    result = {
        "Модель": label,
        "R² (train)":   round(model.score(X_tr, y_tr), 4),
        "R² (test)":    round(model.score(X_te, y_te), 4),
        "MAE (train)":  round(mean_absolute_error(y_tr, y_tr_pred), 4),
        "MAE (test)":   round(mean_absolute_error(y_te, y_te_pred), 4),
        "MSLE (train)": safe_msle(y_tr, y_tr_pred),
        "MSLE (test)":  safe_msle(y_te, y_te_pred),
        "MaxErr (train)": round(max_error(y_tr, y_tr_pred), 4),
        "MaxErr (test)":  round(max_error(y_te, y_te_pred), 4),
        "D²AbsErr (train)": round(d2_absolute_error_score(y_tr, y_tr_pred), 4),
        "D²AbsErr (test)":  round(d2_absolute_error_score(y_te, y_te_pred), 4),
    }
    return result


# =============================================================================
# ДОПОМІЖНА ФУНКЦІЯ ДЛЯ ВІЗУАЛІЗАЦІЇ
# =============================================================================

def plot_regression(models_info, X_tr, X_te, y_tr, y_te, best_feat, title="Регресія"):
    """
    models_info: список словників {"label": str, "model": fitted_model, "color": str}
    """
    feat_idx = list(X_tr.columns).index(best_feat)
    x_tr_feat = X_tr.iloc[:, feat_idx].values
    x_te_feat = X_te.iloc[:, feat_idx].values

    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plots
    ax.scatter(x_tr_feat, y_tr, alpha=0.3, s=10, color='steelblue', label='Train (факт)')
    ax.scatter(x_te_feat, y_te, alpha=0.3, s=10, color='darkorange', label='Test (факт)')

    # Prediction lines
    pred_colors = ['blue', 'red', 'green', 'purple', 'brown']
    line_styles = ['-', '--', '-.', ':', '-']

    for i, info in enumerate(models_info):
        model = info["model"]
        color = pred_colors[i % len(pred_colors)]
        ls    = line_styles[i % len(line_styles)]

        # Сортуємо x для гладкої лінії
        sort_tr = np.argsort(x_tr_feat)
        sort_te = np.argsort(x_te_feat)

        ax.plot(x_tr_feat[sort_tr],
                model.predict(X_tr.iloc[sort_tr]),
                color=color, ls=ls, lw=1.5,
                label=f"{info['label']} — train (прогноз)")
        ax.plot(x_te_feat[sort_te],
                model.predict(X_te.iloc[sort_te]),
                color=color, ls=':', lw=1.5,
                label=f"{info['label']} — test (прогноз)")

    ax.set_xlabel(best_feat)
    ax.set_ylabel(TARGET_COL)
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png", dpi=120)
    plt.show()


# =============================================================================
# ЗАВДАННЯ 2. ПОБУДОВА РЕГРЕСІЙНИХ МОДЕЛЕЙ
# =============================================================================
print("\n" + "=" * 60)
print("ЗАВДАННЯ 2. ПОБУДОВА РЕГРЕСІЙНИХ МОДЕЛЕЙ")
print("=" * 60)

all_metrics = []  # збираємо всі метрики сюди

# ─────────────────────────────────────────────────────────────
# 2.1 Проста лінійна регресія
# ─────────────────────────────────────────────────────────────
print("\n▶ 2.1 Проста лінійна регресія")

lr = LinearRegression()
lr.fit(X_train, y_train)

metrics_21 = compute_metrics(lr, X_train, X_test, y_train, y_test, label="LR (2.1)")
all_metrics.append(metrics_21)

df_metrics_21 = pd.DataFrame([metrics_21]).set_index("Модель")
print(df_metrics_21.T)

plot_regression(
    [{"label": "LinearRegression", "model": lr}],
    X_train, X_test, y_train, y_test,
    BEST_FEATURE,
    title="2.1 Лінійна регресія"
)

print("\n▶ Висновок (2.1):")
r2_tr = metrics_21["R² (train)"]
r2_te = metrics_21["R² (test)"]
gap = abs(r2_tr - r2_te)
print(f"  R²(train)={r2_tr}, R²(test)={r2_te}, різниця={gap:.4f}")
if gap < 0.03:
    print("  Модель добре узагальнюється — переоснащення відсутнє.")
else:
    print("  Є ознаки переоснащення або нелінійності — варто спробувати перетворення.")


# ─────────────────────────────────────────────────────────────
# 2.2 Лінійна регресія з перетворенням змінних (Yeo-Johnson)
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("▶ 2.2 Лінійна регресія з перетворенням до нормального розподілу")

pt = PowerTransformer(method='yeo-johnson', standardize=True)
X_train_pt = pd.DataFrame(pt.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_pt  = pd.DataFrame(pt.transform(X_test),      columns=X_test.columns,  index=X_test.index)

# Цільову змінну також трансформуємо
pt_y = PowerTransformer(method='yeo-johnson', standardize=True)
y_train_pt = pd.Series(pt_y.fit_transform(y_train.values.reshape(-1,1)).ravel(), index=y_train.index)
y_test_pt  = pd.Series(pt_y.transform(y_test.values.reshape(-1,1)).ravel(),      index=y_test.index)

lr_pt = LinearRegression()
lr_pt.fit(X_train_pt, y_train_pt)

metrics_22 = compute_metrics(lr_pt, X_train_pt, X_test_pt, y_train_pt, y_test_pt, label="LR+PowerTransform (2.2)")
all_metrics.append(metrics_22)

df_metrics_22 = pd.DataFrame([metrics_22]).set_index("Модель")
print(df_metrics_22.T)

plot_regression(
    [{"label": "LR+PowerTransform", "model": lr_pt}],
    X_train_pt, X_test_pt, y_train_pt, y_test_pt,
    BEST_FEATURE,
    title="2.2 Лінійна регресія з перетворенням"
)


# ─────────────────────────────────────────────────────────────
# 2.3 Гребенева регресія (Ridge) — 3 моделі
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("▶ 2.3 Гребенева регресія (Ridge): alpha=1.0, 0.01, 100.0")

alpha_values = [1.0, 0.01, 100.0]  # default + 2 власні значення
ridge_models = {}
ridge_labels = {}

for alpha in alpha_values:
    label_str = f"Ridge α={alpha} (2.3)"
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    ridge_models[alpha] = ridge
    ridge_labels[alpha] = label_str
    m = compute_metrics(ridge, X_train, X_test, y_train, y_test, label=label_str)
    all_metrics.append(m)

df_metrics_23 = pd.DataFrame(
    [compute_metrics(ridge_models[a], X_train, X_test, y_train, y_test, label=f"α={a}")
     for a in alpha_values]
).set_index("Модель")
print(df_metrics_23.T)

plot_regression(
    [{"label": f"Ridge α={a}", "model": ridge_models[a]} for a in alpha_values],
    X_train, X_test, y_train, y_test,
    BEST_FEATURE,
    title="2.3 Гребенева регресія (Ridge)"
)


# ─────────────────────────────────────────────────────────────
# 2.4 Гребенева регресія з перетворенням змінних
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("▶ 2.4 Ridge + перетворення до нормального розподілу")

ridge_pt_models = {}
for alpha in alpha_values:
    label_str = f"Ridge+PT α={alpha} (2.4)"
    ridge_pt = Ridge(alpha=alpha)
    ridge_pt.fit(X_train_pt, y_train_pt)
    ridge_pt_models[alpha] = ridge_pt
    m = compute_metrics(ridge_pt, X_train_pt, X_test_pt, y_train_pt, y_test_pt, label=label_str)
    all_metrics.append(m)

df_metrics_24 = pd.DataFrame(
    [compute_metrics(ridge_pt_models[a], X_train_pt, X_test_pt, y_train_pt, y_test_pt, label=f"Ridge+PT α={a}")
     for a in alpha_values]
).set_index("Модель")
print(df_metrics_24.T)

plot_regression(
    [{"label": f"Ridge+PT α={a}", "model": ridge_pt_models[a]} for a in alpha_values],
    X_train_pt, X_test_pt, y_train_pt, y_test_pt,
    BEST_FEATURE,
    title="2.4 Ridge + перетворення"
)


# =============================================================================
# 2.5 Сукупне порівняння метрик
# =============================================================================
print("\n" + "=" * 60)
print("2.5 Сукупне порівняння метрик усіх моделей")
print("=" * 60)

df_all = pd.DataFrame(all_metrics).set_index("Модель")

# ─── Датафрейм абсолютних метрик ───────────────────────────
abs_cols = [c for c in df_all.columns if any(k in c for k in ["MAE", "MSLE", "MaxErr"])]
df_absolute = df_all[abs_cols].copy()

# ─── Датафрейм відносних метрик ────────────────────────────
rel_cols = [c for c in df_all.columns if any(k in c for k in ["R²", "D²"])]
df_relative = df_all[rel_cols].copy()

print("\n▶ Абсолютні метрики:")
print(df_absolute)
print("\n▶ Відносні метрики:")
print(df_relative)

# ─── Стилізація: червоний(низьке) → зелений(високе) для кожного датафрейму ─
# Перевіряємо наявність jinja2, якщо немає — встановлюємо
try:
    import jinja2
    HAS_JINJA2 = True
except ImportError:
    import subprocess, sys
    print("⚙️  Встановлення jinja2...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "jinja2", "-q"])
    import jinja2
    HAS_JINJA2 = True

# Абсолютні: менше — краще → RdYlGn_r
styled_abs = df_absolute.style.background_gradient(cmap="RdYlGn_r", axis=0).format("{:.4f}")
# Відносні: більше — краще → RdYlGn
styled_rel = df_relative.style.background_gradient(cmap="RdYlGn", axis=0).format("{:.4f}")

# Виведення у вигляді HTML (якщо є Jupyter)
try:
    from IPython.display import display
    print("\n▶ Абсолютні метрики (з кольоровою шкалою):")
    display(styled_abs)
    print("\n▶ Відносні метрики (з кольоровою шкалою):")
    display(styled_rel)
except ImportError:
    pass

# Збереження у HTML для перегляду поза Jupyter
html_abs = styled_abs.to_html()
html_rel = styled_rel.to_html()

with open("metrics_absolute.html", "w", encoding="utf-8") as f:
    f.write("<h2>Абсолютні метрики</h2>\n" + html_abs)

with open("metrics_relative.html", "w", encoding="utf-8") as f:
    f.write("<h2>Відносні метрики</h2>\n" + html_rel)

print("\n✅ Таблиці зі стилізацією збережено у:")
print("   metrics_absolute.html")
print("   metrics_relative.html")
print("\n✅ Графіки збережено у поточній директорії (*.png)")
print("\n✅ Лабораторну роботу виконано!")