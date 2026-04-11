import warnings
warnings.filterwarnings("ignore")

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_log_error,
    d2_absolute_error_score
)


# ============================================================
# CONFIG
# ============================================================
FILE_PATH = "netflix_titles.csv"

TARGET_COLUMN = "release_year"

TEST_SIZE = 0.2
RANDOM_STATE = 42

STRONG_CORR_THRESHOLD = 0.85
RIDGE_ALPHAS = [1.0, 10.0, 100.0]

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (10, 6)


# ============================================================
# ДОПОМІЖНІ ФУНКЦІЇ
# ============================================================
def print_title(title: str):
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)


def print_conclusion(text: str):
    print(f"Висновок: {text}\n")


def save_plot(filename: str):
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Графік збережено: {filename}")


def check_required_columns(df: pd.DataFrame, required_columns: list):
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"У датасеті відсутні обов'язкові колонки: {missing}\n"
            f"Наявні колонки: {list(df.columns)}"
        )


def safe_msle(y_true, y_pred):
    y_true_clip = np.clip(np.asarray(y_true), 0, None)
    y_pred_clip = np.clip(np.asarray(y_pred), 0, None)
    return mean_squared_log_error(y_true_clip, y_pred_clip)


def parse_duration_minutes(duration_value: str) -> float:
    """
    Для Movie: '90 min' -> 90
    Для TV Show: не хвилини, тому NaN
    """
    if pd.isna(duration_value):
        return np.nan
    text = str(duration_value).strip().lower()
    match = re.search(r"(\d+)\s*min", text)
    if match:
        return float(match.group(1))
    return np.nan


def parse_seasons(duration_value: str) -> float:
    """
    Для TV Show: '2 Seasons' -> 2
    Для Movie: NaN
    """
    if pd.isna(duration_value):
        return np.nan
    text = str(duration_value).strip().lower()
    match = re.search(r"(\d+)\s*season", text)
    if match:
        return float(match.group(1))
    return np.nan


def count_items(cell_value: str) -> int:
    """
    Підрахунок елементів у рядку, розділених комами.
    """
    if pd.isna(cell_value) or str(cell_value).strip() == "":
        return 0
    return len([item.strip() for item in str(cell_value).split(",") if item.strip()])


def remove_outliers_iqr(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    result = df.copy()
    for col in columns:
        q1 = result[col].quantile(0.25)
        q3 = result[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        result = result[(result[col] >= lower) & (result[col] <= upper)]
    return result.copy()


def choose_features_by_correlation(corr_matrix: pd.DataFrame, target: str, threshold: float = 0.85):
    ordered = (
        corr_matrix[target]
        .drop(target)
        .abs()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    selected = []
    for feature in ordered:
        too_correlated = False
        for chosen in selected:
            if abs(corr_matrix.loc[feature, chosen]) >= threshold:
                too_correlated = True
                break
        if not too_correlated:
            selected.append(feature)

    return selected


def evaluate_regression(model, X_train, X_test, y_train, y_test):
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    metrics_df = pd.DataFrame(
        {
            "R2_score": [
                model.score(X_train, y_train),
                model.score(X_test, y_test)
            ],
            "MAE": [
                mean_absolute_error(y_train, y_pred_train),
                mean_absolute_error(y_test, y_pred_test)
            ],
            "MSLE": [
                safe_msle(y_train, y_pred_train),
                safe_msle(y_test, y_pred_test)
            ],
            "D2_absolute_error": [
                d2_absolute_error_score(y_train, y_pred_train),
                d2_absolute_error_score(y_test, y_pred_test)
            ]
        },
        index=["train", "test"]
    )

    return metrics_df, y_pred_train, y_pred_test


def transform_xy_power(X_train, X_test, y_train, y_test):
    x_transformer = PowerTransformer(method="yeo-johnson", standardize=True)
    y_transformer = PowerTransformer(method="yeo-johnson", standardize=True)

    X_train_t = x_transformer.fit_transform(X_train)
    X_test_t = x_transformer.transform(X_test)

    y_train_arr = np.asarray(y_train).reshape(-1, 1)
    y_test_arr = np.asarray(y_test).reshape(-1, 1)

    y_train_t = y_transformer.fit_transform(y_train_arr).ravel()
    y_test_t = y_transformer.transform(y_test_arr).ravel()

    return X_train_t, X_test_t, y_train_t, y_test_t, x_transformer, y_transformer


def inverse_target_predictions(y_pred_transformed, y_transformer):
    return y_transformer.inverse_transform(
        np.asarray(y_pred_transformed).reshape(-1, 1)
    ).ravel()


def evaluate_transformed_model(model, X_train_t, X_test_t, y_train_t, y_test_t,
                               y_train_original, y_test_original, y_transformer):
    y_pred_train_t = model.predict(X_train_t)
    y_pred_test_t = model.predict(X_test_t)

    y_pred_train = inverse_target_predictions(y_pred_train_t, y_transformer)
    y_pred_test = inverse_target_predictions(y_pred_test_t, y_transformer)

    metrics_df = pd.DataFrame(
        {
            "R2_score": [
                model.score(X_train_t, y_train_t),
                model.score(X_test_t, y_test_t)
            ],
            "MAE": [
                mean_absolute_error(y_train_original, y_pred_train),
                mean_absolute_error(y_test_original, y_pred_test)
            ],
            "MSLE": [
                safe_msle(y_train_original, y_pred_train),
                safe_msle(y_test_original, y_pred_test)
            ],
            "D2_absolute_error": [
                d2_absolute_error_score(y_train_original, y_pred_train),
                d2_absolute_error_score(y_test_original, y_pred_test)
            ]
        },
        index=["train", "test"]
    )

    return metrics_df, y_pred_train, y_pred_test


def plot_predictions_scatter(
    X_train_feature,
    X_test_feature,
    y_train,
    y_test,
    y_pred_train,
    y_pred_test,
    feature_name: str,
    target_name: str,
    title: str,
    filename: str
):
    plt.figure(figsize=(10, 6))

    plt.scatter(
        X_train_feature, y_train,
        alpha=0.6,
        label="Train actual",
        marker="o"
    )
    plt.scatter(
        X_test_feature, y_test,
        alpha=0.6,
        label="Test actual",
        marker="s"
    )

    sort_idx_train = np.argsort(X_train_feature)
    sort_idx_test = np.argsort(X_test_feature)

    plt.plot(
        np.array(X_train_feature)[sort_idx_train],
        np.array(y_pred_train)[sort_idx_train],
        linewidth=2,
        label="Train prediction"
    )
    plt.plot(
        np.array(X_test_feature)[sort_idx_test],
        np.array(y_pred_test)[sort_idx_test],
        linewidth=2,
        linestyle="--",
        label="Test prediction"
    )

    plt.title(title)
    plt.xlabel(feature_name)
    plt.ylabel(target_name)
    plt.legend()
    plt.grid(True)
    save_plot(filename)
    plt.show()


# ============================================================
# 1.1 ВИБІР ТА ЗАВАНТАЖЕННЯ НАБОРУ ДАНИХ
# ============================================================
print_title("1.1 ВИБІР ТА ЗАВАНТАЖЕННЯ НАБОРУ ДАНИХ")

df = pd.read_csv(FILE_PATH)
df.columns = df.columns.str.strip()

print("Перші 5 рядків датасету:")
print(df.head())

required_columns = [
    "show_id", "type", "title", "director", "cast", "country",
    "date_added", "release_year", "rating", "duration",
    "listed_in", "description"
]
check_required_columns(df, required_columns)

print("\nРозмір датасету:", df.shape)
print("\nКолонки датасету:")
print(df.columns.tolist())

print_conclusion(
    "Датасет Netflix успішно завантажено. Цільовою змінною для регресії буде рік випуску фільму або серіалу."
)


# ============================================================
# 1.2 БАЗОВА ПІДГОТОВКА ДАНИХ
# ============================================================
print_title("1.2 БАЗОВА ПІДГОТОВКА ДАНИХ")

# Базове очищення тексту
text_columns = [
    "show_id", "type", "title", "director", "cast", "country",
    "date_added", "rating", "duration", "listed_in", "description"
]
for col in text_columns:
    df[col] = df[col].astype(str).str.strip()

# Приведення release_year
df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")

# Ознаки
df["duration_minutes"] = df["duration"].apply(parse_duration_minutes)
df["seasons_count"] = df["duration"].apply(parse_seasons)
df["cast_count"] = df["cast"].apply(count_items)
df["country_count"] = df["country"].apply(count_items)
df["genre_count"] = df["listed_in"].apply(count_items)
df["director_count"] = df["director"].apply(count_items)
df["title_length"] = df["title"].astype(str).str.len()
df["description_length"] = df["description"].astype(str).str.len()
df["is_movie"] = (df["type"].str.lower() == "movie").astype(int)
df["has_director"] = (df["director"].astype(str).str.strip() != "").astype(int)

# Дата додавання
df["date_added_parsed"] = pd.to_datetime(df["date_added"], errors="coerce")
df["date_added_year"] = df["date_added_parsed"].dt.year
df["date_added_month"] = df["date_added_parsed"].dt.month

# Вибір числових колонок
candidate_features = [
    "duration_minutes",
    "seasons_count",
    "cast_count",
    "country_count",
    "genre_count",
    "director_count",
    "title_length",
    "description_length",
    "is_movie",
    "has_director",
    "date_added_year",
    "date_added_month"
]

# Заповнення пропусків
for col in [TARGET_COLUMN] + candidate_features:
    if df[col].isna().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

print("Кількість пропусків після обробки:")
print(df[[TARGET_COLUMN] + candidate_features].isna().sum())

# Для регресії беремо лише числові ознаки
df_model = df[[TARGET_COLUMN] + candidate_features].copy()

# Видалення аномалій
df_model = remove_outliers_iqr(df_model, [TARGET_COLUMN] + candidate_features)

print("\nРозмір датасету після відбору числових полів і видалення аномалій:")
print(df_model.shape)

print_conclusion(
    "Для моделювання сформовано числові ознаки з текстових полів: тривалість, кількість сезонів, кількість акторів, жанрів, країн, режисерів і довжини текстів."
)


# ============================================================
# 1.3 ЗНАХОДЖЕННЯ ЗАЛЕЖНОСТЕЙ
# ============================================================
print_title("1.3 ЗНАХОДЖЕННЯ ЗАЛЕЖНОСТЕЙ")

corr_matrix = df_model.corr()

ordered_corr = (
    corr_matrix[TARGET_COLUMN]
    .drop(TARGET_COLUMN)
    .sort_values(key=np.abs, ascending=False)
)

plt.figure(figsize=(11, 8))
sns.heatmap(
    corr_matrix.loc[[TARGET_COLUMN] + ordered_corr.index.tolist(),
                    [TARGET_COLUMN] + ordered_corr.index.tolist()],
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    linewidths=0.5
)
plt.title("Матриця кореляцій для числових ознак Netflix")
save_plot("lab3_netflix_corr_heatmap.png")
plt.show()

print("Ознаки у порядку сили зв'язку із цільовою змінною:")
print(ordered_corr)

selected_features = choose_features_by_correlation(
    corr_matrix.loc[[TARGET_COLUMN] + ordered_corr.index.tolist(),
                    [TARGET_COLUMN] + ordered_corr.index.tolist()],
    TARGET_COLUMN,
    STRONG_CORR_THRESHOLD
)

# Щоб було 5-10 ознак, як вимагає лабораторна
if len(selected_features) > 10:
    selected_features = selected_features[:10]

if len(selected_features) < 5:
    for col in ordered_corr.index.tolist():
        if col not in selected_features:
            selected_features.append(col)
        if len(selected_features) >= 5:
            break

print("\nОзнаки після усунення сильної взаємної кореляції:")
print(selected_features)

X = df_model[selected_features].copy()
y = df_model[TARGET_COLUMN].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

most_related_feature = ordered_corr.index[0]

print(f"\nНайбільш пов'язана ознака з цільовою змінною: {most_related_feature}")
print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")

print_conclusion(
    "Сильною кореляцією між ознаками у цій роботі вважаємо |r| ≥ 0.85. Для train/test використано поділ 80/20, що є раціональним для великого датасету."
)


# ============================================================
# 2.1 ПРОСТА ЛІНІЙНА РЕГРЕСІЯ
# ============================================================
print_title("2.1 ПРОСТА ЛІНІЙНА РЕГРЕСІЯ")

linreg = LinearRegression()
linreg.fit(X_train, y_train)

linreg_metrics, linreg_pred_train, linreg_pred_test = evaluate_regression(
    linreg, X_train, X_test, y_train, y_test
)

print("Метрики LinearRegression:")
print(linreg_metrics)

plot_predictions_scatter(
    X_train[most_related_feature].values,
    X_test[most_related_feature].values,
    y_train.values,
    y_test.values,
    linreg_pred_train,
    linreg_pred_test,
    feature_name=most_related_feature,
    target_name=TARGET_COLUMN,
    title="LinearRegression: реальні значення і прогнози",
    filename="lab3_netflix_linear_scatter.png"
)

print_conclusion(
    "Базова лінійна регресія показує, наскільки добре побудовані числові ознаки пояснюють рік випуску контенту."
)


# ============================================================
# 2.2 ЛІНІЙНА РЕГРЕСІЯ З ПЕРЕТВОРЕННЯМ ЗМІННИХ
# ============================================================
print_title("2.2 ЛІНІЙНА РЕГРЕСІЯ З ПЕРЕТВОРЕННЯМ ЗМІННИХ")

X_train_t, X_test_t, y_train_t, y_test_t, x_pt, y_pt = transform_xy_power(
    X_train, X_test, y_train, y_test
)

linreg_t = LinearRegression()
linreg_t.fit(X_train_t, y_train_t)

linreg_t_metrics, linreg_t_pred_train, linreg_t_pred_test = evaluate_transformed_model(
    linreg_t,
    X_train_t, X_test_t, y_train_t, y_test_t,
    y_train.values, y_test.values, y_pt
)

print("Метрики LinearRegression + transform:")
print(linreg_t_metrics)

plot_predictions_scatter(
    X_train[most_related_feature].values,
    X_test[most_related_feature].values,
    y_train.values,
    y_test.values,
    linreg_t_pred_train,
    linreg_t_pred_test,
    feature_name=most_related_feature,
    target_name=TARGET_COLUMN,
    title="LinearRegression + transform: реальні значення і прогнози",
    filename="lab3_netflix_linear_transform_scatter.png"
)

print_conclusion(
    "Перетворення Yeo-Johnson приводить розподіли змінних ближче до нормальних і може змінити якість моделі."
)


# ============================================================
# 2.3 ГРЕБЕНЕВА РЕГРЕСІЯ
# ============================================================
print_title("2.3 ГРЕБЕНЕВА РЕГРЕСІЯ")

ridge_metrics_all = {}
ridge_predictions = {}

for alpha in RIDGE_ALPHAS:
    model_name = f"Ridge_alpha_{alpha}"
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)

    metrics_df, pred_train, pred_test = evaluate_regression(
        model, X_train, X_test, y_train, y_test
    )

    ridge_metrics_all[model_name] = metrics_df
    ridge_predictions[model_name] = (pred_train, pred_test)

    print(f"\nМетрики {model_name}:")
    print(metrics_df)

plt.figure(figsize=(10, 6))
plt.scatter(X_train[most_related_feature], y_train, alpha=0.5, label="Train actual", marker="o")
plt.scatter(X_test[most_related_feature], y_test, alpha=0.5, label="Test actual", marker="s")

for model_name, (pred_train, pred_test) in ridge_predictions.items():
    sort_idx = np.argsort(X_train[most_related_feature].values)
    plt.plot(
        X_train[most_related_feature].values[sort_idx],
        pred_train[sort_idx],
        linewidth=2,
        label=f"{model_name} train"
    )

plt.title("Ridge: порівняння прогнозів для різних alpha")
plt.xlabel(most_related_feature)
plt.ylabel(TARGET_COLUMN)
plt.legend(fontsize=8)
plt.grid(True)
save_plot("lab3_netflix_ridge_scatter.png")
plt.show()

print_conclusion(
    "Ridge-регресія дозволяє перевірити, як регуляризація впливає на стабільність коефіцієнтів і якість прогнозування."
)


# ============================================================
# 2.4 ГРЕБЕНЕВА РЕГРЕСІЯ З ПЕРЕТВОРЕННЯМ ЗМІННИХ
# ============================================================
print_title("2.4 ГРЕБЕНЕВА РЕГРЕСІЯ З ПЕРЕТВОРЕННЯМ ЗМІННИХ")

ridge_t_metrics_all = {}
ridge_t_predictions = {}

for alpha in RIDGE_ALPHAS:
    model_name = f"Ridge_transform_alpha_{alpha}"
    model = Ridge(alpha=alpha)
    model.fit(X_train_t, y_train_t)

    metrics_df, pred_train, pred_test = evaluate_transformed_model(
        model,
        X_train_t, X_test_t, y_train_t, y_test_t,
        y_train.values, y_test.values, y_pt
    )

    ridge_t_metrics_all[model_name] = metrics_df
    ridge_t_predictions[model_name] = (pred_train, pred_test)

    print(f"\nМетрики {model_name}:")
    print(metrics_df)

plt.figure(figsize=(10, 6))
plt.scatter(X_train[most_related_feature], y_train, alpha=0.5, label="Train actual", marker="o")
plt.scatter(X_test[most_related_feature], y_test, alpha=0.5, label="Test actual", marker="s")

for model_name, (pred_train, pred_test) in ridge_t_predictions.items():
    sort_idx = np.argsort(X_train[most_related_feature].values)
    plt.plot(
        X_train[most_related_feature].values[sort_idx],
        pred_train[sort_idx],
        linewidth=2,
        label=f"{model_name} train"
    )

plt.title("Ridge + transform: порівняння прогнозів для різних alpha")
plt.xlabel(most_related_feature)
plt.ylabel(TARGET_COLUMN)
plt.legend(fontsize=8)
plt.grid(True)
save_plot("lab3_netflix_ridge_transform_scatter.png")
plt.show()

print_conclusion(
    "Поєднання регуляризації і перетворення змінних дає змогу оцінити, чи покращується узагальнювальна здатність моделі."
)


# ============================================================
# 2.5 СУКУПНЕ ПОРІВНЯННЯ МЕТРИК
# ============================================================
print_title("2.5 СУКУПНЕ ПОРІВНЯННЯ МЕТРИК")

all_results = []

for split_name, row in linreg_metrics.iterrows():
    all_results.append({
        "experiment": "LinearRegression",
        "split": split_name,
        **row.to_dict()
    })

for split_name, row in linreg_t_metrics.iterrows():
    all_results.append({
        "experiment": "LinearRegression + transform",
        "split": split_name,
        **row.to_dict()
    })

for exp_name, metrics_df in ridge_metrics_all.items():
    for split_name, row in metrics_df.iterrows():
        all_results.append({
            "experiment": exp_name,
            "split": split_name,
            **row.to_dict()
        })

for exp_name, metrics_df in ridge_t_metrics_all.items():
    for split_name, row in metrics_df.iterrows():
        all_results.append({
            "experiment": exp_name,
            "split": split_name,
            **row.to_dict()
        })

results_df = pd.DataFrame(all_results)
results_df = results_df.set_index(["experiment", "split"]).sort_index()

print("Спільний датафрейм метрик:")
print(results_df)

styled_results = (
    results_df.style
    .background_gradient(cmap="RdYlGn", subset=["R2_score", "D2_absolute_error"])
    .background_gradient(cmap="RdYlGn_r", subset=["MAE", "MSLE"])
    .format("{:.6f}")
)

styled_results.to_html("lab3_netflix_metrics_comparison.html")
results_df.to_csv("lab3_netflix_metrics_comparison.csv", encoding="utf-8-sig")

print("\nФайли з метриками збережено:")
print("- lab3_netflix_metrics_comparison.html")
print("- lab3_netflix_metrics_comparison.csv")

plt.figure(figsize=(14, 6))
results_reset = results_df.reset_index()
sns.barplot(
    data=results_reset[results_reset["split"] == "test"],
    x="experiment",
    y="R2_score"
)
plt.title("Порівняння R² на тестовій вибірці")
plt.xlabel("Модель")
plt.ylabel("R²")
plt.xticks(rotation=45, ha="right")
save_plot("lab3_netflix_test_r2.png")
plt.show()

print_conclusion(
    "Зведена таблиця дозволяє порівняти всі експерименти одночасно. Для R² та D² більші значення кращі, а для MAE і MSLE — менші."
)


# ============================================================
# ЗАВЕРШЕННЯ
# ============================================================
print_title("РОБОТУ ЗАВЕРШЕНО")
print("Усі пункти лабораторної роботи №3 виконані успішно.")