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
    max_error,
    d2_absolute_error_score
)


FILE_PATH = "netflix_titles.csv"   # зміни шлях, якщо треба
TARGET_COLUMN = "release_year"

TEST_SIZE = 0.2
RANDOM_STATE = 42
STRONG_CORR_THRESHOLD = 0.85

# default Ridge + 2 довільні alpha
RIDGE_MODELS = [
    ("Ridge_default", Ridge()),
    ("Ridge_alpha_10", Ridge(alpha=10.0)),
    ("Ridge_alpha_100", Ridge(alpha=100.0)),
]

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (11, 6)


def print_title(title: str):
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)

def print_conclusion(text: str):
    print(f"\nВисновок: {text}\n")

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
    Для TV Show: NaN
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
    if pd.isna(cell_value):
        return 0
    text = str(cell_value).strip()
    if text == "" or text.lower() == "nan":
        return 0
    return len([item.strip() for item in text.split(",") if item.strip()])

def remove_outliers_iqr(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Послідовне видалення аномалій методом IQR по кожній колонці.
    """
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
    """
    Обирає ознаки, прибираючи ті, що дуже сильно корелюють між собою.
    """
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
            "MaxError": [
                max_error(y_train, y_pred_train),
                max_error(y_test, y_pred_test)
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
    """
    Перетворення усіх змінних до розподілу, близького до нормального.
    """
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
    """
    .score рахується в трансформованому просторі (бо так працює модель),
    а абсолютні метрики — в оригінальному масштабі цільової змінної.
    """
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
            "MaxError": [
                max_error(y_train_original, y_pred_train),
                max_error(y_test_original, y_pred_test)
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
    """
    В одній системі координат:
    - точки train actual
    - точки test actual
    - лінія train prediction
    - лінія test prediction
    """
    plt.figure(figsize=(11, 6))

    X_train_feature = np.asarray(X_train_feature)
    X_test_feature = np.asarray(X_test_feature)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    y_pred_train = np.asarray(y_pred_train)
    y_pred_test = np.asarray(y_pred_test)

    plt.scatter(
        X_train_feature, y_train,
        alpha=0.65,
        label="Train actual",
        marker="o"
    )
    plt.scatter(
        X_test_feature, y_test,
        alpha=0.65,
        label="Test actual",
        marker="s"
    )

    sort_idx_train = np.argsort(X_train_feature)
    sort_idx_test = np.argsort(X_test_feature)

    plt.plot(
        X_train_feature[sort_idx_train],
        y_pred_train[sort_idx_train],
        linewidth=2,
        label="Train prediction"
    )
    plt.plot(
        X_test_feature[sort_idx_test],
        y_pred_test[sort_idx_test],
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

def add_metrics_to_list(storage_list, experiment_name, metrics_df):
    for split_name, row in metrics_df.iterrows():
        storage_list.append({
            "experiment": experiment_name,
            "split": split_name,
            **row.to_dict()
        })


print_title("1.1 ВИБІР ТА ЗАВАНТАЖЕННЯ НАБОРУ ДАНИХ")

df = pd.read_csv(FILE_PATH)
df.columns = df.columns.str.strip()

required_columns = [
    "show_id", "type", "title", "director", "cast", "country",
    "date_added", "release_year", "rating", "duration",
    "listed_in", "description"
]
check_required_columns(df, required_columns)

print("Перші 5 рядків датасету:")
print(df.head())

print("\nРозмір датасету:", df.shape)
print("\nКолонки датасету:")
print(df.columns.tolist())

print_conclusion(
    "Датасет успішно завантажено. Він містить 8807 рядків, що задовольняє вимогу лабораторної щодо кількості спостережень."
)


print_title("1.2 БАЗОВА ПІДГОТОВКА ДАНИХ")

# Базове очищення тексту
text_columns = [
    "show_id", "type", "title", "director", "cast", "country",
    "date_added", "rating", "duration", "listed_in", "description"
]

for col in text_columns:
    df[col] = df[col].astype(str).str.strip()

# Приведення типу цілі
df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")

# Створення числових ознак
df["duration_minutes"] = df["duration"].apply(parse_duration_minutes)
df["seasons_count"] = df["duration"].apply(parse_seasons)
df["cast_count"] = df["cast"].apply(count_items)
df["country_count"] = df["country"].apply(count_items)
df["genre_count"] = df["listed_in"].apply(count_items)
df["director_count"] = df["director"].apply(count_items)
df["title_length"] = df["title"].astype(str).str.len()
df["description_length"] = df["description"].astype(str).str.len()
df["is_movie"] = (df["type"].str.lower() == "movie").astype(int)
df["has_director"] = (
    df["director"].astype(str).str.strip().replace("nan", "") != ""
).astype(int)

# Розбір дати додавання
df["date_added_parsed"] = pd.to_datetime(df["date_added"], errors="coerce")
df["date_added_year"] = df["date_added_parsed"].dt.year
df["date_added_month"] = df["date_added_parsed"].dt.month

# Від 5 до 10 числових ознак
candidate_features = [
    "duration_minutes",
    "seasons_count",
    "cast_count",
    "country_count",
    "genre_count",
    "director_count",
    "title_length",
    "description_length",
    "date_added_year",
    "date_added_month"
]

print("Вибрані числові ознаки:")
print(candidate_features)
print("Кількість вибраних числових ознак:", len(candidate_features))

# Обробка пропусків
for col in [TARGET_COLUMN] + candidate_features:
    if df[col].isna().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

print("\nКількість пропусків після обробки:")
print(df[[TARGET_COLUMN] + candidate_features].isna().sum())

# Модельний датафрейм
df_model = df[[TARGET_COLUMN] + candidate_features].copy()

print("\nРозмір до видалення аномалій:", df_model.shape)
df_model = remove_outliers_iqr(df_model, [TARGET_COLUMN] + candidate_features)
print("Розмір після видалення аномалій:", df_model.shape)

print_conclusion(
    "Для моделювання сформовано 10 числових ознак. Пропуски заповнено медіаною, а аномалії видалено методом IQR."
)


print_title("1.3 ЗНАХОДЖЕННЯ ЗАЛЕЖНОСТЕЙ")

corr_matrix = df_model.corr(numeric_only=True)

ordered_corr = (
    corr_matrix[TARGET_COLUMN]
    .drop(TARGET_COLUMN)
    .sort_values(key=np.abs, ascending=False)
)

plt.figure(figsize=(12, 8))
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
plt.title("Матриця кореляцій для числових ознак")
save_plot("lab3_corr_heatmap.png")
plt.show()

print("Ознаки у порядку сили зв'язку із цільовою змінною:")
print(ordered_corr)

selected_features = choose_features_by_correlation(
    corr_matrix.loc[[TARGET_COLUMN] + ordered_corr.index.tolist(),
                    [TARGET_COLUMN] + ordered_corr.index.tolist()],
    TARGET_COLUMN,
    STRONG_CORR_THRESHOLD
)

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

# Для простих моделей беремо одну найбільш пов'язану ознаку
most_related_feature = ordered_corr.index[0]
print(f"\nНайбільш пов'язана ознака з цільовою змінною: {most_related_feature}")

# simple linear regression -> одна ознака
X_simple = df_model[[most_related_feature]].copy()
y = df_model[TARGET_COLUMN].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X_simple, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")

print_conclusion(
    f"Сильною кореляцією у роботі вважаємо |r| ≥ {STRONG_CORR_THRESHOLD}. "
    f"Для train/test використано поділ 80/20. "
    f"Для простої регресії використано одну ознаку: {most_related_feature}."
)


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
    title="2.1 LinearRegression: реальні значення і прогнози",
    filename="lab3_linear_scatter.png"
)

print_conclusion(
    "Побудовано просту лінійну регресію на одній найсильніше пов'язаній ознаці."
)

print_title("2.2 ПРОСТА ЛІНІЙНА РЕГРЕСІЯ З ПЕРЕТВОРЕННЯМ ЗМІННИХ")

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
    title="2.2 LinearRegression + transform: реальні значення і прогнози",
    filename="lab3_linear_transform_scatter.png"
)

print_conclusion(
    "Перед побудовою моделі виконано перетворення X та y методом Yeo-Johnson."
)


print_title("2.3 ГРЕБЕНЕВА РЕГРЕСІЯ")

ridge_metrics_all = {}
ridge_predictions = {}

for model_name, model in RIDGE_MODELS:
    model.fit(X_train, y_train)

    metrics_df, pred_train, pred_test = evaluate_regression(
        model, X_train, X_test, y_train, y_test
    )

    ridge_metrics_all[model_name] = metrics_df
    ridge_predictions[model_name] = (pred_train, pred_test)

    print(f"\nМетрики {model_name}:")
    print(metrics_df)

    plot_predictions_scatter(
        X_train[most_related_feature].values,
        X_test[most_related_feature].values,
        y_train.values,
        y_test.values,
        pred_train,
        pred_test,
        feature_name=most_related_feature,
        target_name=TARGET_COLUMN,
        title=f"2.3 {model_name}: реальні значення і прогнози",
        filename=f"lab3_{model_name.lower()}_scatter.png"
    )

print_conclusion(
    "Побудовано три моделі Ridge: з параметрами за замовчуванням, alpha=10 та alpha=100."
)


print_title("2.4 ГРЕБЕНЕВА РЕГРЕСІЯ З ПЕРЕТВОРЕННЯМ ЗМІННИХ")

ridge_t_metrics_all = {}
ridge_t_predictions = {}

for model_name, model in RIDGE_MODELS:
    transformed_name = model_name + "_transform"
    model.fit(X_train_t, y_train_t)

    metrics_df, pred_train, pred_test = evaluate_transformed_model(
        model,
        X_train_t, X_test_t, y_train_t, y_test_t,
        y_train.values, y_test.values, y_pt
    )

    ridge_t_metrics_all[transformed_name] = metrics_df
    ridge_t_predictions[transformed_name] = (pred_train, pred_test)

    print(f"\nМетрики {transformed_name}:")
    print(metrics_df)

    plot_predictions_scatter(
        X_train[most_related_feature].values,
        X_test[most_related_feature].values,
        y_train.values,
        y_test.values,
        pred_train,
        pred_test,
        feature_name=most_related_feature,
        target_name=TARGET_COLUMN,
        title=f"2.4 {transformed_name}: реальні значення і прогнози",
        filename=f"lab3_{transformed_name.lower()}_scatter.png"
    )

print_conclusion(
    "Побудовано три Ridge-моделі після перетворення X та y до розподілу, близького до нормального."
)


print_title("2.5 СУКУПНЕ ПОРІВНЯННЯ МЕТРИК")

all_results = []

add_metrics_to_list(all_results, "LinearRegression", linreg_metrics)
add_metrics_to_list(all_results, "LinearRegression_transform", linreg_t_metrics)

for exp_name, metrics_df in ridge_metrics_all.items():
    add_metrics_to_list(all_results, exp_name, metrics_df)

for exp_name, metrics_df in ridge_t_metrics_all.items():
    add_metrics_to_list(all_results, exp_name, metrics_df)

results_df = pd.DataFrame(all_results)
results_df = results_df.set_index(["experiment", "split"]).sort_index()

print("Усі метрики:")
print(results_df)

# 1) Абсолютні метрики
absolute_metrics_df = results_df[["MAE", "MSLE", "MaxError"]].copy()

# 2) Відносні метрики
relative_metrics_df = results_df[["R2_score", "D2_absolute_error"]].copy()

print("\nАбсолютні метрики:")
print(absolute_metrics_df)

print("\nВідносні метрики:")
print(relative_metrics_df)

# Стилізація
absolute_styled = (
    absolute_metrics_df.style
    .background_gradient(cmap="RdYlGn_r")
    .format("{:.6f}")
)

relative_styled = (
    relative_metrics_df.style
    .background_gradient(cmap="RdYlGn")
    .format("{:.6f}")
)

# Збереження
absolute_metrics_df.to_csv("lab3_absolute_metrics.csv", encoding="utf-8-sig")
relative_metrics_df.to_csv("lab3_relative_metrics.csv", encoding="utf-8-sig")

with open("lab3_absolute_metrics.html", "w", encoding="utf-8") as f:
    f.write(absolute_styled.to_html())

with open("lab3_relative_metrics.html", "w", encoding="utf-8") as f:
    f.write(relative_styled.to_html())

print("\nФайли збережено:")
print("- lab3_absolute_metrics.csv")
print("- lab3_relative_metrics.csv")
print("- lab3_absolute_metrics.html")
print("- lab3_relative_metrics.html")

print_conclusion(
    "Створено два окремі датафрейми: один для абсолютних метрик, другий — для відносних. "
    "Для відносних метрик кращими є більші значення, для абсолютних — менші."
)

