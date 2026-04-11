import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# ============================================================
# CONFIG
# ============================================================
FILE_PATH = "global_fuel_prices.csv"

TEXT_COLUMNS = [
    "country",
    "region",
    "iso3",
    "local_currency"
]

NUMERIC_COLUMNS = [
    "gasoline_usd_per_liter",
    "diesel_usd_per_liter",
    "gasoline_local_price",
    "diesel_local_price",
    "avg_fuel_usd"
]

OPTIONAL_COLUMNS = [
    "price_date",
    "is_asian"
]


# ============================================================
# STYLE
# ============================================================
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (10, 6)


# ============================================================
# ДОПОМІЖНІ ФУНКЦІЇ
# ============================================================
def print_title(title: str):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def print_conclusion(text: str):
    print(f"Висновок: {text}\n")


def check_columns(df: pd.DataFrame, required_columns: list):
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"У датасеті відсутні такі колонки: {missing}\n"
            f"Наявні колонки: {list(df.columns)}"
        )


def save_plot(filename: str):
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Графік збережено: {filename}")


def remove_outliers_iqr(df: pd.DataFrame, column: str) -> pd.DataFrame:
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return df[(df[column] >= lower) & (df[column] <= upper)].copy()


# ============================================================
# 1.1 ЗАВАНТАЖЕННЯ ДАНИХ
# ============================================================
print_title("1.1 ВИБІР ТА ЗАВАНТАЖЕННЯ НАБОРУ ДАНИХ")

df = pd.read_csv(FILE_PATH)
df.columns = df.columns.str.strip()

print("Перші 5 рядків набору даних:")
print(df.head())

required_columns = TEXT_COLUMNS + NUMERIC_COLUMNS
check_columns(df, required_columns)

print("\nНазви колонок:")
print(df.columns.tolist())

print_conclusion(
    "Набір даних успішно завантажено. Він містить текстові ознаки країн і регіонів, а також числові показники цін на пальне."
)


# ============================================================
# 1.2 БАЗОВА ПІДГОТОВКА ДАНИХ
# ============================================================
print_title("1.2 БАЗОВА ПІДГОТОВКА ДАНИХ")

# Очищення текстових колонок
for col in TEXT_COLUMNS:
    df[col] = df[col].astype(str).str.strip()

# Числові колонки
for col in NUMERIC_COLUMNS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Обробка optional колонок
if "price_date" in df.columns:
    df["price_date"] = pd.to_datetime(df["price_date"], errors="coerce")

if "is_asian" in df.columns:
    df["is_asian"] = (
        df["is_asian"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({
            "1": True,
            "0": False,
            "true": True,
            "false": False,
            "yes": True,
            "no": False
        })
    )

missing_before = df.isna().sum().sum()
print(f"Загальна кількість пропущених значень до обробки: {missing_before}")

# Заповнення пропусків для числових колонок медіаною
for col in NUMERIC_COLUMNS:
    if df[col].isna().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

# Заповнення пропусків для текстових колонок модою
for col in TEXT_COLUMNS:
    if df[col].isna().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

missing_after = df.isna().sum().sum()
print(f"Загальна кількість пропущених значень після обробки: {missing_after}")

print("\nТипи даних після підготовки:")
print(df.dtypes)

# Версія без викидів для частини графіків
df_no_outliers = remove_outliers_iqr(df, "avg_fuel_usd")

# Залишимо топ-10 регіонів за кількістю записів для деяких візуалізацій, якщо треба
top_regions = df["region"].value_counts().head(10).index
df_top_regions = df[df["region"].isin(top_regions)].copy()

print_conclusion(
    "Пропущені значення оброблено, типи даних приведено до коректного формату, а також підготовлено версію даних без викидів."
)


# ============================================================
# 2.1 РОЗПОДІЛ ОДНІЄЇ ЗМІННОЇ
# ============================================================
print_title("2.1 РОЗПОДІЛ ОДНІЄЇ ЗМІННОЇ")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(
    data=df_no_outliers,
    x="avg_fuel_usd",
    bins=20,
    color="skyblue",
    edgecolor="black",
    ax=axes[0]
)
axes[0].set_title("Гістограма розподілу середньої ціни пального")
axes[0].set_xlabel("Середня ціна пального, USD/л")
axes[0].set_ylabel("Кількість країн")

sns.kdeplot(
    data=df_no_outliers,
    x="avg_fuel_usd",
    fill=True,
    color="crimson",
    ax=axes[1]
)
axes[1].set_title("Графік щільності середньої ціни пального")
axes[1].set_xlabel("Середня ціна пального, USD/л")
axes[1].set_ylabel("Щільність")

save_plot("lab2_fuel_2_1_distribution.png")
plt.show()

print_conclusion(
    "Гістограма та графік щільності показують, як у світі розподіляються середні ціни на пальне, та дозволяють побачити, які значення трапляються найчастіше."
)

plt.figure(figsize=(10, 6))
sns.histplot(
    data=df_top_regions,
    x="avg_fuel_usd",
    hue="region",
    bins=15,
    multiple="layer",
    alpha=0.45
)
plt.title("Розподіл середньої ціни пального за регіонами")
plt.xlabel("Середня ціна пального, USD/л")
plt.ylabel("Кількість країн")
plt.legend(title="Регіон", bbox_to_anchor=(1.05, 1), loc="upper left")
save_plot("lab2_fuel_2_1_group_hist.png")
plt.show()

print_conclusion(
    "Групова гістограма дозволяє порівняти, як відрізняється розподіл середніх цін на пальне між різними регіонами."
)


# ============================================================
# 2.2 ПОРІВНЯННЯ КАТЕГОРІЙ
# ============================================================
print_title("2.2 ПОРІВНЯННЯ КАТЕГОРІЙ")

region_platform = (
    df.groupby(["region", "local_currency"])["avg_fuel_usd"]
    .mean()
    .reset_index()
)

# залишаємо 10 найбільших регіонів і 5 найчастіших валют для читабельності
top_currencies = df["local_currency"].value_counts().head(5).index
region_platform = region_platform[
    region_platform["region"].isin(top_regions) &
    region_platform["local_currency"].isin(top_currencies)
]

plt.figure(figsize=(14, 7))
ax = sns.barplot(
    data=region_platform,
    x="region",
    y="avg_fuel_usd",
    hue="local_currency",
    estimator=np.mean,
    errorbar=None
)

plt.title("Середня ціна пального за регіонами та валютою")
plt.xlabel("Регіон")
plt.ylabel("Середня ціна пального, USD/л")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Місцева валюта", bbox_to_anchor=(1.05, 1), loc="upper left")
save_plot("lab2_fuel_2_2_bar.png")
plt.show()

print_conclusion(
    "Стовпчикова діаграма показує, як відрізняється середня ціна пального між регіонами, а колір дозволяє побачити додатковий розподіл за місцевою валютою."
)


# ============================================================
# 2.3 АНАЛІЗ РОЗКИДУ ТА ВИКИДІВ
# ============================================================
print_title("2.3 АНАЛІЗ РОЗКИДУ ТА ВИКИДІВ")

plt.figure(figsize=(14, 7))
sns.boxplot(
    data=df_top_regions,
    x="region",
    y="gasoline_usd_per_liter"
)
plt.title("Розподіл цін на бензин за регіонами")
plt.xlabel("Регіон")
plt.ylabel("Ціна бензину, USD/л")
plt.xticks(rotation=45, ha="right")
save_plot("lab2_fuel_2_3_boxplot.png")
plt.show()

print_conclusion(
    "Діаграма типу 'ящик з вусами' дає змогу порівняти медіани, міжквартильний розмах і можливі викиди цін на бензин у різних регіонах."
)


# ============================================================
# 2.4 СТРУКТУРА ТА ПРОПОРЦІЇ
# ============================================================
print_title("2.4 СТРУКТУРА ТА ПРОПОРЦІЇ")

pie_data = df.groupby("region")["avg_fuel_usd"].sum().sort_values(ascending=False).head(8)

plt.figure(figsize=(8, 8))
plt.pie(
    pie_data,
    labels=pie_data.index,
    autopct="%1.1f%%",
    startangle=90,
    wedgeprops={"width": 0.45}
)
plt.title("Частка регіонів у сумарній середній вартості пального")
save_plot("lab2_fuel_2_4_donut.png")
plt.show()

print_conclusion(
    "Кільцева діаграма відображає, яку частку в сумарній середній вартості пального формують окремі регіони."
)


# ============================================================
# 2.5 ВЗАЄМОЗВ'ЯЗОК ДВОХ ЧИСЛОВИХ ЗМІННИХ
# ============================================================
print_title("2.5 ВЗАЄМОЗВ'ЯЗОК ДВОХ ЧИСЛОВИХ ЗМІННИХ")

scatter_regions = df["region"].value_counts().head(6).index
df_scatter = df[df["region"].isin(scatter_regions)].copy()

plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df_scatter,
    x="gasoline_usd_per_liter",
    y="diesel_usd_per_liter",
    hue="region",
    alpha=0.75
)
plt.title("Залежність між ціною бензину та дизеля")
plt.xlabel("Ціна бензину, USD/л")
plt.ylabel("Ціна дизеля, USD/л")
plt.legend(title="Регіон", bbox_to_anchor=(1.05, 1), loc="upper left")
save_plot("lab2_fuel_2_5_scatter.png")
plt.show()

print_conclusion(
    "Точкова діаграма дозволяє оцінити, чи існує залежність між ціною бензину та дизеля, а розфарбування за регіонами показує територіальні відмінності."
)


# ============================================================
# 2.6 ДИНАМІКА АБО ТРЕНД
# ============================================================
print_title("2.6 ДИНАМІКА АБО ТРЕНД")

# Оскільки справжньої часової послідовності по багатьох датах може не бути,
# створюємо логічний порядок за відсортованими регіонами за середньою ціною
trend_df = (
    df.groupby("region")["avg_fuel_usd"]
    .mean()
    .sort_values()
    .reset_index()
)

plt.figure(figsize=(12, 6))
sns.lineplot(
    data=trend_df,
    x="region",
    y="avg_fuel_usd",
    marker="o"
)
plt.title("Зміна середньої ціни пального за регіонами у впорядкованому вигляді")
plt.xlabel("Регіон")
plt.ylabel("Середня ціна пального, USD/л")
plt.xticks(rotation=45, ha="right")
save_plot("lab2_fuel_2_6_line.png")
plt.show()

print_conclusion(
    "Лінійний графік показує, як змінюється середня ціна пального між регіонами у логічно впорядкованому вигляді — від нижчих значень до вищих."
)


# ============================================================
# 2.7 МАТРИЦЯ КОРЕЛЯЦІЙ
# ============================================================
print_title("2.7 МАТРИЦЯ КОРЕЛЯЦІЙ")

corr_columns = [
    "gasoline_usd_per_liter",
    "diesel_usd_per_liter",
    "gasoline_local_price",
    "diesel_local_price",
    "avg_fuel_usd"
]

corr_matrix = df[corr_columns].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.5
)
plt.title("Теплова карта матриці кореляцій цін на пальне")
save_plot("lab2_fuel_2_7_heatmap.png")
plt.show()

corr_pairs = corr_matrix.abs().unstack()
corr_pairs = corr_pairs[corr_pairs < 1]
max_corr = corr_pairs.max()

print_conclusion(
    f"Теплова карта показує силу взаємозв'язку між числовими показниками. Найбільша кореляція за модулем між різними змінними становить приблизно {max_corr:.2f}."
)


# ============================================================
# 2.8 КОМБІНОВАНИЙ ГРАФІК
# ============================================================
print_title("2.8 КОМБІНОВАНИЙ ГРАФІК")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Гістограма
sns.histplot(
    data=df_no_outliers,
    x="avg_fuel_usd",
    bins=15,
    color="skyblue",
    edgecolor="black",
    ax=axes[0, 0]
)
axes[0, 0].set_title("Розподіл середньої ціни пального")
axes[0, 0].set_xlabel("USD/л")
axes[0, 0].set_ylabel("Кількість країн")

# 2. Стовпчикова
avg_region = df.groupby("region")["avg_fuel_usd"].mean().sort_values(ascending=False).head(10)
sns.barplot(
    x=avg_region.index,
    y=avg_region.values,
    ax=axes[0, 1]
)
axes[0, 1].set_title("Топ-10 регіонів за середньою ціною")
axes[0, 1].set_xlabel("Регіон")
axes[0, 1].set_ylabel("USD/л")
axes[0, 1].tick_params(axis="x", rotation=45)

# 3. Лінійна
sns.lineplot(
    data=trend_df,
    x="region",
    y="avg_fuel_usd",
    marker="o",
    ax=axes[1, 0]
)
axes[1, 0].set_title("Упорядкована зміна середньої ціни")
axes[1, 0].set_xlabel("Регіон")
axes[1, 0].set_ylabel("USD/л")
axes[1, 0].tick_params(axis="x", rotation=45)

# 4. Scatter
sns.scatterplot(
    data=df_scatter,
    x="gasoline_usd_per_liter",
    y="diesel_usd_per_liter",
    hue="region",
    ax=axes[1, 1]
)
axes[1, 1].set_title("Бензин і дизель")
axes[1, 1].set_xlabel("Бензин, USD/л")
axes[1, 1].set_ylabel("Дизель, USD/л")
axes[1, 1].legend(title="Регіон", fontsize=8)

plt.suptitle("Комплексна панель візуального аналізу цін на пальне", fontsize=14)
save_plot("lab2_fuel_2_8_dashboard.png")
plt.show()

print_conclusion(
    "Комбінований графік об'єднує кілька типів аналізу: розподіл цін, порівняння регіонів, упорядкований тренд і зв'язок між цінами бензину та дизеля."
)


# ============================================================
# ЗАВЕРШЕННЯ
# ============================================================
print_title("РОБОТУ ЗАВЕРШЕНО")
