import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



FILE_PATH = "Teen_Mental_Health_Dataset.csv"

TEXT_COLUMNS = [
    "gender",
    "platform_usage",
    "social_interaction_level"
]

NUMERIC_COLUMNS = [
    "age",
    "daily_social_media_hours",
    "sleep_hours",
    "screen_time_before_sleep",
    "academic_performance",
    "physical_activity",
    "stress_level",
    "anxiety_level",
    "addiction_level"
]

OPTIONAL_COLUMNS = [
    "depression_label"
]



sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (10, 6)



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


def validate_lab_requirements(df: pd.DataFrame):
    row_count, col_count = df.shape

    text_cols_present = [col for col in TEXT_COLUMNS if col in df.columns]
    numeric_cols_present = [col for col in NUMERIC_COLUMNS if col in df.columns]

    print(f"Кількість рядків у наборі даних: {row_count}")
    print(f"Кількість ознак у наборі даних: {col_count}")
    print(f"Текстові ознаки, що використовуються: {text_cols_present}")
    print(f"Числові ознаки, що використовуються: {numeric_cols_present}")

    if row_count < 1000:
        raise ValueError("Набір даних не відповідає умові лабораторної: менше 1000 рядків.")
    if col_count < 5:
        raise ValueError("Набір даних не відповідає умові лабораторної: менше 5 ознак.")
    if len(text_cols_present) < 2:
        raise ValueError("Набір даних не відповідає умові лабораторної: менше 2 текстових ознак.")
    if len(numeric_cols_present) < 2:
        raise ValueError("Набір даних не відповідає умові лабораторної: менше 2 числових ознак.")



print_title("1.1 Вибір та завантаження набору даних")

df = pd.read_csv(FILE_PATH)
df.columns = df.columns.str.strip()

required_columns = TEXT_COLUMNS + NUMERIC_COLUMNS
check_columns(df, required_columns)

print("Перші 5 рядків набору даних:")
print(df.head())

print("\nНазви колонок:")
print(df.columns.tolist())

validate_lab_requirements(df)

print_conclusion(
    "Набір даних успішно завантажено. Він містить текстові та числові ознаки, придатні для аналізу впливу використання соціальних мереж на поведінкові та психологічні показники."
)



print_title("1.2 Базова підготовка змінних")

# Текстові колонки
for col in TEXT_COLUMNS:
    df[col] = df[col].replace(r"^\s*$", np.nan, regex=True)
    df[col] = df[col].astype("string").str.strip()

# Числові колонки
for col in NUMERIC_COLUMNS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Опціональна колонка depression_label
if "depression_label" in df.columns:
    df["depression_label"] = pd.to_numeric(df["depression_label"], errors="coerce")

missing_before = df.isna().sum().sum()
print(f"Загальна кількість пропущених значень до обробки: {missing_before}")

# Заповнення числових пропусків медіаною
for col in NUMERIC_COLUMNS:
    if df[col].isna().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

if "depression_label" in df.columns and df["depression_label"].isna().sum() > 0:
    df["depression_label"] = df["depression_label"].fillna(df["depression_label"].median())

# Заповнення текстових пропусків модою
for col in TEXT_COLUMNS:
    if df[col].isna().sum() > 0:
        mode_value = df[col].mode(dropna=True)
        if not mode_value.empty:
            df[col] = df[col].fillna(mode_value.iloc[0])
        else:
            df[col] = df[col].fillna("Unknown")

missing_after = df.isna().sum().sum()
print(f"Загальна кількість пропущених значень після обробки: {missing_after}")

print("\nТипи даних після підготовки:")
print(df.dtypes)

# Фільтрація викидів для окремих графіків
df_no_outliers = remove_outliers_iqr(df, "daily_social_media_hours")

# Топ платформ для кращої читабельності
top_platforms = df["platform_usage"].value_counts().head(6).index
df_top_platforms = df[df["platform_usage"].isin(top_platforms)].copy()

print_conclusion(
    "Пропущені значення оброблено, типи даних приведено до правильного формату, а для деяких візуалізацій додатково підготовлено вибірку без викидів."
)



print_title("2.1 Розподіл однієї змінної")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(
    data=df_no_outliers,
    x="daily_social_media_hours",
    bins=20,
    color="skyblue",
    edgecolor="black",
    ax=axes[0]
)
axes[0].set_title("Гістограма щоденного часу використання соцмереж")
axes[0].set_xlabel("Години використання соцмереж на день")
axes[0].set_ylabel("Кількість учасників")

sns.kdeplot(
    data=df_no_outliers,
    x="daily_social_media_hours",
    fill=True,
    color="crimson",
    ax=axes[1]
)
axes[1].set_title("Графік щільності щоденного часу використання соцмереж")
axes[1].set_xlabel("Години використання соцмереж на день")
axes[1].set_ylabel("Щільність")

save_plot("lab2_2_1_distribution.png")
plt.show()

print_conclusion(
    "Розподіл показує, скільки часу респонденти щодня проводять у соціальних мережах. Основна частина спостережень зосереджена в середньому діапазоні значень, а дуже великі значення трапляються значно рідше."
)

plt.figure(figsize=(10, 6))
ax = sns.histplot(
    data=df_top_platforms,
    x="daily_social_media_hours",
    hue="platform_usage",
    bins=15,
    multiple="layer",
    alpha=0.45
)
plt.title("Розподіл часу використання соцмереж за платформами")
plt.xlabel("Години використання соцмереж на день")
plt.ylabel("Кількість учасників")

legend = ax.get_legend()
if legend is not None:
    legend.set_title("Платформа")
    legend.set_bbox_to_anchor((1.05, 1))
    legend.set_loc("upper left")

save_plot("lab2_2_1_group_hist.png")
plt.show()

print_conclusion(
    "Групова гістограма показує, що характер використання соціальних мереж відрізняється залежно від платформи, тобто різні платформи пов’язані з різною інтенсивністю щоденного використання."
)



print_title("2.2 Порівняння категорій")

bar_data = (
    df.groupby(["platform_usage", "gender"], as_index=False)["stress_level"]
    .mean()
)

top_bar_platforms = df["platform_usage"].value_counts().head(6).index
bar_data = bar_data[bar_data["platform_usage"].isin(top_bar_platforms)]

plt.figure(figsize=(12, 6))
sns.barplot(
    data=bar_data,
    x="platform_usage",
    y="stress_level",
    hue="gender",
    palette="Set2"
)
plt.title("Середній рівень стресу за платформами та статтю")
plt.xlabel("Платформа")
plt.ylabel("Середній рівень стресу")
plt.xticks(rotation=30, ha="right")
plt.legend(title="Стать")
save_plot("lab2_2_2_bar.png")
plt.show()

print_conclusion(
    "Стовпчикова діаграма демонструє, як середній рівень стресу відрізняється між користувачами різних платформ, а колір дозволяє додатково порівняти ці відмінності за статтю."
)



print_title("2.3 Аналіз розкиду та викидів")

platform_order = df["platform_usage"].value_counts().head(6).index
df_box = df[df["platform_usage"].isin(platform_order)].copy()

plt.figure(figsize=(12, 6))
sns.boxplot(
    data=df_box,
    x="platform_usage",
    y="sleep_hours",
    hue="platform_usage",
    palette="pastel",
    legend=False
)
plt.title("Розподіл тривалості сну за платформами")
plt.xlabel("Платформа")
plt.ylabel("Тривалість сну, год")
plt.xticks(rotation=30, ha="right")
save_plot("lab2_2_3_boxplot.png")
plt.show()

print_conclusion(
    "Boxplot дозволяє порівняти медіану, міжквартильний розмах і викиди тривалості сну між користувачами різних платформ, що допомагає виявити відмінності в режимі відпочинку."
)



print_title("2.4 Структура та пропорції")

pie_data = (
    df.groupby("platform_usage")["daily_social_media_hours"]
    .sum()
    .sort_values(ascending=False)
    .head(8)
)

plt.figure(figsize=(8, 8))
plt.pie(
    pie_data,
    labels=pie_data.index,
    autopct="%1.1f%%",
    startangle=90,
    wedgeprops={"width": 0.45}
)
plt.title("Частка платформ у сумарному часі використання соцмереж")
save_plot("lab2_2_4_donut.png")
plt.show()

print_conclusion(
    "Кільцева діаграма показує, яку частку в загальному часі використання соціальних мереж займає кожна платформа. Це дає змогу оцінити, які платформи формують найбільшу частину сумарної активності."
)



print_title("2.5 Взаємозвʼязок двох числових змінних")

scatter_platforms = df["platform_usage"].value_counts().head(6).index
df_scatter = df[df["platform_usage"].isin(scatter_platforms)].copy()

plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df_scatter,
    x="daily_social_media_hours",
    y="sleep_hours",
    hue="platform_usage",
    alpha=0.75,
    palette="tab10"
)
plt.title("Залежність між часом у соцмережах і тривалістю сну")
plt.xlabel("Години використання соцмереж на день")
plt.ylabel("Тривалість сну, год")
plt.legend(title="Платформа", bbox_to_anchor=(1.05, 1), loc="upper left")
save_plot("lab2_2_5_scatter.png")
plt.show()

corr_sleep_social = df[["daily_social_media_hours", "sleep_hours"]].corr().iloc[0, 1]

print_conclusion(
    f"Точкова діаграма показує, що вираженого лінійного зв’язку між часом використання соціальних мереж і тривалістю сну в цьому наборі даних не виявлено. Коефіцієнт кореляції становить приблизно {corr_sleep_social:.2f}, що близько до нуля."
)



print_title("2.6 Динаміка або тренд")

trend_df = (
    df.groupby("age", as_index=False)["academic_performance"]
    .mean()
    .sort_values("age")
)

plt.figure(figsize=(12, 6))
sns.lineplot(
    data=trend_df,
    x="age",
    y="academic_performance",
    marker="o",
    color="teal"
)
plt.title("Зміна середньої академічної успішності залежно від віку")
plt.xlabel("Вік")
plt.ylabel("Середня академічна успішність")
save_plot("lab2_2_6_line.png")
plt.show()

print_conclusion(
    "Лінійний графік показує, як змінюється середній рівень академічної успішності зі зростанням віку. Вісь X має природний логічний порядок, тому цей графік відповідає вимозі аналізу тренду."
)



print_title("2.7 Матриця кореляцій")

corr_columns = [
    "age",
    "daily_social_media_hours",
    "sleep_hours",
    "screen_time_before_sleep",
    "academic_performance",
    "physical_activity",
    "stress_level",
    "anxiety_level",
    "addiction_level"
]

corr_matrix = df[corr_columns].corr()

plt.figure(figsize=(12, 8))
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
plt.title("Теплова карта матриці кореляцій")
save_plot("lab2_2_7_heatmap.png")
plt.show()

corr_pairs = corr_matrix.abs().unstack().sort_values(ascending=False)
corr_pairs = corr_pairs[corr_pairs < 1]
max_corr = corr_pairs.iloc[0]

print_conclusion(
    f"Теплова карта показує, що в наборі даних немає сильних лінійних залежностей між числовими показниками. Найбільша кореляція між різними змінними становить приблизно {max_corr:.2f}, тобто зв’язок є слабким."
)



print_title("2.8 Комбінований графік")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Гістограма
sns.histplot(
    data=df_no_outliers,
    x="daily_social_media_hours",
    bins=15,
    color="skyblue",
    edgecolor="black",
    ax=axes[0, 0]
)
axes[0, 0].set_title("Розподіл часу використання соцмереж")
axes[0, 0].set_xlabel("Години на день")
axes[0, 0].set_ylabel("Кількість учасників")

# 2. Стовпчикова діаграма
avg_stress_platform = (
    df.groupby("platform_usage")["stress_level"]
    .mean()
    .sort_values(ascending=False)
    .head(8)
    .reset_index()
)
avg_stress_platform.columns = ["platform_usage", "stress_level"]

sns.barplot(
    data=avg_stress_platform,
    x="platform_usage",
    y="stress_level",
    hue="platform_usage",
    ax=axes[0, 1],
    palette="viridis",
    legend=False
)
axes[0, 1].set_title("Середній рівень стресу за платформами")
axes[0, 1].set_xlabel("Платформа")
axes[0, 1].set_ylabel("Середній рівень стресу")
axes[0, 1].tick_params(axis="x", rotation=30)

# 3. Лінійний графік
sns.lineplot(
    data=trend_df,
    x="age",
    y="academic_performance",
    marker="o",
    ax=axes[1, 0],
    color="teal"
)
axes[1, 0].set_title("Академічна успішність за віком")
axes[1, 0].set_xlabel("Вік")
axes[1, 0].set_ylabel("Середня успішність")

# 4. Scatter plot
sns.scatterplot(
    data=df_scatter,
    x="daily_social_media_hours",
    y="sleep_hours",
    hue="platform_usage",
    ax=axes[1, 1],
    palette="tab10"
)
axes[1, 1].set_title("Соцмережі та сон")
axes[1, 1].set_xlabel("Години в соцмережах")
axes[1, 1].set_ylabel("Сон, год")
axes[1, 1].legend(title="Платформа", fontsize=8)

plt.suptitle("Комплексна панель візуального аналізу впливу соцмереж", fontsize=14)
save_plot("lab2_2_8_dashboard.png")
plt.show()

print_conclusion(
    "Комбінований графік об’єднує кілька аспектів аналізу: інтенсивність користування соціальними мережами, порівняння платформ, зміну успішності з віком та зв’язок між використанням соцмереж і сном."
)


