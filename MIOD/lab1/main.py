import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================
FILE_PATH = "Teen_Mental_Health_Dataset.csv"

CONFIG = {
    "columns_to_keep": [
        "age",
        "gender",
        "daily_social_media_hours",
        "platform_usage",
        "sleep_hours",
        "screen_time_before_sleep",
        "academic_performance",
        "physical_activity",
        "social_interaction_level",
        "stress_level",
        "anxiety_level",
        "addiction_level",
        "depression_label"
    ],

    # Ключовий стовпчик для видалення NaN
    "required_column": "stress_level",

    # Булевий стовпчик
    "bool_column": "depression_label",

    # Для завдання 2.2
    "main_numeric_filter_column": "daily_social_media_hours",
    "main_numeric_threshold": 5,
    "second_numeric_column": "stress_level",
    "top10_sort_column": "daily_social_media_hours",

    # Для завдання 2.3
    "exact_match_column": "gender",
    "exact_match_value": "female",

    "substring_column": "platform_usage",
    "substring_value": "Instagram",

    # Для завдання 2.4
    "equal_check_column": "age",
    "equal_check_value": 16,

    "range_column": "sleep_hours",
    "range_min": 6,
    "range_max": 8,

    # Для завдання 2.5
    "significant_column": "addiction_level",
    "significant_threshold": 7,
    "top5_sort_column": "stress_level",
    "top10_metric_column": "anxiety_level",

    # Для завдання 2.6
    "group_column": "gender",
    "group_1": "male",
    "group_2": "female",
    "group_avg_column": "stress_level",

    # Для завдання 2.7
    "complex_filter_hours_threshold": 4,
    "complex_filter_stress_threshold": 7,
    "complex_filter_exclude_platform": "TikTok",

    # Для графіків
    "plot_column": "stress_level"
}


# ============================================================
# ДОПОМІЖНІ ФУНКЦІЇ
# ============================================================
def print_title(title: str):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def print_conclusion(text: str):
    print(f"\nВисновок: {text}")


def check_columns(df: pd.DataFrame, required_columns: list):
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"У датасеті відсутні такі колонки: {missing}\n"
            f"Наявні колонки: {list(df.columns)}"
        )


def normalize_text_column(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def save_current_plot(filename: str):
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Графік збережено у файл: {filename}")


# ============================================================
# 1.1 ВИБІР ТА ЗАВАНТАЖЕННЯ НАБОРУ ДАНИХ
# ============================================================
print_title("1.1 ВИБІР ТА ЗАВАНТАЖЕННЯ НАБОРУ ДАНИХ")

df = pd.read_csv(FILE_PATH)
df.columns = df.columns.str.strip()

print("Перші 5 рядків початкового набору даних:")
print(df.head())

print("\nНазви колонок у датасеті:")
print(df.columns.tolist())

check_columns(df, CONFIG["columns_to_keep"])

print_conclusion("Набір даних успішно завантажено, всі потрібні колонки присутні.")


# ============================================================
# 1.2 ОЧИЩЕННЯ НАБОРУ ДАНИХ
# ============================================================
print_title("1.2 ОЧИЩЕННЯ НАБОРУ ДАНИХ")

df = df[CONFIG["columns_to_keep"]].copy()

# Додаємо штучний ідентифікатор запису
df = df.reset_index().rename(columns={"index": "record_id"})

print("Після видалення зайвих стовпців залишили такі колонки:")
print(df.columns.tolist())

print("\nПерші 5 рядків після очищення:")
print(df.head())

print_conclusion("Залишено лише ті ознаки, які потрібні для подальшого аналізу.")


# ============================================================
# 1.3 ОБРОБКА ПРОПУЩЕНИХ ЗНАЧЕНЬ
# ============================================================
print_title("1.3 ОБРОБКА ПРОПУЩЕНИХ ЗНАЧЕНЬ")

before_dropna = len(df)
df = df.dropna(subset=[CONFIG["required_column"]])
after_dropna = len(df)

print(f"Кількість рядків до видалення NaN у стовпчику '{CONFIG['required_column']}': {before_dropna}")
print(f"Кількість рядків після видалення NaN: {after_dropna}")
print(f"Видалено рядків: {before_dropna - after_dropna}")

print_conclusion(
    f"Після перевірки ключового стовпчика '{CONFIG['required_column']}' "
    f"у датасеті залишилось {after_dropna} записів."
)


# ============================================================
# 1.4 ФОРМАТУВАННЯ ДАНИХ
# ============================================================
print_title("1.4 ФОРМАТУВАННЯ ДАНИХ")

# Нормалізація текстових колонок
text_columns = ["gender", "platform_usage", "social_interaction_level"]
for col in text_columns:
    df[col] = normalize_text_column(df[col])

# Приведення тексту до нижнього регістру для зручності фільтрації
df["gender"] = df["gender"].str.lower()
df["platform_usage"] = df["platform_usage"].str.strip()

# Перетворення depression_label до bool
df[CONFIG["bool_column"]] = (
    df[CONFIG["bool_column"]]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({
        "yes": True,
        "no": False,
        "true": True,
        "false": False,
        "1": True,
        "0": False
    })
)

print("Типи даних після форматування:")
print(df.dtypes)

print_conclusion("Текстові значення уніфіковано, а depression_label успішно приведено до булевого типу.")


# ============================================================
# 2.1 ЗАГАЛЬНА КІЛЬКІСТЬ РЯДКІВ
# ============================================================
print_title("2.1 ЗАГАЛЬНА КІЛЬКІСТЬ РЯДКІВ")

print(f"Загальна кількість записів після очищення: {len(df)}")

print_conclusion("Набір даних готовий до аналізу.")


# ============================================================
# 2.2 РОБОТА З ЧИСЛОВИМИ ПОКАЗНИКАМИ
# ============================================================
print_title("2.2 РОБОТА З ЧИСЛОВИМИ ПОКАЗНИКАМИ")

filtered_numeric = df[
    df[CONFIG["main_numeric_filter_column"]] > CONFIG["main_numeric_threshold"]
].copy()

print(
    f"1) Відфільтровано записи, де "
    f"{CONFIG['main_numeric_filter_column']} > {CONFIG['main_numeric_threshold']}"
)
print(f"2) Кількість таких записів: {len(filtered_numeric)}")

avg_value = filtered_numeric[CONFIG["second_numeric_column"]].mean()
print(
    f"3) Середнє значення стовпчика '{CONFIG['second_numeric_column']}' "
    f"для відфільтрованої вибірки: {avg_value:.2f}"
)

top_10_rows = df.nlargest(10, CONFIG["top10_sort_column"])[
    ["record_id", "daily_social_media_hours", "age", "gender", "platform_usage"]
]
print(
    f"4) Ідентифікатори 10 записів з найбільшим значенням "
    f"'{CONFIG['top10_sort_column']}':"
)
print(top_10_rows)

print_conclusion(
    "Було досліджено вибірку підлітків із високою тривалістю використання соціальних мереж."
)


# ============================================================
# 2.3 ДОСЛІДЖЕННЯ КАТЕГОРІЙ ТА ТЕКСТУ
# ============================================================
print_title("2.3 ДОСЛІДЖЕННЯ КАТЕГОРІЙ ТА ТЕКСТУ")

cond1 = df[CONFIG["exact_match_column"]] == CONFIG["exact_match_value"]
cond2 = df[CONFIG["substring_column"]].str.contains(
    CONFIG["substring_value"], case=False, na=False
)

exact_match_count = cond1.sum()
substring_filtered = df[cond2].copy()
both_conditions_count = (cond1 & cond2).sum()

only_first_count = (cond1 & ~cond2).sum()
only_first_share = only_first_count / exact_match_count if exact_match_count > 0 else 0
neither_count = (~cond1 & ~cond2).sum()

print(
    f"1) Кількість записів, де "
    f"{CONFIG['exact_match_column']} == '{CONFIG['exact_match_value']}': {exact_match_count}"
)

print(
    f"2) Кількість записів, де у стовпчику '{CONFIG['substring_column']}' "
    f"міститься підрядок '{CONFIG['substring_value']}': {len(substring_filtered)}"
)

print(f"3) Кількість записів, які одночасно задовольняють обидві умови: {both_conditions_count}")
print(f"4) Частка записів, які задовольняють умову 1, але не умову 2: {only_first_share:.4f}")
print(f"5) Кількість записів, які не задовольняють жодну з двох умов: {neither_count}")

print_conclusion(
    "Було виконано аналіз категоріальних ознак за точним та частковим збігом."
)


# ============================================================
# 2.4 ДОСЛІДЖЕННЯ ЧИСЛОВИХ ДІАПАЗОНІВ ТА ЗРІЗІВ ДАНИХ
# ============================================================
print_title("2.4 ДОСЛІДЖЕННЯ ЧИСЛОВИХ ДІАПАЗОНІВ ТА ЗРІЗІВ ДАНИХ")

count_equal = (df[CONFIG["equal_check_column"]] == CONFIG["equal_check_value"]).sum()
count_range = df[
    (df[CONFIG["range_column"]] >= CONFIG["range_min"]) &
    (df[CONFIG["range_column"]] <= CONFIG["range_max"])
].shape[0]

comparison_result = count_equal > count_range

print(
    f"1) Кількість записів, де "
    f"{CONFIG['equal_check_column']} == {CONFIG['equal_check_value']}: {count_equal}"
)

print(
    f"2) Кількість записів, де "
    f"{CONFIG['range_min']} <= {CONFIG['range_column']} <= {CONFIG['range_max']}: {count_range}"
)

print(f"3) Чи кількість записів з першої умови більша, ніж з другої? {comparison_result}")

print_conclusion("Було порівняно точковий зріз даних і діапазон числових значень.")


# ============================================================
# 2.5 КОМБІНОВАНІ ФІЛЬТРИ ТА ОЦІНКИ
# ============================================================
print_title("2.5 КОМБІНОВАНІ ФІЛЬТРИ ТА ОЦІНКИ")

significant_df = df[df[CONFIG["significant_column"]] > CONFIG["significant_threshold"]].copy()

print(
    f"1) Кількість 'значущих' записів, де "
    f"{CONFIG['significant_column']} > {CONFIG['significant_threshold']}: {len(significant_df)}"
)

top_5_significant = significant_df.nlargest(5, CONFIG["top5_sort_column"])
print(f"\n2) Топ-5 записів з найвищим значенням '{CONFIG['top5_sort_column']}':")
print(top_5_significant[
    ["record_id", "age", "gender", "platform_usage", "addiction_level", "stress_level"]
])

top_10_other = df.nlargest(10, "daily_social_media_hours")
top10_metric_mean = top_10_other[CONFIG["top10_metric_column"]].mean()

print(
    f"\n3) Середнє значення '{CONFIG['top10_metric_column']}' "
    f"для топ-10 записів, відсортованих за 'daily_social_media_hours': {top10_metric_mean:.2f}"
)

print_conclusion("Було оцінено найбільш значущі записи та розраховано агреговані показники.")


# ============================================================
# 2.6 ПОРІВНЯННЯ ГРУП
# ============================================================
print_title("2.6 ПОРІВНЯННЯ ГРУП")

group_1_df = df[df[CONFIG["group_column"]] == CONFIG["group_1"]].copy()
group_2_df = df[df[CONFIG["group_column"]] == CONFIG["group_2"]].copy()

comparison_df = pd.DataFrame({
    "category_name": [CONFIG["group_1"].capitalize(), CONFIG["group_2"].capitalize()],
    "total_records": [len(group_1_df), len(group_2_df)],
    "average_value": [
        group_1_df[CONFIG["group_avg_column"]].mean(),
        group_2_df[CONFIG["group_avg_column"]].mean()
    ]
})

print("Порівняльний DataFrame:")
print(comparison_df)

print_conclusion("Проведено порівняння двох груп за кількістю записів та середнім рівнем стресу.")


# ============================================================
# 2.7 КОМПЛЕКСНІ ЗАВДАННЯ
# ============================================================
print_title("2.7 КОМПЛЕКСНІ ЗАВДАННЯ")

# AND, OR, NOT одночасно
complex_filter = df[
    (
        (df["daily_social_media_hours"] > CONFIG["complex_filter_hours_threshold"]) &
        (df["stress_level"] > CONFIG["complex_filter_stress_threshold"])
    )
    |
    (
        ~(df["platform_usage"].str.lower() == CONFIG["complex_filter_exclude_platform"].lower()) &
        (df["anxiety_level"] >= 8)
    )
].copy()

print("1) Результат складного фільтра з використанням AND, OR, NOT:")
print(f"Кількість записів: {len(complex_filter)}")
print(complex_filter.head())

print_conclusion(
    "Складний фільтр дозволив виділити підлітків із підвищеними ризиковими показниками."
)


# ============================================================
# 3. ПОБУДОВА ГРАФІКІВ
# ============================================================
print_title("3. ПОБУДОВА ГРАФІКІВ")

# ------------------------------------------------------------
# Графік 1: гістограма stress_level
# ------------------------------------------------------------
plt.figure(figsize=(10, 6))
df["stress_level"].plot(kind="hist", bins=10)
plt.title("Гістограма розподілу рівня стресу")
plt.xlabel("Рівень стресу")
plt.ylabel("Кількість записів")
plt.grid(True)
save_current_plot("diagram_1_stress_hist.png")
plt.show()

# ------------------------------------------------------------
# Графік 2: середній рівень стресу за platform_usage
# ------------------------------------------------------------
avg_stress_by_platform = df.groupby("platform_usage")["stress_level"].mean().sort_values()

plt.figure(figsize=(10, 6))
avg_stress_by_platform.plot(kind="bar")
plt.title("Середній рівень стресу залежно від платформи")
plt.xlabel("Платформа")
plt.ylabel("Середній рівень стресу")
plt.grid(True)
save_current_plot("diagram_2_avg_stress_by_platform.png")
plt.show()

# ------------------------------------------------------------
# Графік 3: залежність anxiety_level від daily_social_media_hours
# ------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.scatter(df["daily_social_media_hours"], df["anxiety_level"])
plt.title("Залежність рівня тривожності від часу в соцмережах")
plt.xlabel("Години в соцмережах за день")
plt.ylabel("Рівень тривожності")
plt.grid(True)
save_current_plot("diagram_3_anxiety_vs_social_media.png")
plt.show()

print_conclusion(
    "Побудовано три діаграми: розподіл рівня стресу, порівняння стресу за платформами "
    "та залежність тривожності від часу в соціальних мережах."
)


# ============================================================
# 4. ЗАВЕРШЕННЯ
# ============================================================
print_title("РОБОТУ ЗАВЕРШЕНО")
print("Усі пункти лабораторної роботи виконані.")