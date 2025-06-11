import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import make_pipeline

# Настройки страницы
st.set_page_config(page_title="COVID-19 Dashboard", layout="wide")
st.title("🦠 COVID-19: Статистика по регионам России")


# Функция очистки данных
def extract_value(x):
    try:
        clean_str = ''.join(filter(lambda c: c.isdigit() or c == '-', str(x).split("(")[0]))
        return int(clean_str) if clean_str != '' else 0
    except:
        return 0


def extract_change(x):
    try:
        if "(" in str(x):
            change_str = str(x).split("(")[1].strip("+)").replace(",", "")
            return int(change_str) if change_str != '' else 0
        else:
            return 0
    except:
        return 0


# Функция для форматирования больших чисел
def format_number(n):
    try:
        n = float(n)
        if abs(n) >= 1e12:
            return f"{n / 1e12:.2f} трлн"
        elif abs(n) >= 1e9:
            return f"{n / 1e9:.2f} млрд"
        elif abs(n) >= 1e6:
            return f"{n / 1e6:.2f} млн"
        elif abs(n) >= 1e3:
            return f"{n / 1e3:.2f} тыс"
        elif abs(n) < 0.01 and abs(n) > 0:
            return f"{n:.2e}"
        else:
            return f"{n:,.0f}"
    except:
        return str(n)


# Функция для вычисления adjusted R²
def adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2


# Загрузка данных
@st.cache_data
def load_data():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(current_dir, "covid.csv")
    except NameError:
        csv_path = "covid.csv"

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, encoding="utf-8")
    else:
        st.error(f"Файл данных 'covid.csv' не найден по пути: {csv_path}")
        st.info("Пожалуйста, загрузите файл данных")
        uploaded_file = st.file_uploader("Загрузите CSV файл с данными", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file, encoding="utf-8")
        else:
            df = pd.DataFrame(columns=[
                "Регион России", "Заражений", "Смертей", "Выздоровлений",
                "Заражено на настоящий момент"
            ])
            return df

    df = df.dropna()
    df["Заражений_число"] = df["Заражений"].apply(extract_value)
    df["Смертей_число"] = df["Смертей"].apply(extract_value)
    df["Выздоровлений_число"] = df["Выздоровлений"].apply(extract_value)
    df["Активные_случаи"] = df["Заражено на настоящий момент"].apply(extract_value)
    df["Изменение_заражений"] = df["Заражений"].apply(extract_change)

    df["Летальность"] = df["Смертей_число"] / df["Заражений_число"].replace(0, 1)
    df["Доля_выздоровлений"] = df["Выздоровлений_число"] / df["Заражений_число"].replace(0, 1)
    df["Доля_активных"] = df["Активные_случаи"] / df["Заражений_число"].replace(0, 1)

    df.replace([np.inf, -np.inf], 0, inplace=True)

    # Удаление выбросов с использованием IQR
    Q1 = df["Активные_случаи"].quantile(0.25)
    Q3 = df["Активные_случаи"].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df["Активные_случаи"] >= Q1 - 1.5 * IQR) & (df["Активные_случаи"] <= Q3 + 1.5 * IQR)]

    return df


df = load_data()

# --- Боковая панель ---
st.sidebar.header("Фильтры")
top_n = st.sidebar.slider("Выберите количество топ-регионов", min_value=5, max_value=30, value=10)

# --- Общая статистика ---
st.subheader("📊 Общая статистика по России")
if not df.empty:
    col1, col2, col3, col4 = st.columns(4)
    metrics = {
        "Заражений": df['Заражений_число'].sum(),
        "Смертей": df['Смертей_число'].sum(),
        "Выздоровлений": df['Выздоровлений_число'].sum(),
        "Активные случаи": df['Активные_случаи'].sum()
    }
    for i, (name, value) in enumerate(metrics.items()):
        with [col1, col2, col3, col4][i]:
            st.metric(f"Всего {name.lower()}", format_number(value))

# --- ТОП Регионы ---
if not df.empty and "Заражений_число" in df.columns:
    st.markdown("---")
    st.subheader(f"📌 Топ-{top_n} регионов по числу заражений")
    top_regions = df.sort_values(by="Заражений_число", ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=top_regions, y="Регион России", x="Заражений_число", palette="viridis")
    ax.set_xlabel("Число заражений")
    ax.set_ylabel("Регион")
    ax.set_title(f"Топ-{top_n} регионов по количеству заражений")
    plt.tight_layout()
    st.pyplot(fig)

# --- Диаграмма рассеяния ---
if not df.empty and 'Активные_случаи' in df.columns and 'Заражений_число' in df.columns:
    st.markdown("---")
    st.subheader("📉 Соотношение заражений и активных случаев")
    fig, ax = plt.subplots(figsize=(10, 6))
    hue_col = "Доля_выздоровлений" if "Доля_выздоровлений" in df.columns else None
    size_col = "Летальность" if "Летальность" in df.columns else None
    sns.scatterplot(
        data=df,
        x="Заражений_число",
        y="Активные_случаи",
        size=size_col,
        hue=hue_col,
        sizes=(20, 200) if size_col else (20, 20),
        alpha=0.7,
        palette="viridis"
    )
    ax.set_title("Зависимость активных случаев от общего числа заражений")
    ax.set_xlabel("Общее число заражений")
    ax.set_ylabel("Активные случаи")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig)

# --- Анализ корреляции ---
if not df.empty and len(df) > 10 and 'Активные_случаи' in df.columns:
    st.markdown("---")
    st.subheader("🔍 Анализ корреляции признаков")

    numeric_cols = ['Заражений_число', 'Смертей_число', 'Выздоровлений_число',
                    'Активные_случаи', 'Летальность', 'Доля_выздоровлений']
    numeric_cols = [col for col in numeric_cols if col in df.columns]

    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                    vmin=-1, vmax=1, fmt='.2f')
        ax.set_title("Матрица корреляции признаков")
        plt.tight_layout()
        st.pyplot(fig)

        st.write("Анализ корреляции с целевой переменной (Активные случаи):")
        corr_with_target = corr_matrix['Активные_случаи'].sort_values(ascending=False)
        for feature, corr_value in corr_with_target.items():
            if feature != 'Активные_случаи':
                strength = "сильная" if abs(corr_value) > 0.7 else "умеренная" if abs(corr_value) > 0.3 else "слабая"
                direction = "положительная" if corr_value > 0 else "отрицательная"
                st.write(f"- {feature}: {corr_value:.3f} ({strength} {direction} корреляция)")

# --- Модель ML ---
if not df.empty and len(df) > 10 and 'Активные_случаи' in df.columns:
    st.markdown("---")
    st.subheader("🤖 Прогнозирование активных случаев")

    feature_options = {
        "Только заражения": ["Заражений_число"],
        "Заражения + показатели": ["Заражений_число", "Летальность", "Доля_выздоровлений"],
        "Все числовые признаки": ["Заражений_число", "Смертей_число", "Летальность", "Доля_выздоровлений"]
    }

    available_features = {}
    for name, features in feature_options.items():
        existing_features = [f for f in features if f in df.columns]
        if existing_features:
            available_features[name] = existing_features

    if not available_features:
        st.warning("Нет доступных признаков для построения модели")
    else:
        selected_features = st.selectbox("Выберите признаки для модели", list(available_features.keys()))
        features_to_use = available_features[selected_features]

        # Проверка корреляции и удаление высококоррелированных признаков
        if len(features_to_use) > 1:
            corr_matrix = df[features_to_use].corr()
            to_remove = set()
            for i in range(len(features_to_use)):
                for j in range(i + 1, len(features_to_use)):
                    if abs(corr_matrix.iloc[i, j]) > 0.8:
                        to_remove.add(features_to_use[j])  # Удаляем второй признак из пары
            features_to_use = [f for f in features_to_use if f not in to_remove]
            if len(features_to_use) == 0:
                features_to_use = [available_features[selected_features][0]]  # Оставляем хотя бы один признак
                st.warning("Все признаки были удалены из-за высокой корреляции. Оставлен только первый признак.")

        model_type = st.selectbox("Выберите модель",
                                  ["Линейная регрессия",
                                   "Полиномиальная регрессия (степень 2)",
                                   "Случайный лес",
                                   "SVR"])

        # Подготовка данных
        X = df[features_to_use]
        y = df["Активные_случаи"]

        # Логарифмическое преобразование целевой переменной
        y = np.log1p(y)  # log1p = log(1 + y) для обработки нулей

        data_clean = pd.concat([X, y], axis=1).dropna()
        if len(data_clean) < 10:
            st.warning("Недостаточно данных после очистки для построения модели")
        else:
            X = data_clean[features_to_use]
            y = data_clean["Активные_случаи"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Определение модели
            if model_type == "Линейная регрессия":
                model = make_pipeline(StandardScaler(), LinearRegression())
            elif model_type == "Полиномиальная регрессия (степень 2)":
                model = make_pipeline(
                    StandardScaler(),
                    PolynomialFeatures(degree=2),
                    LinearRegression()
                )
            elif model_type == "Случайный лес":
                model = make_pipeline(StandardScaler(), RandomForestRegressor(random_state=42))
                param_grid = {
                    'randomforestregressor__n_estimators': [50, 100, 200],
                    'randomforestregressor__max_depth': [3, 5, 10]
                }
                model = GridSearchCV(model, param_grid, cv=5, scoring='r2')
            else:  # SVR
                model = make_pipeline(StandardScaler(), SVR(kernel='rbf'))
                param_grid = {
                    'svr__C': [1, 10, 100],
                    'svr__gamma': ['scale', 'auto', 0.1]
                }
                model = GridSearchCV(model, param_grid, cv=5, scoring='r2')

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Обратное преобразование предсказаний
            y_test = np.expm1(y_test)
            y_pred = np.expm1(y_pred)

            # Вычисление метрик
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            adj_r2 = adjusted_r2(r2, len(y_test), len(features_to_use))

            # Проверка R²
            if r2 > 0.999:
                st.warning("R² близок к 1, возможно переобучение модели")
            elif r2 < 0:
                st.warning(
                    "R² отрицательный, модель хуже, чем простое среднее. Попробуйте другой набор признаков или модель.")

            st.subheader("Результаты моделирования")
            metrics_data = {
                "Метрика": ["MSE (Среднекв. ошибка)", "RMSE (Корень из MSE)",
                            "MAE (Ср. абс. ошибка)", "R² (Доля объясн. дисперсии)",
                            "Adjusted R²"],
                "Значение": [format_number(mse), format_number(rmse),
                             format_number(mae), f"{r2:.4f}", f"{adj_r2:.4f}"]
            }
            metrics_df = pd.DataFrame(metrics_data)
            st.table(metrics_df)

            # Вывод лучших параметров для GridSearchCV
            if model_type in ["Случайный лес", "SVR"]:
                st.write("Лучшие параметры модели:", model.best_params_)

            st.subheader("Визуализация предсказаний")
            fig, ax = plt.subplots(figsize=(10, 6))
            x_plot = X_test.iloc[:, 0] if len(X.columns) > 1 else X_test
            ax.scatter(x_plot, y_test, color="blue", label="Фактические значения")
            ax.scatter(x_plot, y_pred, color="red", alpha=0.5, label="Предсказания")
            ax.set_xlabel(X.columns[0] if len(X.columns) > 1 else features_to_use[0])
            ax.set_ylabel("Активные случаи")
            ax.set_title(f"Предсказания модели ({model_type})")
            ax.legend()
            st.pyplot(fig)

            st.subheader("Анализ остатков")
            residuals = y_test - y_pred
            fig, ax = plt.subplots(1, 2, figsize=(16, 6))
            sns.histplot(residuals, kde=True, ax=ax[0])
            ax[0].set_title("Распределение остатков")
            ax[0].set_xlabel("Ошибка (Факт - Прогноз)")
            sns.scatterplot(x=y_pred, y=residuals, ax=ax[1], alpha=0.7)
            ax[1].axhline(y=0, color='r', linestyle='--')
            ax[1].set_title("Остатки vs Предсказания")
            ax[1].set_xlabel("Предсказанные значения")
            ax[1].set_ylabel("Остатки")
            st.pyplot(fig)

# --- Данные raw ---
st.markdown("---")
if st.checkbox("Показать исходные данные") and not df.empty:
    st.subheader("📄 Исходные данные")
    st.dataframe(df)

# --- Подвал ---
st.markdown("---")
st.markdown("© 2025 | Анализ данных по COVID-19 в России | Streamlit App")
