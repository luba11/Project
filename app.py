import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Настройки страницы
st.set_page_config(page_title="COVID-19 Dashboard", layout="wide")
st.title("🦠 COVID-19: Статистика по регионам России")

# Функция очистки данных
def extract_value(x):
    return int(x.split("(")[0].replace(",", "").strip())

def extract_change(x):
    if "(" in x:
        return int(x.split("(")[1].strip("+)"))
    else:
        return 0

# Загрузка данных
@st.cache_data
def load_data():
    df = pd.read_csv("/content/covid.csv", encoding="utf-8")
    df = df.dropna()

    # Очистка данных
    df["Заражений_число"] = df["Заражений"].apply(extract_value)
    df["Смертей_число"] = df["Смертей"].apply(extract_value)
    df["Выздоровлений_число"] = df["Выздоровлений"].apply(extract_value)
    df["Активные_случаи"] = df["Заражено на настоящий момент"].apply(extract_value)
    df["Изменение_заражений"] = df["Заражений"].apply(extract_change)

    return df[["Регион России", "Заражений_число", "Смертей_число", "Выздоровлений_число", "Активные_случаи"]]

# Загрузка данных
df = load_data()

# --- Боковая панель ---
st.sidebar.header("Фильтры")
top_n = st.sidebar.slider("Выберите количество топ-регионов", min_value=5, max_value=30, value=10)

# --- Основная часть ---

# Общая статистика
st.subheader("📊 Общая статистика по России")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Всего заражений", f"{df['Заражений_число'].sum():,}")
col2.metric("Всего смертей", f"{df['Смертей_число'].sum():,}")
col3.metric("Всего выздоровлений", f"{df['Выздоровлений_число'].sum():,}")
col4.metric("Активные случаи", f"{df['Активные_случаи'].sum():,}")

# --- ТОП Регионы ---
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
st.markdown("---")
st.subheader("📉 Соотношение заражений и активных случаев")
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=df, x="Заражений_число", y="Активные_случаи", alpha=0.7)
ax.set_title("Зависимость активных случаев от общего числа заражений")
ax.set_xlabel("Общее число заражений")
ax.set_ylabel("Активные случаи")
st.pyplot(fig)

# --- Модель ML ---
st.markdown("---")
st.subheader("🤖 Линейная регрессия: предсказание активных случаев")

X = df[["Заражений_число"]]
y = df["Активные_случаи"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

st.write(f"**MSE (ошибка модели):** {mse:.2f}")

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(X_test, y_test, color="blue", label="Фактические значения")
ax.plot(X_test, y_pred, color="red", linewidth=2, label="Модель предсказания")
ax.set_xlabel("Число заражений")
ax.set_ylabel("Активные случаи")
ax.set_title("Линейная регрессия")
ax.legend()
st.pyplot(fig)

# --- Данные raw ---
if st.checkbox("Показать исходные данные"):
    st.subheader("📄 Исходные данные")
    st.dataframe(df)

# --- Подвал ---
st.markdown("---")
st.markdown("© 2025 | Анализ данных по COVID-19 в России | Streamlit App")