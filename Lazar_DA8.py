import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans


df = pd.read_csv('streamlit_dataset.csv')

st.set_page_config(
    page_title="Економічний дашборд",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://docs.streamlit.io/',
        'Report a bug': 'https://github.com/streamlit/streamlit/issues',
        'About': 'Інтерактивна панель для економічного аналізу підприємств'
    }
)

# Бічна панель / SIDEBAR
st.sidebar.title("Панель фільтрації")

# Фільтри 
selected_year = st.sidebar.selectbox("Рік", sorted(df["Year"].unique()))
selected_region = st.sidebar.multiselect("Регіон", df["Region"].unique(), default=df["Region"].unique())
selected_industry = st.sidebar.multiselect("Галузь", df["Industry"].unique(), default=df["Industry"].unique())
selected_scenario = st.sidebar.radio("Сценарій", df["Scenario"].unique())

selected_max_adbudget = st.sidebar.slider(
    "Максимальний рекламний бюджет",
    min_value=int(df["AdBudget"].min()),
    max_value=int(df["AdBudget"].max()),
    value=int(df["AdBudget"].max()),
    step=1000
)

# ---------------------------------
# Фільтрація
# ---------------------------------
df_filtered = df[
    (df["Year"] == selected_year) &
    (df["Region"].isin(selected_region)) &
    (df["Industry"].isin(selected_industry)) &
    (df["Scenario"] == selected_scenario) &
    (df["AdBudget"] <= selected_max_adbudget)
]

# Чекбокси для відображення
show_map = st.sidebar.checkbox("Показати карту компаній")

# Перемикач графіків
chart_option = st.sidebar.radio(
    "📈 Оберіть графік для перегляду:",
    [
        "Доходи на клієнта vs Витрати",
        "Boxplot прибутку по галузях",
        "Scatter: Прибуток vs Інвестиції",
        "Гістограма конверсії по галузях",
        "Теплова карта кореляцій",
        "Кластеризація компаній (KMeans)"
    ]
)

st.sidebar.markdown("---")

# Блок регресії 
st.sidebar.markdown("Побудова регресії")
numeric_columns = df_filtered.select_dtypes(include=np.number).columns.tolist()

reg_x = st.sidebar.selectbox("Оберіть змінну X", numeric_columns, index=0)
reg_y = st.sidebar.selectbox("Оберіть змінну Y", numeric_columns, index=1)
show_regression = st.sidebar.checkbox("Показати регресійну модель")


# Інформаційний блок
st.sidebar.markdown("---")
st.sidebar.markdown(" **Інструкція**: \nФільтруйте дані за параметрами і переглядайте графіки та таблиці на панелі праворуч.")
st.sidebar.markdown(" **Автор**: Lazar_Iryna")
st.sidebar.markdown("---")




# ---------------------------------
# Основна панель
# ---------------------------------

st.title("📊 Економічний дашборд компаній")
st.subheader(f"🔍 Відфільтровано {df_filtered.shape[0]} компаній")

# Кнопка завантаження CSV
csv = df_filtered.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Завантажити CSV",
    data=csv,
    file_name="filtered_companies.csv",
    mime="text/csv"
)


# 📈Інтерактивна таблиця результатів
st.subheader("Оберіть, які стовпці таблиці відображати")

all_columns = df_filtered.columns.tolist()
default_columns = ["Company", "Region", "Industry", "Profit", "ROI"]

selected_columns = st.multiselect(
    "Оберіть стовпці для перегляду",
    options=all_columns,
    default=[col for col in default_columns if col in all_columns]
)

if selected_columns:
    st.dataframe(df_filtered[selected_columns])
else:
    st.info("Оберіть хоча б один стовпець, щоб побачити таблицю.")


# Карта компаній
if show_map:
    st.subheader("🗺 Географія компаній")

    map_data = df_filtered[["Latitude", "Longitude"]].dropna().rename(
        columns={"Latitude": "latitude", "Longitude": "longitude"}
    )

    if not map_data.empty:
        st.map(map_data)
    else:
        st.warning("Немає доступних координат для побудови карти.")

# Відображення обраного графіка     
if chart_option == "Доходи на клієнта vs Витрати":
    st.subheader("📊 Доходи на клієнта vs Витрати")
    chart = alt.Chart(df_filtered).mark_circle(size=60).encode(
        x='Expenses:Q',
        y='RevenuePerCustomer:Q',
        color='Industry:N',
        tooltip=['Company', 'Expenses', 'RevenuePerCustomer', 'Industry']
    ).interactive().properties(title="Дохід на клієнта vs Витрати")
    st.altair_chart(chart, use_container_width=True)

elif chart_option == "Boxplot прибутку по галузях":
    st.subheader("📊 Boxplot прибутку по галузях")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df_filtered, x="Industry", y="Profit", ax=ax)
    ax.set_title("Розподіл прибутку по галузях")
    st.pyplot(fig)

elif chart_option == "Scatter: Прибуток vs Інвестиції":
    st.subheader("📊 Scatter: Прибуток vs Інвестиції")
    chart = alt.Chart(df_filtered).mark_circle(size=60).encode(
        x='Investment:Q',
        y='Profit:Q',
        color='Industry:N',
        tooltip=['Company', 'Investment', 'Profit']
    ).interactive().properties(title="Прибуток vs Інвестиції")
    st.altair_chart(chart, use_container_width=True)

elif chart_option == "Гістограма конверсії по галузях":
    st.subheader("📊 Гістограма конверсії по галузях")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df_filtered, x="Industry", y="ConversionRate", estimator="mean", ax=ax)
    ax.set_title("Середній Conversion Rate по галузях")
    st.pyplot(fig)

elif chart_option == "Теплова карта кореляцій":
    st.subheader("📊 Теплова карта кореляцій")
    numeric_cols = df_filtered.select_dtypes(include=[np.number])
    corr = numeric_cols.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title("Кореляційна матриця числових показників")
    st.pyplot(fig)

elif chart_option == "Кластеризація компаній (KMeans)":
    st.subheader("📊 Кластеризація компаній на основі ROI та Investment")
    cluster_data = df_filtered[["ROI", "Investment"]].dropna()
    if cluster_data.shape[0] >= 3:
        kmeans = KMeans(n_clusters=3, random_state=0)
        cluster_data["Cluster"] = kmeans.fit_predict(cluster_data)

        chart = alt.Chart(cluster_data).mark_circle(size=60).encode(
            x='Investment',
            y='ROI',
            color='Cluster:N',
            tooltip=['ROI', 'Investment', 'Cluster']
        ).interactive().properties(title="Кластеризація за ROI та Investment")
        st.altair_chart(chart, use_container_width=True)
    else:
        st.warning("Недостатньо даних для кластеризації (потрібно ≥ 3 рядки).")


# --- Побудова регресійної моделі ---
if show_regression:
    st.subheader(f"📈 Лінійна регресія: {reg_y} ~ {reg_x}")

    df_reg = df_filtered[[reg_x, reg_y]].dropna()

    if df_reg.shape[0] >= 2:
        model = LinearRegression()
        model.fit(df_reg[[reg_x]], df_reg[reg_y])
        y_pred = model.predict(df_reg[[reg_x]])

        coef = model.coef_[0]
        intercept = model.intercept_
        r2 = model.score(df_reg[[reg_x]], df_reg[reg_y])

        st.markdown(f"**Коефіцієнт нахилу (β):** {coef:.4f}")
        st.markdown(f"**Зсув (intercept):** {intercept:.4f}")
        st.markdown(f"**R²:** {r2:.4f}")

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=df_reg, x=reg_x, y=reg_y, ax=ax)
        sns.lineplot(x=df_reg[reg_x], y=y_pred, color='red', ax=ax)
        ax.set_title(f"Регресія {reg_y} ~ {reg_x}")
        st.pyplot(fig)
    else:
        st.warning("Недостатньо даних для побудови регресії.")
        

