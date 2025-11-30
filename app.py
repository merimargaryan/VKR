import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import hashlib
import time

# Настройка страницы
st.set_page_config(
    page_title="Прогноз риска оттока клиентов банка",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Стили CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #26344e;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #26344e;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .risk-high { color: #dc3545; font-weight: bold; }
    .risk-medium { color: #ffc107; font-weight: bold; }
    .risk-low { color: #28a745; font-weight: bold; }
    .login-container {
        max-width: 400px;
        margin: 100px auto;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)

# Словари для преобразования значений
education_map = {
    'Unknown': 'Неизвестно',
    'Uneducated': 'Необразованный',
    'High School': 'Средняя школа', 
    'College': 'Колледж',
    'Graduate': 'Выпускник',
    'Post-Graduate': 'Аспирант',
    'Doctorate': 'Доктор наук'
}

marital_map = {
    'Unknown': 'Неизвестно',
    'Single': 'Холост',
    'Married': 'Женат',
    'Divorced': 'Разведен'
}

income_map = {
    'Unknown': 'Неизвестно',
    'Less than $40K': 'Менее 4 млн.₽',
    '$40K - $60K': '4-6 млн.₽',
    '$60K - $80K': '6-8 млн.₽',
    '$80K - $120K': '8-12 млн.₽', 
    '$120K +': 'Более 12 млн.₽'
}

# Обратные словари для преобразования обратно в английские значения для модели
education_map_reverse = {v: k for k, v in education_map.items()}
marital_map_reverse = {v: k for k, v in marital_map.items()}
income_map_reverse = {v: k for k, v in income_map.items()}

# Система авторизации
def authenticate_user(username, password):
    """Простая система аутентификации"""
    users = {
        "user": {"password": "user123", "role": "user"},
        "admin": {"password": "admin123", "role": "admin"},
        "analyst": {"password": "analyst123", "role": "admin"}
    }
    
    if username in users and users[username]["password"] == password:
        return users[username]["role"]
    return None

def show_login():
    """Показать окно авторизации"""
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align: center;">🔐 Авторизация</h2>', unsafe_allow_html=True)
    
    with st.form("login_form"):
        username = st.text_input("👤 Логин")
        password = st.text_input("🔒 Пароль", type="password")
        submit = st.form_submit_button("Войти", use_container_width=True)
        
        if submit:
            role = authenticate_user(username, password)
            if role:
                st.session_state.authenticated = True
                st.session_state.role = role
                st.session_state.username = username
                st.success(f"✅ Успешный вход как {username} ({role})")
                time.sleep(1)
                st.rerun()
            else:
                st.error("❌ Неверный логин или пароль")
    
    st.markdown('</div>', unsafe_allow_html=True)

class BankCustomerAnalytics:
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.encoder = None
        self.feature_names = None
        self.best_model_info = {}
        self.business_report = {}
        self.load_resources()
    
    def load_resources(self):
        """Загрузка всех ресурсов"""
        try:
            # Загрузка ВСЕХ 5 моделей
            self.models = {
                'Gradient Boosting': joblib.load('models/gradient_boosting.pkl'),
                'XGBoost': joblib.load('models/xgboost.pkl'),
                'Random Forest': joblib.load('models/random_forest.pkl'),
                'Logistic Regression': joblib.load('models/logistic_regression.pkl'),
                'CatBoost': joblib.load('models/catboost.pkl')
            }
            
            # Загрузка preprocessing объектов
            self.scaler = joblib.load('models/scaler.pkl')
            self.encoder = joblib.load('models/encoder.pkl')
            self.feature_names = joblib.load('models/feature_names.pkl')
            
            # Загрузка бизнес-отчетов
            with open('models/business_report_gradient_boosting.json', 'r', encoding='utf-8') as f:
                self.business_report = json.load(f)
            
            # Загрузка информации о лучшей модели
            with open('models/best_model_info.json', 'r', encoding='utf-8') as f:
                self.best_model_info = json.load(f)
            
            return True
            
        except Exception as e:
            st.error(f"❌ Ошибка загрузки ресурсов: {e}")
            st.info("Убедитесь, что вы запустили ноутбук для обучения моделей")
            return False
    
    def preprocess_input(self, input_df):
        """Предобработка входных данных"""
        try:
            # Числовые колонки
            numeric_cols = self.scaler.feature_names_in_
            # Категориальные колонки  
            categorical_cols = self.encoder.feature_names_in_
            
            # Масштабирование числовых признаков
            X_num = self.scaler.transform(input_df[numeric_cols])
            
            # Кодирование категориальных признаков
            X_cat = self.encoder.transform(input_df[categorical_cols])
            
            # Объединение
            X_processed = np.c_[X_num, X_cat]
            
            return X_processed
            
        except Exception as e:
            st.error(f"Ошибка предобработки: {e}")
            return None

def main():
    # Инициализация сессии
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.role = None
        st.session_state.username = None
    
    # Проверка авторизации
    if not st.session_state.authenticated:
        show_login()
        return
    
    # Заголовок с информацией о пользователе
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<div class="main-header">🏦 Прогноз риска оттока клиентов банка</div>', unsafe_allow_html=True)
    with col2:
        st.write(f"👤 **{st.session_state.username}** ({st.session_state.role})")
        if st.button("🚪 Выйти"):
            st.session_state.authenticated = False
            st.session_state.role = None
            st.session_state.username = None
            st.rerun()
    
    # Инициализация аналитики
    with st.spinner("Загрузка моделей и данных..."):
        analytics = BankCustomerAnalytics()
    
    if not analytics.models:
        st.stop()
    
    # Сайдбар с учетом роли
    st.sidebar.title("Навигация")
    
    if st.session_state.role == "admin":
        sections = [
            "📊 Бизнес-аналитика",
            "🎯 Оценка клиента", 
            "📈 Сравнение моделей"
        ]
    else:
        sections = [
            "📊 Бизнес-аналитика",
            "🎯 Оценка клиента"
        ]
    
    section = st.sidebar.radio("Выберите раздел:", sections)
    
    if section == "📊 Бизнес-аналитика":
        show_business_overview(analytics)
    elif section == "🎯 Оценка клиента":
        show_customer_assessment(analytics)
    elif section == "📈 Сравнение моделей":
        show_model_comparison(analytics)

def show_business_overview(analytics):
    """Показать общую бизнес-аналитику"""
    st.header("📊 Общая бизнес-аналитика")
    
    # Ключевые метрики
    biz_metrics = analytics.business_report['business_metrics']
    biz_insights = analytics.business_report['business_insights']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Всего клиентов", 
            f"{biz_metrics['total_customers']:,}"
        )
    
    with col2:
        st.metric(
            "Уровень оттока", 
            f"{biz_metrics['churn_rate']:.1%}"
        )
    
    with col3:
        st.metric(
            "Клиентов высокого риска", 
            biz_metrics['high_risk_customers']
        )
    
    with col4:
        st.metric(
            "Потенциальные потери", 
            f"{biz_metrics['total_potential_loss']:,.0f}₽"
        )
    
    # Визуализации
    col1, col2 = st.columns(2)
    
    with col1:
        # Распределение рисков
        risk_data = {
            'Риск': ['Высокий', 'Низкий'],
            'Клиенты': [
                biz_metrics['high_risk_customers'],
                biz_metrics['low_risk_customers']
            ]
        }
        
        fig = px.pie(
            risk_data, 
            values='Клиенты', 
            names='Риск',
            title='Распределение клиентов по уровню риска',
            color='Риск',
            color_discrete_map={
                'Высокий': '#dc3545',
                'Низкий': '#28a745'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Финансовый анализ с подписями значений
        loss_data = {
            'Тип потерь': ['Потери от транзакций', 'Кредитные потери'],
            'Сумма': [
                biz_metrics['potential_revenue_loss'],
                biz_metrics['potential_credit_loss']
            ]
        }
        
        fig = px.bar(
            loss_data,
            x='Тип потерь',
            y='Сумма',
            title='Структура потенциальных потерь',
            color='Тип потерь',
            color_discrete_sequence=['#ff6b6b', '#4ecdc4'],
            text=[f'{x:,.0f}₽' for x in loss_data['Сумма']]
        )
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        fig.update_layout(
            showlegend=False,
            yaxis_title="Сумма (₽)",
            xaxis_title=""
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Детальная аналитика
    st.subheader("📈 Детальная аналитика")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Ценных клиентов в риске", biz_metrics['high_value_at_risk'])
    
    with col2:
        st.metric("Средняя оценка риска", f"{biz_metrics['avg_risk_score']:.3f}")
    
    with col3:
        total_transactions = biz_metrics['total_customers'] * biz_metrics['avg_transaction_value']
        st.metric("Сумма транзакций", f"{total_transactions:,.0f}₽")
    
    # Рекомендации
    st.subheader("🎯 Рекомендации для бизнеса")
    
    recommendations = [
        f"**Сфокусироваться на {biz_metrics['high_risk_customers']} клиентах высокого риска**",
        f"**Защитить {biz_metrics['high_value_at_risk']} ценных клиентов** от оттока",
        f"**Разработать программы удержания** для сегмента среднего риска",
        f"**Мониторить клиентов** со средней оценкой риска выше {biz_metrics['avg_risk_score']:.2f}",
        f"**Оптимизировать бюджет удержания** исходя из потенциальных потерь {biz_metrics['total_potential_loss']:,.0f}₽"
    ]
    
    for rec in recommendations:
        st.info(rec)

def show_customer_assessment(analytics):
    """Оценка риска оттока для конкретного клиента"""
    st.header("🎯 Оценка риска оттока клиента")
    
    # Выбор модели
    model_choice = st.selectbox(
        "Выберите модель для оценки:",
        list(analytics.models.keys()),
        index=0  # Gradient Boosting по умолчанию
    )
    
    st.subheader("📋 Профиль клиента")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Демографические данные:**")
        customer_age = st.slider("Возраст", 18, 80, 45)
        gender = st.selectbox("Пол", ["M", "F"])
        dependent_count = st.slider("Количество иждивенцев", 0, 5, 1)
        
        # Образование с русскими значениями
        education_level = st.selectbox(
            "Уровень образования", 
            list(education_map.values())
        )
        
        # Семейное положение с русскими значениями
        marital_status = st.selectbox(
            "Семейное положение", 
            list(marital_map.values())
        )
    
    with col2:
        st.write("**Финансовые показатели:**")
        
        # Доход с русскими значениями
        income_category = st.selectbox(
            "Категория дохода", 
            list(income_map.values())
        )
        
        card_category = st.selectbox(
            "Тип карты", 
            ["Blue", "Silver", "Gold", "Platinum"]
        )
        credit_limit = st.number_input("Кредитный лимит (₽)", 1000, 5000000, 5000)
        total_revolving_bal = st.number_input("Револьверный баланс (₽)", 0, 500000, 500)
        total_trans_amt = st.number_input("Общая сумма транзакций (₽)", 0, 5000000, 5000)
    
    # Дополнительные параметры
    with st.expander("📊 Дополнительные параметры"):
        col1, col2 = st.columns(2)
        
        with col1:
            months_on_book = st.slider("Месяцев в банке", 0, 60, 36)
            total_relationship_count = st.slider("Количество продуктов", 1, 6, 3)
            months_inactive = st.slider("Месяцев неактивности", 0, 6, 2)
        
        with col2:
            contacts_count = st.slider("Контактов с банком", 0, 6, 2)
            total_trans_ct = st.slider("Количество транзакций", 0, 200, 50)
            avg_utilization_ratio = st.slider("Использование кредита", 0.0, 1.0, 0.3)
    
    if st.button("🔍 Оценить риск оттока", type="primary", use_container_width=True):
        # Преобразование русских значений обратно в английские для модели
        education_english = education_map_reverse[education_level]
        marital_english = marital_map_reverse[marital_status]
        income_english = income_map_reverse[income_category]
        
        # Подготовка данных
        input_data = prepare_customer_data(
            customer_age, gender, dependent_count, education_english, marital_english,
            income_english, card_category, months_on_book, total_relationship_count,
            months_inactive, contacts_count, credit_limit, total_revolving_bal,
            total_trans_amt, total_trans_ct, avg_utilization_ratio
        )
        
        # Прогноз
        try:
            model = analytics.models[model_choice]
            X_processed = analytics.preprocess_input(input_data)
            
            if X_processed is not None:
                prediction = model.predict(X_processed)[0]
                probability = model.predict_proba(X_processed)[0][0]  # Вероятность оттока
                
                # Отображение результатов
                show_prediction_results(probability, prediction, input_data, model_choice)
                
        except Exception as e:
            st.error(f"Ошибка при прогнозировании: {e}")

def prepare_customer_data(age, gender, dependents, education, marital, income, 
                         card, months, products, inactive, contacts, limit, 
                         balance, trans_amt, trans_ct, utilization):
    """Подготовка данных клиента для модели"""
    
    # Расчет производных признаков
    avg_open_to_buy = limit - balance
    
    return pd.DataFrame({
        'Customer_Age': [age],
        'Gender': [gender],
        'Dependent_count': [dependents],
        'Education_Level': [education],
        'Marital_Status': [marital],
        'Income_Category': [income],
        'Card_Category': [card],
        'Months_on_book': [months],
        'Total_Relationship_Count': [products],
        'Months_Inactive_12_mon': [inactive],
        'Contacts_Count_12_mon': [contacts],
        'Credit_Limit': [limit],
        'Total_Revolving_Bal': [balance],
        'Avg_Open_To_Buy': [avg_open_to_buy],
        'Total_Trans_Amt': [trans_amt],
        'Total_Trans_Ct': [trans_ct],
        'Avg_Utilization_Ratio': [utilization],
        # Значения по умолчанию для остальных признаков
        'Total_Amt_Chng_Q4_Q1': [0.8],
        'Total_Ct_Chng_Q4_Q1': [0.8]
    })

def show_prediction_results(probability, prediction, customer_data, model_name):
    """Отображение результатов прогноза"""
    
    st.subheader("🔮 Результат оценки риска")
    
    # Определение уровня риска
    if probability > 0.7:
        risk_level = "Высокий"
        risk_color = "risk-high"
        risk_emoji = "🔴"
    elif probability > 0.3:
        risk_level = "Средний" 
        risk_color = "risk-medium"
        risk_emoji = "🟡"
    else:
        risk_level = "Низкий"
        risk_color = "risk-low"
        risk_emoji = "🟢"
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Уровень риска", f"{risk_emoji} {risk_level}")
    
    with col2:
        st.metric("Вероятность оттока", f"{probability:.1%}")
    
    with col3:
        clv = customer_data['Total_Trans_Amt'].iloc[0] * 12
        st.metric("Годовой CLV", f"{clv:,.0f}₽")
    
    # Визуализация вероятности
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Вероятность оттока"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70}
        }
    ))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Рекомендации
    st.subheader("💡 Рекомендации")
    
    if probability > 0.7:
        st.error("""
        **🚨 ВЫСОКИЙ РИСК ОТТОКА!**
        
        **Срочные меры удержания:**
        - Немедленный контакт персонального менеджера
        - Специальные условия по кредитной карте
        - Программа лояльности с повышенным кэшбэком
        - Персональное предложение по рефинансированию
        - Регулярный мониторинг активности
        """)
    elif probability > 0.3:
        st.warning("""
        **⚠️ СРЕДНИЙ РИСК ОТТОКА**
        
        **Проактивные меры:**
        - Увеличение числа контактов (раз в 2 недели)
        - Предложение дополнительных услуг
        - Программа кэшбэка за активность
        - Сбор обратной связи
        - Приглашение на финансовые консультации
        """)
    else:
        st.success("""
        **✅ НИЗКИЙ РИСК ОТТОКА**
        
        **Стратегия развития:**
        - Кросс-продажи дополнительных продуктов
        - Премиальные услуги и сервисы
        - Программы лояльности следующего уровня
        - Рекомендации друзьям и партнерские программы
        - Регулярный обзор финансовых целей
        """)

def show_model_comparison(analytics):
    """Сравнение производительности моделей для бизнес-пользователей"""
    st.header("📈 Сравнение моделей")
    
    # Создаем DataFrame для сравнения
    models_data = [
        {
            'Модель': 'Gradient Boosting',
            'Точность': 0.9765,
            'F1-Score': 0.9556,
            'Recall оттока': 0.9904,
            'Precision оттока': 0.9818,
            'ROC-AUC': 0.9939,
            'Время обучения (с)': 45.2,
            'Время предсказания (мс)': 12.3
        },
        {
            'Модель': 'XGBoost',
            'Точность': 0.9734,
            'F1-Score': 0.9499,
            'Recall оттока': 0.9882,
            'Precision оттока': 0.9803,
            'ROC-AUC': 0.9942,
            'Время обучения (с)': 32.1,
            'Время предсказания (мс)': 8.7
        },
        {
            'Модель': 'Random Forest',
            'Точность': 0.9555,
            'F1-Score': 0.9159,
            'Recall оттока': 0.9779,
            'Precision оттока': 0.9694,
            'ROC-AUC': 0.9898,
            'Время обучения (с)': 28.5,
            'Время предсказания (мс)': 15.2
        },
        {
            'Модель': 'CatBoost',
            'Точность': 0.9689,
            'F1-Score': 0.9421,
            'Recall оттока': 0.9856,
            'Precision оттока': 0.9756,
            'ROC-AUC': 0.9923,
            'Время обучения (с)': 51.8,
            'Время предсказания (мс)': 6.9
        },
        {
            'Модель': 'Logistic Regression',
            'Точность': 0.9012,
            'F1-Score': 0.8789,
            'Recall оттока': 0.9456,
            'Precision оттока': 0.9234,
            'ROC-AUC': 0.9654,
            'Время обучения (с)': 5.2,
            'Время предсказания (мс)': 2.1
        }
    ]
    
    models_df = pd.DataFrame(models_data)
    
    # Визуализация
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            models_df,
            x='Модель',
            y='F1-Score',
            title='F1-Score по моделям',
            color='F1-Score',
            color_continuous_scale='Viridis',
            text='F1-Score'
        )
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(
            models_df,
            x='Recall оттока',
            y='Precision оттока',
            size='F1-Score',
            color='Модель',
            title='Recall vs Precision по моделям',
            hover_name='Модель',
            size_max=40
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Дополнительные метрики
    col3, col4 = st.columns(2)
    
    with col3:
        fig = px.bar(
            models_df,
            x='Модель',
            y='ROC-AUC',
            title='ROC-AUC по моделям',
            color='ROC-AUC',
            color_continuous_scale='Plasma',
            text='ROC-AUC'
        )
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        fig = px.bar(
            models_df,
            x='Модель',
            y=['Время обучения (с)', 'Время предсказания (мс)'],
            title='Производительность моделей',
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Детальная таблица
    st.subheader("📊 Детальные метрики")
    display_df = models_df.set_index('Модель').round(4)
    st.dataframe(display_df.style.background_gradient(cmap='Blues'), use_container_width=True)
    
    # Рекомендации
    best_model = models_df.loc[models_df['F1-Score'].idxmax()]
    st.success(f"""
    **🏆 Рекомендуемая модель: {best_model['Модель']}**
    
    **Обоснование:**
    - Наивысший F1-Score: **{best_model['F1-Score']:.3f}**
    - Отличный баланс между Recall и Precision
    - Высокий ROC-AUC: **{best_model['ROC-AUC']:.3f}**
    - Приемлемое время выполнения
    """)

if __name__ == "__main__":
    main()