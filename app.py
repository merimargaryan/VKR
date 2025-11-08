import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Настройка страницы
st.set_page_config(
    page_title="Аналитика оттока клиентов банка",
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
</style>
""", unsafe_allow_html=True)

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
            # Загрузка моделей
            self.models = {
                'Gradient Boosting': joblib.load('models/gradient_boosting.pkl'),
                'XGBoost': joblib.load('models/xgboost.pkl'),
                'Random Forest': joblib.load('models/random_forest.pkl')
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
    st.markdown('<div class="main-header">🏦 Аналитика оттока клиентов банка</div>', unsafe_allow_html=True)
    
    # Инициализация аналитики
    with st.spinner("Загрузка моделей и данных..."):
        analytics = BankCustomerAnalytics()
    
    if not analytics.models:
        st.stop()
    
    # Сайдбар
    st.sidebar.title("Навигация")
    section = st.sidebar.radio("Выберите раздел:", [
        "📊 Общая аналитика",
        "🎯 Оценка клиента", 
        "📈 Сравнение моделей"
    ])
    
    if section == "📊 Общая аналитика":
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
            f"${biz_metrics['total_potential_loss']:,.0f}"
        )
    
    # Визуализации
    col1, col2 = st.columns(2)
    
    with col1:
        # Распределение рисков
        risk_data = {
            'Риск': ['Высокий', 'Средний', 'Низкий'],
            'Клиенты': [
                biz_metrics['high_risk_customers'],
                biz_metrics['medium_risk_customers'],
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
                'Средний': '#ffc107', 
                'Низкий': '#28a745'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Финансовый анализ
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
            color_discrete_sequence=['#ff6b6b', '#4ecdc4']
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Детальная аналитика
    st.subheader("📈 Детальная аналитика")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Ценных клиентов в риске", biz_metrics['high_value_at_risk'])
    
    with col2:
        st.metric("Средняя оценка риска", f"{biz_metrics['avg_risk_score']:.3f}")
    
    with col3:
        st.metric("Средняя транзакция", f"${biz_metrics['avg_transaction_value']:,.0f}")
    
    # Рекомендации
    st.subheader("🎯 Рекомендации для бизнеса")
    
    recommendations = [
        f"**Сфокусироваться на {biz_metrics['high_risk_customers']} клиентах высокого риска**",
        f"**Защитить {biz_metrics['high_value_at_risk']} ценных клиентов** от оттока",
        f"**Разработать программы удержания** для сегмента среднего риска",
        f"**Мониторить клиентов** со средней оценкой риска выше {biz_metrics['avg_risk_score']:.2f}",
        f"**Оптимизировать бюджет удержания** исходя из потенциальных потерь ${biz_metrics['total_potential_loss']:,.0f}"
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
        education_level = st.selectbox(
            "Уровень образования", 
            ["Unknown", "Uneducated", "High School", "College", "Graduate", "Post-Graduate", "Doctorate"]
        )
        marital_status = st.selectbox(
            "Семейное положение", 
            ["Unknown", "Single", "Married", "Divorced"]
        )
    
    with col2:
        st.write("**Финансовые показатели:**")
        income_category = st.selectbox(
            "Категория дохода", 
            ["Unknown", "Less than $40K", "$40K - $60K", "$60K - $80K", "$80K - $120K", "$120K +"]
        )
        card_category = st.selectbox(
            "Тип карты", 
            ["Blue", "Silver", "Gold", "Platinum"]
        )
        credit_limit = st.number_input("Кредитный лимит ($)", 1000, 50000, 5000)
        total_revolving_bal = st.number_input("Револьверный баланс ($)", 0, 5000, 500)
        total_trans_amt = st.number_input("Общая сумма транзакций ($)", 0, 50000, 5000)
    
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
        # Подготовка данных
        input_data = prepare_customer_data(
            customer_age, gender, dependent_count, education_level, marital_status,
            income_category, card_category, months_on_book, total_relationship_count,
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
        st.metric("Годовой CLV", f"${clv:,.0f}")
    
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
    """Сравнение производительности моделей"""
    st.header("📈 Сравнение моделей")
    
    # Загрузка метрик моделей
    try:
        with open('models/business_report_gradient_boosting.json', 'r') as f:
            gb_report = json.load(f)
        
        # Создаем DataFrame для сравнения
        models_data = []
        
        # Gradient Boosting
        models_data.append({
            'Модель': 'Gradient Boosting',
            'Точность': gb_report['model_performance']['accuracy'],
            'F1-Score': gb_report['model_performance']['f1_macro'],
            'Recall оттока': gb_report['model_performance']['recall_churn'],
            'Precision оттока': gb_report['model_performance']['precision_churn'],
            'ROC-AUC': gb_report['model_performance']['roc_auc']
        })
        
        # Добавьте данные для других моделей если есть
        
        models_df = pd.DataFrame(models_data)
        
        # Визуализация
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                models_df,
                x='Модель',
                y='F1-Score',
                title='F1-Score по моделям',
                color='Модель',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                models_df,
                x='Модель', 
                y='Recall оттока',
                title='Recall оттока по моделям',
                color='Модель',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Детальная таблица
        st.subheader("📊 Детальные метрики")
        st.dataframe(models_df.set_index('Модель'), use_container_width=True)
        
    except Exception as e:
        st.warning(f"Не удалось загрузить данные для сравнения моделей: {e}")

if __name__ == "__main__":
    main()