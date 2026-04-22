import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import streamlit as st
import pandas as pd
import plotly.express as px
from src.data_loader import load_data, get_summary

st.set_page_config(page_title='EDA', page_icon='📊', layout='wide')

@st.cache_data
def get_data():
    return load_data()

df = get_data()

st.title("📊 Exploratory Data Analysis")
st.caption("Understanding patterns, imbalance, and fraud indicators")

st.markdown("---")

st.header('1️⃣ What does this dataset look like?')

summary = get_summary(df)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows",         summary['Shape'][0])
col2.metric("Columns",      summary['Shape'][1])
col3.metric("Fraud Cases",  summary['fraud_cases'])
col4.metric("Fraud %",      f"{summary['fraud_percent']}%")

with st.expander("🔍 View Sample Data"):
    st.dataframe(df.head(), use_container_width=True)

with st.expander("📌 Data Types"):
    st.dataframe(df.dtypes, use_container_width=True)

st.markdown("---")

st.header("2️⃣ How imbalanced is the target?")

counts = df['Class'].value_counts().reset_index()
counts.columns = ['Class', 'Count']

col1, col2 = st.columns(2)

with col1:
    fig_bar = px.bar(counts, x='Class', y='Count',
                     title='Class Distribution',
                     color='Class')
    st.plotly_chart(fig_bar, use_container_width=True)

with col2:
    fig_pie = px.pie(counts, names='Class', values='Count',
                     title='Class Percentage')
    st.plotly_chart(fig_pie, use_container_width=True)

st.warning('⚠️ The dataset is highly imbalanced. Fraud cases are extremely rare, which can bias model performance.')

st.markdown('---')

st.header('3️⃣ How are Amount and Time distributed?')

col1, col2 = st.columns(2)

with col1:
    fig_amount = px.histogram(
        df, x='Amount',
        color=df['Class'].astype(str),
        nbins=50,
        barmode='overlay',
        title='Transaction Amount Distribution'
    )
    st.plotly_chart(fig_amount, use_container_width=True)

with col2:
    df_plot = df.copy()
    df_plot['Class'] = df_plot['Class'].map({0 : 'Legit', 1 : 'Fraud'})
    fig_box = px.box(df_plot, x='Class', y='Amount',
                     title='Amount by class',
                     color='Class')

    st.plotly_chart(fig_box, use_container_width=True)

st.markdown('---')

st.header('4️⃣ Which features separate fraud from legit?')

selected = st.multiselect(
    'Select features',
    [f'V{i}' for i in range(1, 29)],
    default=['V1', 'V2', 'V3']
)

for feature in selected:
    fig = px.histogram(
        df, x=feature,
        color=df['Class'].astype(str),
        barmode='overlay',
        title=f'{feature} Distribution by Class'
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown('---')

st.header('5️⃣ Which features are correlated?')

@st.cache_data
def compute_corr(data):
    return data.corr()

corr = compute_corr(df)

fig_heatmap = px.imshow(
    corr,
    title='Correlation Heatmap'
)
st.plotly_chart(fig_heatmap, use_container_width=True)

# Correlation with target
target_corr = corr['Class'].abs().sort_values(ascending=False)[1:11].reset_index()
target_corr.columns = ['Feature', 'Correlation']

fig_bar = px.bar(
    target_corr,
    x='Feature',
    y='Correlation',
    title='Top Features Correlated with Fraud'
)
st.plotly_chart(fig_bar, use_container_width=True)

st.markdown('---')


st.header('6️⃣ What should we do before modelling?')

st.success('''
✔ Fraud cases are extremely rare (high class imbalance)  
✔ Certain features (V1, V2, V3...) show strong separation  
✔ Amount distribution differs between fraud & legit  
✔ Some features are highly correlated  
''')

st.info('''
📌 Next Steps:
- Handle imbalance using SMOTE or undersampling  
- Normalize Amount & Time  
- Select important features  
- Train multiple models (Logistic, Random Forest, Gradient Boosting)  
''')