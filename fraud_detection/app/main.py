import streamlit as st

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon='🤞',
    layout="wide")
st.title("🤞 Credit card Fraud Detection Dashboard")
st.caption("An interactive overview of fraud detection insights and ML Pipeline")

st.markdown('---')

col1, col2, col3, col4 = st.columns(4)

with col1 :
    st.metric("Total Transaction", '284,807')

with col2:
    st.metric("Fraud Cases", "492")

with col3:
    st.metric("Fraud Rate", "0.17%")

with col4:
    st.metric("Features", "30")

col1, col2, col3 = st.columns(3)


with col1:
    st.info('''
    **📊 Data Overview**
    - Dataset summary and structure  
    - Class imbalance visualization  
    - Feature distribution insights  
    ''')

with col2:
    st.info('''
    **🤖 Model Training**
    - Machine learning models used  
    - Training & validation process  
    - Performance comparison  
    ''')

with col3:
    st.info('''
    **🚨 Fraud Detection**
    - Real-time prediction system  
    - Input transaction testing  
    - Fraud probability output  
    ''')

st.subheader('🔄 ML Pipeline Flow')

st.code("""
1. Load Dataset
2. Data Preprocessing
   - Handle missing values
   - Feature scaling
3. Exploratory Data Analysis (EDA)
4. Handle Class Imbalance (SMOTE / Undersampling)
5. Train-Test Split
6. Model Training
   - Logistic Regression
   - Random Forest
   - Gradient Boosting
6. Model Evaluation
   - Accuracy
   - Precision, Recall
   - ROC-AUC
7. Deployment (Streamlit App)
""", language='text')

st.markdown('---')

# 6. Tech Stack
st.subheader('🛠 Tech Stack')

st.markdown("""
- **Frontend:** Streamlit  
- **Backend:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn  
- **Visualization:** Matplotlib, Seaborn  
- **Modeling:** Logistic Regression, Random Forest, Gradient Boosting  
- **Deployment:** Streamlit Cloud / Local Server  
""")