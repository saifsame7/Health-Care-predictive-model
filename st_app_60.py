

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
# Cache the data loading and figure creation for performance

import plotly.express as px  
    # Load the Titanic dataset
df = pd.read_csv('full_data_used.csv')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Load pre-trained model and preprocessor
model = joblib.load("model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

st.title('Heart Disease Predictor')

st.subheader('EDA')











fig = make_subplots(  rows=3, cols=2,
        specs=[
        [{"type": "xy"}, {"type": "xy"}],
        [{"type": "xy"}, {"type": "domain"}],
        [{"type": "xy"}, {"type": "xy"}]
    ],
                    subplot_titles=('Survival Distribution', 
                                    'Age Distribution', 
                                    'smoking_count', 
                                    'Distribution of Physical Activity',
                                    'Correlation heatmap with adjusted colorbar',
                                    'smoking and heart disease'),
                    vertical_spacing=0.15,  # Add spacing to avoid overlap
                    horizontal_spacing=0.15
                    )
# Survival count plot
survived_counts = df['HeartDisease'].value_counts()
# Add "Did not survive" bar
fig.add_trace(
    go.Bar(
        x=['sick'], 
        y=[survived_counts.get(0, 0)], 
        name='sick',  # Explicit legend label
        marker=dict(color='red'),  # Red for "Did not survive"
        text=[survived_counts.get(0, 0)], 
        textposition='auto'
    ),
    row=1, col=1
)
fig.add_trace(
    go.Bar(
        x=['well'], 
        y=[survived_counts.get(1, 0)], 
        name='well',  # Explicit legend label
        marker=dict(color='green'),  # Blue for "Survived"
        text=[survived_counts.get(1, 0)], 
        textposition='auto'
    ),
    row=1, col=1
)
# Age distribution with histogram and KDE
age = pd.to_numeric(df['AgeCategory'], errors='coerce')
age = age.dropna()
age = df['AgeCategory']
kde = gaussian_kde(age)
x = np.linspace(age.min(), age.max(), 12)
kde_values = kde(x)
fig.add_trace(
go.Histogram(x=age, 
                nbinsx=20, 
                histnorm='probability density', 
                name='AgeCategory', 
                marker=dict(color='orange'),  # Set histogram color to orange
                opacity=0.7  # Add slight transparency to see KDE line
                ),
    row=1, col=2
)
fig.add_trace(
    go.Scatter(x=x, y=kde_values, mode='lines', name='KDE', line=dict(color='red')),
    row=1, col=2
)

smoking_count = df['Smoking'].value_counts()
# Add "smokers" bar
fig.add_trace(
    go.Bar(
        x=['YES'], 
        y=[smoking_count.get(0, 0)], 
        name='YES',  # Explicit legend label
        marker=dict(color='red'),  # Red for "Did not survive"
        text=[smoking_count.get(0, 0)], 
        textposition='auto'
    ),
    row=2, col=1
)

# Add "NO smokers" bar
fig.add_trace(
    go.Bar(
        x=['NO'], 
        y=[smoking_count.get(1, 0)], 
        name='NO',  # Explicit legend label
        marker=dict(color='green'),  # Blue for "Survived"
        text=[smoking_count.get(1, 0)], 
        textposition='auto'
    ),
    row=2, col=1
)
physicall=df['PhysicalActivity'].value_counts()
fig.add_trace(
    go.Pie(
    labels=physicall.index.tolist(),  # Gets unique categories in order
    values=physicall.values,  # Gets counts
    name="Physical Activity Distribution",
    marker=dict(colors=px.colors.qualitative.Pastel),
    textinfo='percent+label',
    hole=0.3,  # Optional: makes it a donut chart (remove if you want regular pie)
    pull=[0.1, 0, 0, 0]  # Optional: pulls out the first slice
),
    row=2 ,col=2)
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
corr_matrix = df[numerical_cols].corr()
fig.add_trace(
   
go.Heatmap(
        z=corr_matrix.values, 
        x=corr_matrix.columns, 
        y=corr_matrix.columns, 
        colorscale='RdBu',
        text=corr_matrix.values.round(2),  # Add correlation values rounded to 2 decimal places
        texttemplate="%{text}",  # Display the values in each cell
        textfont={"size": 10},  # Set font size for readability
        colorbar=dict(
            len=0.45,  # Set length to match the subplot height (0.4 is roughly the height of one subplot in a 2x2 grid)
            y=0.21,    # Position the colorbar to align with the bottom subplot (row 2)
            yanchor='middle'  # Center the colorbar vertically within the subplot
        )
    )
    ,row=3 ,col=1)



df.grouped = df.groupby(['Smoking', 'HeartDisease']).size().unstack(fill_value=0)


fig.add_trace(
    go.Bar(
        name='Sick = 1',
        x=df.grouped.index,
        y=df.grouped[1],
        marker_color='red'
    ),    row=3, col=2
)

fig.add_trace(
    go.Bar(
        name='Sick = 0',
        x=df.grouped.index,
        y=df.grouped[0],
        marker_color='blue'
    ),    row=3, col=2

)

fig.update_layout(
    height=1200, 
    width=1200, 
    title_text="Univariate and Multivariate Analysis",
    title_x=0.5,  # Center the title
    showlegend=True,
    barmode='group'  # Group the bars for Survival Distribution
)

# st.pyplot(fig)

st.plotly_chart(fig, use_container_width=False)

# Load pre-trained model and preprocessor
model = joblib.load("model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# Predictive interface
st.subheader("Model Prediction")
# BMI	Smoking	AlcoholDrinking	Stroke	PhysicalHealth	MentalHealth	DiffWalking	Sex	AgeCategory	Race	Diabetic	PhysicalActivity	GenHealth	SleepTime	Asthma	KidneyDisease	SkinCancer
# Input widgets
col1, col2 = st.columns(2)
with col1:
    BMI = st.number_input("BMI", value=0.0, format="%.2f")
    Smoking = st.selectbox("Smoking", [0,1])
    AlcoholDrinking = st.selectbox("AlcoholDrinking", [0,1])
    Stroke = st.selectbox("Stroke", [0,1])
    PhysicalHealth = st.number_input("PhysicalHealth", min_value=0, max_value=30, value=15)
    MentalHealth = st.number_input("MentalHealth", min_value=0, max_value=30, value=15)
    DiffWalking = st.selectbox("DiffWalking", [0,1])
    Sex = st.selectbox("Sex", [0,1])


with col2:
    AgeCategory = st.number_input("AgeCategory", min_value=0, max_value=12, value=10)
    Race = st.selectbox("Race", [0,1])
    Diabetic = st.selectbox("Diabetic", [0,1])
    PhysicalActivity = st.selectbox("PhysicalActivity", [0,1])
    GenHealth = st.number_input("GenHealth", min_value=0, max_value=4, value=2)
    SleepTime = st.number_input("SleepTime", min_value=0, max_value=24, value=10)
    Asthma = st.selectbox("Asthma", [0,1])
    KidneyDisease = st.selectbox("KidneyDisease", [0,1])
    SkinCancer = st.selectbox("SkinCancer", [0,1])

    # Prediction logic
if st.button("Predict case"):
    input_data = pd.DataFrame([[BMI, Smoking, AlcoholDrinking,Stroke,PhysicalHealth,MentalHealth,DiffWalking,Sex,AgeCategory,
                                   Race,Diabetic,	PhysicalActivity,	GenHealth,	SleepTime,	Asthma,	KidneyDisease,	SkinCancer]],
                                columns=['BMI', 'Smoking', 'AlcoholDrinking','Stroke','PhysicalHealth','MentalHealth','DiffWalking','Sex','AgeCategory',
                                   'Race',	'Diabetic',	'PhysicalActivity',	'GenHealth',	'SleepTime',	'Asthma',	'KidneyDisease',	'SkinCancer'])

    processed_data = preprocessor.transform(input_data)
    prediction = model.predict(processed_data)[0]
    probability = model.predict_proba(processed_data)[0][1]

    st.subheader("Result")
    st.metric("Survival Probability", f"{probability:.1%}")
    st.write(f"Prediction {'Survived' if prediction == 1 else 'Did not survive'}")
