import streamlit as st
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt 
import seaborn as sns 

st.title("Churn Prediction App")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Introduction", "Analysis", "Prediction"])

df = pd.read_csv("Churn_Modelling.csv")
x = df.drop(columns=["Excited", "Surname", "RowNumber"])
y = df["Exited"]

x = pd.get_dummies(x);

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = LogisticRegression()
model.fit(x_train, y_train)

y_pred_test = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred_test)

if page == "Introduction":
    st.title("Welcome to the Churn Prediction Model")
    st.write("Top five rows of the dataset")
    st.write(df.head())
    st.write("### Dataset Statistics")
    st.write(df.describe())


elif page == "Analysis":
    st.title("Dataset Analysis")
    st.write("Use the options below to explore the dataset and model results.")

if st.button("show Scatter Plot(Age vs. Balance)"):
    st.subheader("Scatter Plot: Age vs. Balance")
    fig ax = plt.subplots(figsize=(2,2))
    sns.scatterplot(data=df, x="Age", y="Balance", pallet="viridis", alpha=0.7)
    plt.title("Age vs. Balance with Churn Status")
    plt.xlable("Age")
    plt.ylable("Balance")
    st.pyplot(fig)

    if st.button("Show Geography-wise Customer Count"):
        st.subheader("Geography-wise Customer Count")
        geography_count