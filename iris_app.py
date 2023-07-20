import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

iris_df = pd.read_csv("iris-species.csv")
iris_df['Label'] = iris_df['Species'].map({'Iris-setosa': 0, 'Iris-virginica': 1, 'Iris-versicolor':2})
X = iris_df[['SepalLengthCm','SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris_df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)
rf_clf = RandomForestClassifier(n_jobs = -1, n_estimators = 100)
rf_clf.fit(X_train, y_train)
log_reg = LogisticRegression(n_jobs = -1)
log_reg.fit(X_train, y_train)

@st.cache()
def prediction(model,sepal_length, sepal_width, petal_length, petal_width):
  species = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
  species = species[0]
  if species == 0:
    return "Iris-setosa"
  elif species == 1:
    return "Iris-virginica"
  else:
    return "Iris-versicolor"

st.title("Iris Flower Species Prediction App")
st.sidebar.title("Enter the following:")
s_len = st.sidebar.slider("Sepal Length", 0.0, 10.0)
s_wid = st.sidebar.slider("Sepal Width", 0.0, 10.0)
p_len = st.sidebar.slider("Petal Length", 0.0, 10.0)
p_wid = st.sidebar.slider("Petal Width", 0.0, 10.0)
classifier = st.sidebar.selectbox("Classifier type:", ("Support Vector Machine", "Logistic Regression", "Random Forest Classifier"))
if st.sidebar.button("Predict"):
	if classifier == "Support Vector Machine":
		species_type = prediction(svc_model,s_len, s_wid, p_len, p_wid)
		score = svc_model.score(X_train, y_train)
	elif classifier == "Logistic Regression":
		species_type = prediction(log_reg,s_len, s_wid, p_len, p_wid)
		score = log_reg.score(X_train, y_train)
	else:
		species_type = prediction(rf_clf,s_len, s_wid, p_len, p_wid)
		score = rf_clf.score(X_train, y_train)
	st.write("Species predicted:", species_type)
	st.write("Accuracy score of this model is:", score)
