import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
iris_df = pd.read_csv("iris-species.csv")
iris_df['Label'] = iris_df['Species'].map({'Iris-setosa': 0, 'Iris-virginica': 1, 'Iris-versicolor':2})
X = iris_df[['SepalLengthCm','SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris_df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)
score = svc_model.score(X_train, y_train)

@st.cache()
def prediction(sepal_length, sepal_width, petal_length, petal_width):
  species = svc_model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
  species = species[0]
  if species == 0:
    return "Iris-setosa"
  elif species == 1:
    return "Iris-virginica"
  else:
    return "Iris-versicolor"

st.title("Iris Flower Species Prediction App")
st.subheader.title("Enter the following:")
s_len = st.subheader.slider("Sepal Length", 0.0, 10.0)
s_wid = st.subheader.slider("Sepal Width", 0.0, 10.0)
p_len = st.subheader.slider("Petal Length", 0.0, 10.0)
p_wid = st.subheader.slider("Petal Width", 0.0, 10.0)
if st.sidebar.button("Predict"):
	species_type = prediction(s_len, s_wid, p_len, p_wid)
	st.write("Species predicted:", species_type)
	st.write("Accuracy score of this model is:", score)