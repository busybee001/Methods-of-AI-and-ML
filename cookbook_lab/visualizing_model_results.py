
import streamlit as st
import pandas as pd
import altair as alt

from sklearn.datasets import fetch_covtype, load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

st.title("Visualizing Model Results")

st.sidebar.header("Model and dataset selection")

dataset_name = st.sidebar.selectbox(
    "Dataset",
    ["Covertype", "Wine"]
)

n_estimators = st.sidebar.slider("n_estimators", 1, 100, 25)
max_depth = st.sidebar.slider("max_depth", 1, 150, 10)

model_name = st.sidebar.selectbox(
    "Model",
    ["DecisionTreeClassifier", "RandomForestClassifier", "ExtraTreesClassifier"]
)

@st.cache_data
def load_data(dataset_name):
    if dataset_name == "Covertype":
        dataset = fetch_covtype()
        X = dataset.data[:10000]
        y = dataset.target[:10000]
        feature_names = dataset.feature_names
    else:
        dataset = load_wine()
        X = dataset.data
        y = dataset.target
        feature_names = dataset.feature_names

    return X, y, feature_names

X, y, feature_names = load_data(dataset_name)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

if model_name == "DecisionTreeClassifier":
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
elif model_name == "RandomForestClassifier":
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
else:
    model = ExtraTreesClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )

model.fit(X_train, y_train)
predictions = model.predict(X_test)

st.subheader(f"{model_name} on {dataset_name}")

st.write("Model performance in test")
report = classification_report(y_test, predictions, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())

st.write("Confusion Matrix")
cm = confusion_matrix(y_test, predictions)
st.dataframe(pd.DataFrame(cm))

st.write("Test data with predictions")
test_df = pd.DataFrame(X_test, columns=feature_names)
test_df["actual"] = y_test
test_df["prediction"] = predictions
test_df["correct"] = test_df["actual"] == test_df["prediction"]
st.dataframe(test_df.head(100))

st.write("Scatter plot of variables")

x_axis = st.selectbox("X axis", feature_names)
y_axis = st.selectbox("Y axis", feature_names)

chart = alt.Chart(test_df.head(2000)).mark_circle(size=40).encode(
    x=x_axis,
    y=y_axis,
    color="prediction:N",
    tooltip=["actual", "prediction", "correct"]
).interactive()

st.altair_chart(chart, use_container_width=True)
