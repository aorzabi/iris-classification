import pickle
import warnings
import streamlit as st

warnings.filterwarnings("ignore")
from PIL import Image

pickle_in = open("classifier.pkl","rb")
classifler = pickle.load(pickle_in)

def predict_iris_variety(sepal_length,sepal_width,petal_length,petal_width):
    prediction = classifler.predict([[sepal_length,sepal_width,petal_length,petal_width]])
    print(prediction)
    return prediction

def Input_Output():
    st.title("Iris Variety Prediction")
    st.image("https://machinelearninghd.com/wp-content/uploads/2021/03/iris-dataset.png", width=600)

    st.markdown("You are using Streamlit...", unsafe_allow_html=True)
    sepal_length = st.text_input("Enter Sepal Length" , ".")
    sepal_width = st.text_input("Enter Sepal Width" , ".")
    petal_length = st.text_input("Enter Petal Length" , ".")
    petal_width = st.text_input("Enter Petal Width" , ".")
    
    result = ""
    if st.button("Click here to predict"):
        result = predict_iris_variety(sepal_length,sepal_width,petal_length,petal_width)
        st.balloons()
    st.success('The output is {}'.format(result))
if __name__ == '__main__':
    Input_Output()