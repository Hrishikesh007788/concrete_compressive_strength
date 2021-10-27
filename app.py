import numpy as np
import pickle
import pandas as pd
import streamlit as st

pickle_in = open("strength.pkl", "rb")
compressive_strength = pickle.load(pickle_in)


def welcome():
    return "Welcome All"


def predict_strength(cement, blast_furnace_slag, superplasticizer, coarse_aggregate, fine_aggregate, age):
    prediction = compressive_strength.predict([[cement,blast_furnace_slag,superplasticizer,coarse_aggregate,fine_aggregate,age]])
    print(prediction)
    return prediction


def main():
    st.image('Images/ineuron-logo.png')
    st.title("Compressive Strength Predictor")
    if st.button("About"):
        st.caption("This is an ML app which predicts the compressive strength of concrete, given its attributes.")
    html_temp = """
    <div style="background-color:brown;padding:5px">
    <h2 style="color:white;text-align:center;">Enter values:</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    cement = st.text_input("Cement (kg/m^3)")
    blast_furnace_slag = st.text_input("Blast furnace slag (kg/m^3)")
    superplasticizer = st.text_input("Superplasticizer (kg/m^3)")
    coarse_aggregate = st.text_input("Coarse aggregate (kg/m^3)")
    fine_aggregate = st.text_input("Fine aggregate (kg/m^3)")
    age = st.text_input("Age (Days)")

    result = ""
    if st.button("Predict"):
        result = predict_strength(cement, blast_furnace_slag, superplasticizer, coarse_aggregate, fine_aggregate, age)
    st.success('The compressive strength (MPa) is {}'.format(result))
    if st.button("Acknowledgement"):
        st.markdown("Special thanks to Krish Naik sir and Sudhanshu Kumar sir for providing such a wonderful platform for beginners to learn and build end to end projects.")
    if st.button("Creator info"):
        st.markdown("Name: Hrishikesh Dutta.")
        st.markdown("Education: National Institute of Technology, Silchar.")
        st.markdown("Dept: Computer Science and Engineering.")
        st.markdown("Year of graduation: 2023.")
        st.markdown("LinkedIn: https://www.linkedin.com/in/hrishikesh-dutta-6776321a0/")
        st.markdown("GitHub: https://github.com/Hrishikesh007788")

if __name__ == '__main__':
    main()



