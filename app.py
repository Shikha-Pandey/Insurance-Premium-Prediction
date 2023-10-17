# from flask import Flask, request, render_template
# import pandas as pd
# import pickle

# app = Flask(__name__)


# file = open("D:\DS Projects\insurance\insurance_pemium_mdl.pkl", 'rb')
# model = pickle.load(file)

# data = pd.read_csv('D:\DS Projects\insurance\insurance.csv')
# data.head()

# @app.route('/')
# def index():
#     sex = sorted(data['sex'].unique())
#     smoker = sorted(data['smoker'].unique())
##     region = sorted(data['region'].unique())
#     return render_template('index.html', sex= sex, smoker= smoker, region= region)

# @app.route('/predict', methods=['POST'])
# def predict():
#     age = int(request.form.get('age'))
#     sex = request.form.get('sex')
#     bmi = float(request.form.get('bmi'))
#     children = int(request.form.get('children'))
#     smoker = request.form.get('smoker')
##     region = request.form.get('region')

#     prediction = model.predict(pd.DataFrame([[age, sex, bmi, children, smoker, region]], 
#                 columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region']))

#     return str(prediction[0])           

# if __name__=="__main__":
#     app.run(debug=True)
import gradio as gr
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('insurance_pemium_mdl.pkl')

def predict(age, sex, bmi, children, smoker, region):
    # Create a DataFrame with the input features
    data = {
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region]
    }
    df = pd.DataFrame(data)

    # Make the prediction using the loaded model
    prediction = model.predict(df)[0]

    return float(prediction)  # Convert the prediction to a float

iface = gr.Interface(fn=predict,
                     inputs=[
                         gr.inputs.Number(label="Age"),
                         gr.inputs.Radio(["Male", "Female"], label="Sex"),
                         gr.inputs.Number(label="BMI"),
                         gr.inputs.Number(label="Children"),
                         gr.inputs.Radio(["Yes", "No"], label="Smoker"),
                         gr.inputs.Radio(["Northeast", "Northwest", "Southeast", "Southwest"], label="Region")
                     ],
                     outputs="number",
                     title="Insurance Premium Prediction",
                     description="Predict the insurance premium based on the given features.")

if __name__ == '__main__':
    iface.launch(share=True)





