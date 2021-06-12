import flask
import pickle
import pandas as pd
# Use pickle to load in the pre-trained model.
with open(f'CustomerChurn.pkl', 'rb') as f:
    model = pickle.load(f)
app = flask.Flask(__name__, template_folder='templates')
@app.route('/', methods=['GET', 'POST'])
def main():
    
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))
    if flask.request.method == 'POST':
        MonthlyCharges = flask.request.form['MonthlyCharges']
        SeniorCitizen = flask.request.form['SeniorCitizen']
        TotalCharges = flask.request.form['TotalCharges']
        tenure = flask.request.form['tenure']
        input_variables = pd.DataFrame([[MonthlyCharges , SeniorCitizen , TotalCharges , tenure ]],
                                       columns=["MonthlyCharges", "SeniorCitizen","TotalCharges","tenure"],
                                       dtype=float)
        prediction = model.predict(input_variables)[4]
        return flask.render_template('main.html',
                                     original_input={'MonthlyCharges':MonthlyCharges,
                                                     'SeniorCitizen':SenoirCitizen,
                                                     'TotalCharges':TotalCharges,
                                                     'tenure': tenure},
                                     result=prediction,
                                     )
if __name__ == '__main__':
    app.run(debug=False)





