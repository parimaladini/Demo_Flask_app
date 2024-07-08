from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)


loaded_model = joblib.load('linear_regression_model.pkl')


features = ["Cycle", "Time Measured(Sec)", "Voltage Measured(V)", "Current Measured", "Temperature Measured"]


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        
        cycle = float(request.form['cycle'])
        time_measured = float(request.form['time_measured'])
        voltage_measured = float(request.form['voltage_measured'])
        current_measured = float(request.form['current_measured'])
        temperature_measured = float(request.form['temperature_measured'])
        
      
        custom_data = {
            "Cycle": [cycle],  
            "Time Measured(Sec)": [time_measured],  
            "Voltage Measured(V)": [voltage_measured],  
            "Current Measured": [current_measured],  
            "Temperature Measured": [temperature_measured]  
        }
        custom_data_df = pd.DataFrame(custom_data)

       
        X_custom = custom_data_df[features].values

    
        predictions = loaded_model.predict(X_custom)

       
        return render_template('predict.html', prediction=predictions[0])

if __name__ == '__main__':
    app.run(debug=True)
