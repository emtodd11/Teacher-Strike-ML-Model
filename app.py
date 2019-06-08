# Create API of ML model using flask

'''
This code takes the JSON data while POST request an performs the prediction using loaded model and returns
the results in JSON format.
'''

# Import libraries
from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict',methods=['POST'])
def predict():

        df = pd.read_csv("data.csv")

        X1 = df.drop("Strike",axis=1)
        X2 = X1.drop("District",axis=1)
        X = X2.drop("State",axis=1)
        y = df["Strike"]

        class_weight = {0:0.07, 1:0.93}

        X_scaler = StandardScaler().fit(X)
        X_scaled = X_scaler.transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=1)

        classifier = LogisticRegression(class_weight=class_weight)

        classifier.fit(X_train, y_train)

        y_predict = classifier.predict(X_test)

        train_pred = classifier.predict(X_train)

        if request.method == "POST":

                data = []

                data.append(float(request.form['numStudents']))
                data.append(float(request.form['charterPercent'])/100)
                data.append(float(request.form['ELLPercent'])/100)
                data.append(float(request.form['IEPPercent'])/100)
                data.append(float(request.form['lowIncomePercent'])/100)
                data.append(float(request.form['studentRatio']))
                data.append(float(request.form['revenueRatio']))

                prediction = classifier.predict_proba(X_scaler.transform([data]))
                output = round(prediction[0][1],2) * 100


        return render_template("results.html", output=output)


if __name__ == '__main__':
    app.run(debug=True)
