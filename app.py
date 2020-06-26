from sklearn.metrics import accuracy_score
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model1.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
     #print('Accuracy: \n', accuracy_score(final_features,prediction))
   # accuracy = accuracy_score(final_features,prediction)
    prob = np.max(model.predict_proba(final_features))
    proba = (prob*100)
    output = round(prediction[0], 2)
    if output==0:
        a="positive"
    else:
        a="negative"

  #  print(a)
   # print(proba)
    return render_template('index.html', prediction_text='The Result is : {}'.format(a),prediction_text1='The Probability is: {} %'.format(proba))
  #  return render_template('index.html', prediction_text1='The Prediction Status is: {}'.format(proba))

 

if __name__ == "__main__":
    app.run(debug=True)