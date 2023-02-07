from flask import Flask, render_template, url_for, request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
# from sklearn.externals import joblib

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict_fun():
	
    clf = pickle.load(open('NB_spam_model.pkl','rb'))    
    cv = pickle.load(open('cv.pkl','rb'))

    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
        print(my_prediction)
	
    return render_template('result.html',prediction = my_prediction)


if __name__ == '__main__':
	app.run(host="0.0.0.0", port=80, debug=False)
