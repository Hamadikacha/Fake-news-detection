"""
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

app = Flask(__name__)
data=pd.read_csv('Concat_all (1).csv')

X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = TfidfVectorizer(stop_words='english', max_features= 5000)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)





# Load the saved vectorizer and svm model

#with open('model_vectorizer.pkl', 'rb') as f:
#    vectorizer = pickle.load(f)

with open('model_final_passive_agressive.pkl', 'rb') as f:
    passive = pickle.load(f)


def fake_news_det(news):
    #tfid_x_train = tfvect.fit_transform(x_train)
    vectorized_input_data = vectorizer.transform([news])
    
    prediction =  passive.predict(vectorized_input_data)
    return prediction
#message = "Nicolas Sarkozy Sarkozy to wear tag after losing corruption appeal"
#pred = fake_news_det(message)
#print(pred)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        pred = fake_news_det(message)
        print(pred)
        return render_template('index.html', prediction=pred)
    else:
        return render_template('index.html', prediction="Something went wrong")

if __name__ == '__main__':
    app.run(debug=True)
"""

from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import requests 
import bs4
import numpy as np 
import json 


# Fonction pour calculer la similarité du cosinus entre deux informations
def cosine_similarity(text1, text2):
    # Convertir les informations en vecteurs de représentation
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2]).toarray()

    # Extraire les vecteurs de représentation
    vector1 = vectors[0]
    vector2 = vectors[1]

    # Calculer le produit scalaire des vecteurs
    dot_product = np.dot(vector1, vector2)

    # Calculer les normes des vecteurs
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)

    # Calculer la similarité du cosinus
    similarity = dot_product / (norm1 * norm2)

    return similarity

#search news 

def search(text):
    data=[]
    url = 'https://google.com/search?q=' + text 
    request_result=requests.get( url ) 
    soup = bs4.BeautifulSoup(request_result.text, "html.parser") 
    heading_object=soup.find_all( 'h3' ) 
    for info in heading_object: 
        print(info.getText())
        data.append(info.getText())
    return data

from textblob import TextBlob
def get_sentiment(text):
    vs = TextBlob(text).sentiment[0]
    if vs > 0:
        return 'Positive'
    elif vs < 0:
        return 'Negative'
    else:
        return 'Neutral'
    return vs    


app = Flask(__name__)
##tfvect = TfidfVectorizer(stop_words='english', max_df=0.7 , max_features= 5000)
##loaded_model = pickle.load(open('RandomForestClassifier.pkl', 'rb'))
tfidf_v=pickle.load(open("tfidf_v.pkl", 'rb'))
model=pickle.load(open("RandomForestClassifier.pkl", 'rb'))
#dataframe = pd.read_csv('Concat_all (1).csv')
#x = dataframe['text']
#y = dataframe['label']
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

def fake_news_det(news):
    #tfid_x_train = tfidf_v.fit_transform(x_train)
    #tfid_x_test = tfidf_v.transform(x_test)
    input_data = [news]
    vectorized_input_data = tfidf_v.transform(input_data)
    prediction = model.predict(vectorized_input_data)
    return prediction
data_new_cos={}
news={}

values_new_cos={}
keys_new_cos={}
sentiment={}



@app.route('/')
def home():
    return render_template('index.html',keys_new_cos={},values_new_cos={},sentiment={},news={},len = len(news) )

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        pred = fake_news_det(message)
       # print(pred)
        data = search(message)
        sentiment=get_sentiment(message)
        print(sentiment)
        for i in data :
            #data_new_cos={}
            cos_sin=cosine_similarity(message , i )
            data_new_cos[i]= cos_sin       
        #print(json_object)
        i=0
        for key in data_new_cos.keys():
         i+=1
         keys_new_cos[i]=key
        # print(data_new_cos[key])
         values_new_cos[i]=data_new_cos[key]

         
        return render_template('index.html', prediction=pred,keys_new_cos=keys_new_cos,values_new_cos=values_new_cos ,sentiment=sentiment, news = data_new_cos,len = len(news)  )
    else:
        return render_template('index.html', prediction="Something went wrong")

        
if __name__ == '__main__':
    app.run(debug=True)
