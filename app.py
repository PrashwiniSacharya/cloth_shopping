from flask import Flask,render_template,request
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
app = Flask(__name__)

#Load the saved model
model = pickle.load(open('model_save.pkl','rb'))
cv = pickle.load(open('cv.pkl','rb'))
import nltk
nltk.download('stopwords')


def clean_review(review):
    stp_words = stopwords.words('english')
    clean_review = " ".join(word for word in review.split() if word not in stp_words)
    return clean_review

#Function to predict sentiment
def predict_sentiment(review_text):
    #Preprocess the input review
    cleaned_review = clean_review(review_text)
    #Transfrom the review using the TF-IDF vectorizer
    transformed_review = cv.transform([cleaned_review]).toarray()
    #Predict sentiment using the trained model
    prediction = model.predict(transformed_review)

    if prediction[0] == 0:
        return 'Negative'
    else:
        return 'Positive'

#Route for the index page
@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/products')
def products():
    return render_template('products.html')

# @app.route('/single-product')
# def single_product():
#     return render_template('single-product.html')

#Route to handle from submission and display result
@app.route('/result', methods = ['POST'])
def result():
    if request.method == 'POST':
        review = request.form['review']
        result = predict_sentiment(review)
        return render_template('result.html',review = review, result = result)

if __name__ == '__main__':
    app.run(debug = True)
