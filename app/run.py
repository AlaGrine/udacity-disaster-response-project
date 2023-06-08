import json
import plotly
import pandas as pd
import numpy as np

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie, Histogram, Heatmap, Scatter
import joblib
from sqlalchemy import create_engine, text as sql_text

import re
import string
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords
import colorlover as cl

from plotly_figures import return_plots

app = Flask(__name__)


def tokenize(text):
    """
    Tokenize and preprocess text:
        1. find urls and replace them with 'urlplaceholder'
        2. Normalization of the text : Convert to lowercase
        3. Normalization of the text : Remove punctuation characters
        4. Split text into words using NLTK
        5. remove stop words
        6. Lemmatization  
    """
    # 1. Find urls and replace them with 'urlplaceholder'
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    text = re.sub(url_regex, 'urlplaceholder', text)

    # 2. Convert to lowercase
    text = text.lower().strip()

    # 3. Remove punctuation characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)

    # 4. Split text into words using NLTK
    words = word_tokenize(text)

    # 5. Remove stop words
    words = [w for w in words if w not in stopwords.words("english")]

    # 6. Lemmatization
    lemmatizer = WordNetLemmatizer()
    # 6.1 Reduce words to their root form
    lemmed = [lemmatizer.lemmatize(w) for w in words]
    # 6.2 Lemmatize verbs by specifying pos
    clean_tokens = [WordNetLemmatizer().lemmatize(w, pos='v') for w in lemmed]

    return clean_tokens

# 1 Load model
model = joblib.load("../models/classifier.pkl")

# 2. Load data
connection = create_engine('sqlite:///../data/DisasterResponse.db')
query = "select * from Disaster"
df = pd.read_sql_query(con=connection.connect(),  sql=sql_text(query))

# 3. Call `return_plots` to get plotly figure configuration:
graphs_dahboard,graphs_classifier,graphs_metrics = return_plots(df)

# 4. Classifier page
# A webpage that receives user qurey (message) 
@app.route('/')
@app.route('/classifier')
def classifier():
   # Encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs_classifier)]
    graphJSON = json.dumps(graphs_classifier, cls=plotly.utils.PlotlyJSONEncoder)

    # Render web page with plotly graphs
    return render_template('classifier.html', ids=ids, graphJSON=graphJSON)

# 5. Dashboard page
@app.route('/dashboard')
def dashboard():
    # Encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs_dahboard)]
    graphJSON = json.dumps(graphs_dahboard, cls=plotly.utils.PlotlyJSONEncoder)

    # Render web page with plotly graphs
    return render_template('dashboard.html', ids=ids, graphJSON=graphJSON)


# 6. Web page that handles user query and displays model results
@app.route('/go')
def go():
    # Save user input in query
    query = request.args.get('query', '')

    # Dispaly null if query message is empty:
    if (query.strip() == ""):
        return render_template(
            'go.html',
            query="",
            classification_result={'null': 0}
        )
    
    # Use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # Encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs_classifier)]
    graphJSON = json.dumps(graphs_classifier, cls=plotly.utils.PlotlyJSONEncoder)
    
    # This will render the go.html file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results,

        ids=ids, graphJSON=graphJSON


    )


# 7. Model metrics page
@app.route('/model-eval')
def mleval():
    # Encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs_metrics)]
    graphJSON = json.dumps(graphs_metrics, cls=plotly.utils.PlotlyJSONEncoder)

    # Render web page with plotly graphs
    return render_template('model_metrics.html', ids=ids, graphJSON=graphJSON)


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
