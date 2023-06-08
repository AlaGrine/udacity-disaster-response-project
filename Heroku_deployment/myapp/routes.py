from myapp import app

import json
import plotly
import pandas as pd
import numpy as np
import pickle
from flask import Flask
from flask import render_template, request, jsonify
import joblib
from sqlalchemy import create_engine, text as sql_text

# import re
# import string
# from nltk.stem import WordNetLemmatizer
# from nltk.tokenize import word_tokenize
# from wordcloud import WordCloud, STOPWORDS
# from nltk.corpus import stopwords

from myapp.plotly_figures import return_plots

# 1. Load data from SQLite dB
connection = create_engine('sqlite:///data/DisasterResponse.db')
query = "select * from Disaster"
df = pd.read_sql_query(con=connection.connect(),  sql=sql_text(query))

# 1. download data from csv file
#df = pd.read_csv('./data/disaster_clean.csv',dtype={"message": "string", "original": "string" ,"genre": "string"})

# 2.Get plotly figure configuration (call return_plots):
graphs_dahboard,graphs_classifier,graphs_metrics = return_plots(df)

# 3. Load model
# model = joblib.load("models/classifier.pkl") 
# NOK. Error: Can't get attribute 'tokenizer' on <module '__main__'>

# SOLUTION: 
# https://github.com/ania4data/Disaster_response_pipeline_app/blob/master/Disaster_response_app/wrangling_scripts/wrangle_data.py

global model
#update custom pickler
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'tokenize':
            from models.tokenizer import tokenize
            return tokenize
        return super().find_class(module, name)

model = CustomUnpickler(open('models/classifier.pkl', 'rb')).load()

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

# def main():
#     app.run(host='0.0.0.0', port=3001, debug=True)


# if __name__ == '__main__':
#     main()
