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

from wordcloud_parameters import worldcloud_generator, wordcloud_params

import colorlover as cl

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

# 3. Extract data needed for visuals

# 3.1 Distribution of Message Genres
genre_counts = df.groupby('genre').count()['message']
genre_names = list(genre_counts.index)

# 3.2 Distribution by categories
category_counts = df.iloc[:, 4:].sum().sort_values(ascending=False)
category_direct_counts = df[df.genre == 'direct'].iloc[:, 4:].sum().sort_values(ascending=False)
category_news_counts = df[df.genre == 'news'].iloc[:,4:].sum().sort_values(ascending=False)
category_social_counts = df[df.genre == 'social'].iloc[:, 4:].sum().sort_values(ascending=False)

# 3.3 Message length (keep only 99% of samples as the rest are outliers)
message_length_df = df['message'].apply(lambda s: len(s.split(' ')))
percentile_99 = np.percentile(message_length_df, 99)
message_length_df = message_length_df[message_length_df < percentile_99]

# 3.4 WordCloud
wc = worldcloud_generator(df['message'], background_color='white', max_words=200)
# 3.5 Get wordcloud parametres (positions, word frequency, colors...)
position_x_list, position_y_list, freq_list, size_list, color_list, word_list = wordcloud_params(wc)

# 3.6 Get model metrics
df_model_metrics = pd.read_csv('..\models\metrics.csv')

metrics_f1 = df_model_metrics.pivot_table(
    values="f1-score", index='category', columns='class', aggfunc="mean").reset_index()
metrics_recall = df_model_metrics.pivot_table(
    values="recall", index='category', columns='class', aggfunc="mean").reset_index()

# 4. Dashboard page
@app.route('/')
@app.route('/dashboard')
def dashboard():
    # Color palette (using colorlover):
    blue = cl.flipper()['seq']['9']['Blues']
    red = cl.flipper()['seq']['9']['Reds']
    colors = [blue[5], red[3], blue[3], red[4]]   

    # Create visuals
    graphs = [
        # Graph 1 - Pie chart - Distribution of Message Genres
        {
            'data': [
                Pie(
                    labels=genre_names,
                    values=genre_counts,
                    # textinfo='label+percent',
                    hole=0.5,
                    marker={"colors": colors},
                    sort=False,

                )
            ],
            'layout': {
                'title': 'Distribution of Message Genres',
                'showlegend': True,
                'hoverlabel': dict(bgcolor="#444", font_size=13, font_family="Lato, sans-serif"),
                'legend': {'orientation': 'h', 'xanchor': "center", 'x': 0.5, 'y': -0.15}
            }
        },

        # Graph 2 - Distribution of message length
        {
            'data': [
                Histogram(x=message_length_df)
            ],
            'layout': {
                'title': "Distribution of Message Lengths",
                'yaxis': {'title': 'Count'},
                'xaxis': {'title': 'Number of words in message'}
            }
        },

        # Graph 3 - Distribution of Categories
        {
            'data': [
                Bar(
                    x=category_counts.index,
                    y=category_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Categories',
                'yaxis': {'title': 'Count'},
                'xaxis': {'title': ''},
            }
        },

        # Graph 4- Distribution of Categories and genres
        {
            'data': [
                Bar(
                    x=category_direct_counts.index,
                    y=category_direct_counts,
                    name='direct',
                    marker_color=colors[0]
                ),
                Bar(
                    x=category_news_counts.index,
                    y=category_news_counts,
                    name='news',
                    marker_color=colors[1]
                ),
                Bar(
                    x=category_social_counts.index,
                    y=category_social_counts,
                    name='social',
                    marker_color=colors[2]
                )
            ],
            'layout': {
                'title': 'Distribution of Categories per Genre',
                'yaxis': {'title': 'Count'},
                'xaxis': {'title': ''},
                'barmode': 'stack',
                # 'legend': {'orientation' : 'h', 'xanchor' : "center",'x': 0.85, 'y': 1.05}
            }
        },

        # Graph 5 - Wordcloud (Most common words)
        {
            'data': [
                Scatter(x=position_x_list,
                        y=position_y_list,
                        textfont=dict(size=size_list,
                                      color=color_list),
                        hoverinfo='text',
                        #hovertext=['{0}{1}'.format(w, f) for w, f in zip(word_list, freq_list)],
                        mode='text',
                        text=word_list
                        )
            ],
            'layout': {
                'xaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False},
                'yaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False},
                'height': 700,
                'title': 'Most Common Words',
            }
        }
    ]

    # Encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # Render web page with plotly graphs
    return render_template('dashboard.html', ids=ids, graphJSON=graphJSON)


# 5. Classifier page
# A webpage that receives user qurey (message) 
@app.route('/classifier')
def classifier():
    return render_template('classifier.html')


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
    
    # This will render the go.html file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


# 7. Model metrics page
@app.route('/model-eval')
def mleval():
    # Colorscale
    blue = cl.flipper()['seq']['9']['Blues']
    red = cl.flipper()['seq']['9']['Reds']
    
    colorscale=[
        # Let first 10% (0.1) of the values have color red[6]
        [0  , red[6]],
        [0.1, red[6]],

        # Let values between 10-20% of the min and max of z have color red[5
        [0.1, red[5]],
        [0.2, red[5]],

        # Values between 20-30% of the min and max of z have color red[4]
        [0.2, red[4]],
        [0.3, red[4]],

        [0.3, red[3]],
        [0.4, red[3]],

        [0.4, red[2]],
        [0.5, red[2]],

        [0.5, blue[1]],
        [0.6, blue[1]],

        [0.6, blue[2]],
        [0.7, blue[2]],

        [0.7, blue[3]],
        [0.8, blue[3]],

        [0.8, blue[4]],
        [0.9, blue[4]],

        [0.9, blue[5]],
        [1.0, blue[5]]
    ]
    
    # Create visuals
    graphs = [
        # Graph 1 - Heatmap of `F1-score`
        {
            # Reference: https://www.tutorialspoint.com/plotly/plotly_heatmap.htm
            'data': [
                Heatmap(
                    x=['(0)', '(1)', 'weighted avg'],
                    y=metrics_f1.category,
                    z=metrics_f1[['0', '1', 'weighted avg']].to_numpy(),
                    colorscale=colorscale
                )
            ],
            'layout': {
                'title': 'F1-score',
                'height': 800,
                'yaxis': dict( automargin=True ) # to prevent truncation of the y axes labels
            }
        },

        # Graph 2 - Heatmap of `Recall`
        {
            'data': [
                Heatmap(
                    x=['(0)', '(1)', 'weighted avg'],
                    y=metrics_recall.category,
                    z=metrics_recall[['0', '1', 'weighted avg']].to_numpy(),
                    type='heatmap',
                    colorscale=colorscale                    
                )
            ],
            'layout': {
                'title': 'Recall',
                'height': 800,
                'yaxis': dict( automargin=True ) #to prevent truncation of the y axes labels
            }
        }

    ]

    # Encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # Render web page with plotly graphs
    return render_template('model_metrics.html', ids=ids, graphJSON=graphJSON)


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
