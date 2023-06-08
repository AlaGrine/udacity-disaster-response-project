import json
import plotly
import pandas as pd
import numpy as np
import colorlover as cl
from plotly.graph_objs import Bar, Pie, Histogram, Heatmap, Scatter

from wordcloud_parameters import worldcloud_generator, wordcloud_params

def return_plots(df):
    """
    Return Plotly figure configuration (including data and layout config).

    Parameters
    ----------- 
        df: our cleanded dataframe (loaded from SQLite database).
    
    Output
    ----------- 
        graphs_dahboard,graphs_classifier,graphs_metrics: three Plotly figure config
        
    """
    
    # 1. Extract data needed for visuals

    # 1.1 Distribution of Message Genres
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # 1.2 Distribution by categories
    category_counts = df.iloc[:, 4:].sum().sort_values(ascending=False)
    category_direct_counts = df[df.genre == 'direct'].iloc[:, 4:].sum().sort_values(ascending=False)
    category_news_counts = df[df.genre == 'news'].iloc[:,4:].sum().sort_values(ascending=False)
    category_social_counts = df[df.genre == 'social'].iloc[:, 4:].sum().sort_values(ascending=False)

    # 1.3 Message length (keep only 99% of samples as the rest are outliers)
    message_length_df = df['message'].apply(lambda s: len(s.split(' ')))
    percentile_99 = np.percentile(message_length_df, 99)
    message_length_df = message_length_df[message_length_df < percentile_99]

    # 1.4 WordCloud
    wc = worldcloud_generator(df['message'], background_color='white', max_words=200)
    # 1.5 Get wordcloud parametres (positions, word frequency, colors...)
    position_x_list, position_y_list, freq_list, size_list, color_list, word_list = wordcloud_params(wc)

    # 1.6 Get model metrics
    df_model_metrics = pd.read_csv('..\models\metrics.csv')

    metrics_f1 = df_model_metrics.pivot_table(
        values="f1-score", index='category', columns='class', aggfunc="mean").reset_index()
    metrics_recall = df_model_metrics.pivot_table(
        values="recall", index='category', columns='class', aggfunc="mean").reset_index()
    

    #####################################################
    #                   Dashboard figures
    #####################################################

    # Color palette (using colorlover):
    blue = cl.flipper()['seq']['9']['Blues']
    red = cl.flipper()['seq']['9']['Reds']
    colors = [blue[5], red[3], blue[3], red[4]]   

    # 2. Create visuals
    graphs_dahboard = [
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

    #####################################################
    #                 Classifier figures
    #####################################################

    # Color palette (using colorlover):
    blue = cl.flipper()['seq']['9']['Blues']
    red = cl.flipper()['seq']['9']['Reds']
    colors = [blue[5], red[3], blue[3], red[4]]   

    # Create visuals
    graphs_classifier = [
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
        }
    ]

    #####################################################
    #                 Metrics - figures
    #####################################################
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
    graphs_metrics = [
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


    return graphs_dahboard,graphs_classifier,graphs_metrics