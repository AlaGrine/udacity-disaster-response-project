from flask import Flask

app = Flask(__name__)


from myapp import plotly_figures
from myapp import wordcloud_parameters
from myapp import routes

