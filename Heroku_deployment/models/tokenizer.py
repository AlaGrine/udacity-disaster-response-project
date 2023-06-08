import re
import string
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords

def tokenize(text):
    """
    Tokenize and preprocess text:
        1. find urls and replace them with 'urlplaceholder'.
        2. Normalization of the text : Convert to lowercase.
        3. Normalization of the text : Remove punctuation characters.
        4. Split text into words using NLTK.
        5. remove stop words.
        6. Lemmatization.    

    Parameters
    -----------
        text: text

    Returns
    -----------
        clean_tokens
    """
    # 1. find urls and replace them with 'urlplaceholder'
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
