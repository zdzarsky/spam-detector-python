from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords


def tokenize_content(subject, message):
    tokenizer = RegexpTokenizer(r'\w+')
    sub = tokenizer.tokenize(subject)
    mes = tokenizer.tokenize(message)
    return sub + mes


def lemmatize_content(subject, message):
    word_list = tokenize_content(subject, message)
    wnl = WordNetLemmatizer()
    lemmatized = " ".join(wnl.lemmatize(i) for i in word_list)
    return lemmatized


def remove_stopwords(message):
    stops = set(stopwords.words('english'))
    final_list = [word.lower() for word in message.lower().split() if
                  word not in stops]
    return ' '.join(word for word in final_list)


def process_message(subject, message):
    content = lemmatize_content(subject, message)
    return remove_stopwords(content)
