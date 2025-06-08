import nltk
from nltk import RegexpTokenizer
nltk.download('stopwords')
from nltk.corpus import stopwords
import json
from pathlib import Path
from typing import Generator
from nltk.stem.wordnet import WordNetLemmatizer
import numpy as np
from scipy import sparse
from gensim.corpora import Dictionary
from gensim.models import Phrases
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from gensim.models import LdaModel
from argparse import ArgumentParser
from rss_reader import rss_reader_etree
from datastructures import Corpus, Article


def loop_dirs(path:Path) -> list:
    full_path = []
    for c in path.iterdir():
        if c.is_dir():
            full_path.extend(loop_dirs(c))
        else:
            full_path.append(c)
    return full_path

# each article is the concatnation between description and title
def read_json(args) -> Generator:
    with open(args.corpus,'r') as file:
        articles = json.load(file)
        for article in articles:
            if args.pos:
                data = [word["form"] for analysis in article["analysis"] for word in analysis if word["pos"].lower()==pos.lower()]
                yield " ".join(data)
            else:
                data = article["description"] + " " + article["title"]
                yield data # deal with big data

def read_xml(args) -> Generator:
    data_paths = loop_dirs(Path(args.corpus))
    corpus = []
    for path in data_paths:
        corpus.append(rss_reader_etree(path))
    # Si l’utilisateur a précisé une POS et que l’article contient des tokens on filtre uniquement les mots correspondant à cette catégorie
    corpus = Corpus(corpus)
    corpus = [a for a in corpus.articles]
    for article in corpus:
        if args.pos and hasattr(article, "tokens") and article.tokens:
            #POS filtering
            data = [token.form for token in article.tokens if token.pos.lower() == pos.lower()]
            yield " ".join(data)
        else:
           for data in article.articles:
            #sinon on utilise simplement le description + title
                yield f"{data.title} {data.description}"

def read_pickle(args) -> Generator:
    #on ouvre et charge le fichier pickle
    with open(args.corpus, "rb") as f:
        corpus = pickle.load(f)

    #on vérifie que corpus contient bien une liste d'articles
    articles = corpus.articles if hasattr(corpus, "articles") else corpus

    #certains fichiers pickles contiennent .articles, d’autres non
    for article in corpus.articles if hasattr(corpus, "articles") else corpus:
        #si une POS est spécifiée et que l’article contient des tokens on garde uniquement les formes correspondant à la POS choisie
        if args.pos and hasattr(article, "tokens") and article.tokens:
            data = [token.form for token in article.tokens if token.pos.lower() == pos.lower()]
            yield " ".join(data)
        else:
            yield f"{article.title} {article.description}"

def read_txt(args):
    """
    Read iteratively a folder

    Args:
        ArgumentParser

    Returns:
        list
    """
    data_dir = Path(args.corpus)
    for sub_dir in data_dir.iterdir():
        for f in sub_dir.iterdir():
            with open(f,"r",encoding="iso-8859-15") as doc:
                data = doc.read()
            yield data

def build_vocab(corpus:list) -> list:
    """
    A vocabulary contains types(all words in a corpus without repetition)
    """
    vocab = set()
    for article in corpus:
        for token in article:
            vocab.add(token)
    return list(sorted(vocab))

def build_bow_matrix(corpus:list, vocab:list):
    """
    A BoW matrix has the length of the vocabulary, the width of articles' number
    """
    vocab_idx = {word:idx for idx, word in enumerate(vocab)}
    bow = np.zeros((len(corpus),len(vocab))) # width * length
    for id,article in enumerate(corpus):
        for word in article:
            bow[(id,vocab_idx[word])] += 1
    return bow

def build_bow_sparse(corpus:list, vocab:list):
    vocab_idx = {word:idx for idx, word in enumerate(vocab)}
    rows, cols, data = [], [], []
    
    # ... 
    for doc_id, article in enumerate(corpus):
        for word in article:
           
            rows.append(doc_id)
            cols.append(vocab_idx[word])
            data.append(article.count(word))
    
    bow = sparse.csr_matrix((data, (rows, cols)), 
                          shape=(len(corpus), len(vocab)))
    return bow


#### Preprocessing the data ####
def preprocesse(data:list,lemma:bool=True) -> None:
    stoplist = set(stopwords.words("french"))
    tokenizer = RegexpTokenizer(r'\w+') 
    if lemma:
        lemmatizer = WordNetLemmatizer()
        for idx in range(len(data)):
            data[idx] = [lemmatizer.lemmatize(token) for token in tokenizer.tokenize(data[idx].lower()) if token not in stoplist and not token.isnumeric() and len(token)>1]
    else:
        for idx in range(len(data)):
            data[idx] = [token for token in tokenizer.tokenize(data[idx].lower()) if token not in stoplist and not token.isnumeric() and len(token)>1]

def show_data(data:list):
    for idx, article in enumerate(data):
        if idx < 5:
            print(article)

# show_data(docs)

#### find bigrams(usually name entities) ####

def bigram(docs:list):
    # add bigrams and trigrams to docs (only ones that appear more than 15 times)
    # Phrases is a model learns from the given corpus the coccurrence or pair words
    bigram = Phrases(docs, min_count=15)

    for idx in range(len(docs)):
        for token in bigram[docs[idx]]: # calls __getitem__ allows to use [], read the list of tokens and returns a list of original pair words and pair words connected with _
            if '_' in token:
                docs[idx].append(token)


name_to_reader = {
    "xml":read_xml,
    "pickle":read_pickle,
    "json":read_json,
    "txt":read_txt
}

def main():
    parser = ArgumentParser()
    parser.add_argument("corpus",help="the path to a the corpora in following formats: xml, json and pickle")
    parser.add_argument("-t","--topics",type=int,default=10)
    parser.add_argument("-c","--chunksize",default=3000)
    parser.add_argument("-p","--passes",type=int,default=15)
    parser.add_argument("-i","--iterations",type=int,default=400)
    parser.add_argument("-e","--evalEvery",type=bool,default=False)
    parser.add_argument("-l","--loader",choices=("xml","json","pickle","txt"),default="json")
    parser.add_argument("--lemmatization",action="store_true",help="decide if we do lemmatization during the preprocessing step")
    parser.add_argument("--pos",help="let user decides if he only interests in a certain pos",type=str)
    args = parser.parse_args()

    #### finally train our model
    num_topics = args.topics
    chunksize = args.chunksize # number of document to process at once
    passes = args.passes # how many times the entier corpus is processed to get a more fine result
    iterations = args.iterations # how many times the model updates its parameters for each pass through chunk of documents
    eval_every = args.evalEvery # how often the model calculates perplexity(model quality matrix)

    reader = name_to_reader.get(args.loader)
    docs = list(reader(args))
    preprocesse(docs)
    bigram(docs)

    #### Remove rare and common tokens
    dictionay = Dictionary(docs)

    # Only when the option pos is not enabled remove words appear less than 10 documents, or more than 50% of the doc
    if not args.pos:
        dictionay.filter_extremes(no_above=0.5,no_below=10)
        
    docs = [dictionay.doc2bow(doc) for doc in docs] # a list of articles which is a list of tuples [[(word_idx, occurrence)]]
    temp = dictionay[0]
    id2word = dictionay.id2token

    model = LdaModel(
        corpus=docs,
        id2word=id2word,
        chunksize=chunksize,
        alpha="auto", # controls document-topic density parameter, if auto, let machies learn this parameter. How many topics contains a document
        eta="auto", # contrls topic-words density paremeter. How many words belong to a topic
        iterations=iterations,
        num_topics=num_topics,
        passes=passes,
        eval_every=eval_every
    )

    model.save("./lda_model")


if __name__ == "__main__":
    main()
