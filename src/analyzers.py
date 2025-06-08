from datastructures import *
import spacy
# from trankit import Pipeline

def analyser_stanza(article:Article,analyser:callable=None)-> Article: 
    '''
    Proriétés d'analyse : pour word : lemma, pos, depl, udep...
    '''
    text = analyser((article.title) + "\n" + (article.description))
    analysis = []
    for sent in text.sentences:
        analysis.append([])
        for word in sent.words:
            analysis[-1].append(Token(
                id=word.id,
                form=word.text,
                lemma=word.lemma,
                pos=word.upos,
                deprel=word.deprel,
            ))
    article.analysis = analysis
    return article

def analyzer_trankit(article: Article,analyser:callable=None) -> Article:
    # only modify the attribute analysis of article
    result = analyser((article.title) + "\n" + (article.description))
    analysis = []
    for sent in result['sentences']:
        analysis.append([])
        for token in sent['tokens']:
            if "expanded" not in token.keys():
                token['expanded'] = [token]
            for w in token['expanded']:
                analysis[-1].append(Token(w['id'],w['text'], w['lemma'], w['upos'],w['deprel']))

    article.analysis = analysis
    return article

def analyze_with_spacy(article: Article, analyser: callable = None) -> Article:
    """
    Analyse un article avec spaCy et enrichit ses tokens.
    """
    result = analyser((article.title) + "\n" + (article.description))
    analysis = []
    for sentence in result.sents:
        analysis.append([])
        for token in sentence:
            if token.text.strip():
                analysis[-1].append(
                    Token(
                        form=token.text,
                        lemma=token.lemma_,
                        pos=token.pos_,
                        deprel=token.dep_,
                    )
                )
    article.analysis = analysis
    return article

# 新增批量处理函数
def batch_analyze_with_spacy(articles: list, analyser: callable = None) -> list:
    """
    批量处理文章，使用spacy的pipe方法加速分析。
    """
    texts = [(article.title + "\n" + article.description) for article in articles]
    docs = list(analyser.pipe(texts))
    for article, doc in zip(articles, docs):
        analysis = []
        for sentence in doc.sents:
            analysis.append([])
            for token in sentence:
                if token.text.strip():
                    analysis[-1].append(
                        Token(
                            form=token.text,
                            lemma=token.lemma_,
                            pos=token.pos_,
                            deprel=token.dep_,
                        )
                    )
        article.analysis = analysis
    return articles

def load_stanza():
    import stanza
    return stanza.Pipeline('fr',processors='tokenize,pos,lemma,depparse')

parsers = {
    "stanza":analyser_stanza,
    "trankit":analyzer_trankit,
    "spacy":analyze_with_spacy

}

models = {
    "stanza": None,
    "spacy":spacy.load("fr_core_news_sm"),
    # "trankit":Pipeline(lang='french', gpu=False)
}

def main():
    parser = ArgumentParser()


    parser.add_argument("load",help="load a serialiazed file")
    parser.add_argument("output_file", help="output analysed file")
    parser.add_argument("saver",choices=("json","xml","pickle"))
    parser.add_argument("parser",choices=("stanza","spacy","trankit"))
    parser.add_argument("-l", "--loader", choices=("xml",
                        "json", "pickle"), required=True)
    
    args = parser.parse_args()

    loader = name_to_load.get(args.loader) # reader of different formats: json, xml and pickle
    corpus:Corpus = loader(args.load) # read the file which attribute is a list of dict

    load_analyzer = models.get(args.parser) # load model
    analyzer = parsers.get(args.parser)

    corpus.articles = [analyzer(article,load_analyzer) for article in corpus.articles] # analyse, returns a list of Articles

    saver = name_to_save.get(args.saver)
    saver(corpus,Path(args.output_file))

if __name__ == "__main__":
    main()