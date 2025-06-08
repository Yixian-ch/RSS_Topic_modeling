from dataclasses import dataclass, field, asdict
import json
import pickle
from pathlib import Path
import xml.etree.ElementTree as ET
from argparse import ArgumentParser
import re

@dataclass
class Token:
    id : str = None
    form : str  = None# Forme originale du mot
    lemma : str = None # Lemme
    pos : str = None # Partie du discours 
    deprel : str = None # Relation de dépendance

@dataclass
class Article:
    id: str
    source: str
    title: str
    description: str
    date: str  
    categories: list[str] = field(default_factory=list)  # une liste vide par défaut
    analysis: list[list[Token]] = field(default_factory=list) #store analyzed tokens here

@dataclass
class Corpus:
    articles: list[Article] # quand on veut créer un argument avec dataclass, on utilise :, pas =

    
def save_json(corpus:Corpus, output_file: Path, encoding="utf-8") -> None:
    data = []
    for article in corpus.articles:
        current = {
            "id":article.id,
            "source":article.source,
            "title":article.title,
            "description":article.description,
            "date":article.date,
            "categories":sorted(article.categories),
            "analysis": [[asdict(token) for token in sentences] for sentences in article.analysis]
        }
        data.append(current)

    with open(output_file, 'w',encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def read_json(input_file: Path) -> Corpus:
    list_articles = []
    with open(input_file,'rb') as f:
        articles = json.load(f)
    for article in articles: # articles est une list de dictionnaire
        current = Article(
            id= article["id"],
            source= article["source"],
            title= article["title"],
            description= article["description"],
            date= article["date"],  
            categories= set(article["categories"]),
            )
        list_articles.append(current)

    return Corpus(list_articles)

def save_pickle(corpus:Corpus, output_file: Path) -> None:
    with open(output_file, "wb") as f:
        pickle.dump(corpus, f)

def read_pickle(input_file: Path) -> Corpus:
    with open(input_file, "rb") as f:
        data = pickle.load(f)
    return data

def save_xml(corpus:Corpus, output_file: Path) -> None: # some articles contain xml characters like < which will cause the problem
    root = ET.Element("corpus")
    with open(output_file, "w") as f:
        for item in corpus.articles:
            item_e = ET.SubElement(root,"item")

            id_e = ET.SubElement(item_e,"id")
            id_e.text = item.id
            
            source_e = ET.SubElement(item_e,"source")
            source_e.text = item.source

            title_e = ET.SubElement(item_e,"title")
            title_e.text = item.title

            description_e = ET.SubElement(item_e,"description")
            description_e.text = item.description

            date_e = ET.SubElement(item_e,"date")
            date_e.text = item.date

            cate_e = ET.SubElement(item_e,"categories")
            for cat in sorted(item.categories):
                cat_e = ET.SubElement(cate_e,"category")
                cat_e.text = cat

            analysis_e = ET.SubElement(item_e,"analysis")
            for sentence in item.analysis:
                sent_e = ET.SubElement(analysis_e,"sentence")
                for token in sentence:
                    token_e = ET.SubElement(sent_e,"token")
                    text_e = ET.SubElement(token_e, "text").text = token.form
                    lemma_e = ET.SubElement(token_e, "lemma").text = token.lemma
                    pos_e = ET.SubElement(token_e, "pos").text = token.pos
                    deprel_e = ET.SubElement(token_e, "deprel").text = token.deprel

    tree = ET.ElementTree(root)
    ET.indent(tree)
    tree.write(output_file, encoding="utf-8", xml_declaration=True)
            


def read_xml(input_file: Path) -> Corpus:
    articles = []
    with open(input_file, "r") as f:
        data = f.read()
        data = data.split("</article>")[:-1]
        for article in data:
            id = re.search("<id>.*</id>",article).group(0)
            source = re.search(r"<source>.*</source>", article).group(0)
            title = re.search(r"<title>.*</title>", article).group(0)
            desc = re.search(r"<description>.*</description>", article).group(0)
            date = re.search(r"<date>.*</date>", article).group(0)
            categories = re.search(r"<categories>.*</categories>", article).group(0)
            article = Article(
                        id=id,
                        source=source,
                        title=title,
                        description=desc,
                        date=date,
                        categories=set(categories.split(","))
                    )
            articles.append(article)

    return Corpus(articles)

name_to_save = {
    "xml":save_xml,
    "json":save_json,
    "pickle":save_pickle
}

name_to_load = {
    "xml":read_xml,
    "pickle":read_pickle,
    "json":read_json
}

def main(input_file, output_file, loader, saver):
    load = name_to_load[loader]
    save = name_to_save[saver]

    corpus = load(input_file)

    save(corpus, Path(output_file))

if __name__ == "__main__":
    parser = ArgumentParser("(de)serialize some RSS data.")

    parser.add_argument("input_file", help="input serialized RSS file")
    parser.add_argument("output_file", help="output serialized RSS file")
    parser.add_argument("-l", "--loader", choices=("xml",
                        "json", "pickle"), required=True)
    parser.add_argument("-s", "--saver", choices=("xml",
                        "json", "pickle"), required=True)
    args = parser.parse_args()

    main(args.input_file, args.output_file, args.loader, args.saver)