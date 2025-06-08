from typing import Callable
import argparse
import re
import html
import os.path
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET
import feedparser
from datastructures import Article, Corpus

def nettoyage(texte): # pas nettroyer tout le corpus mais la ligne extraite par chaque reader
    texte_net = re.sub(r"<!\[CDATA\[(.*?)\]\]>", "\\1", texte, flags=re.DOTALL)
    texte_net = html.unescape(texte_net)
    texte_net = re.sub("<.+?>", "", texte_net)
    texte_net = re.sub("\n+", " ", texte_net)
    texte_net = texte_net.strip()
    return texte_net


def rss_reader_re(filename: str | Path) -> Corpus:
    articles = []
    name = Path(filename).name
    global_categories = set()

    with open(filename, "r") as input_rss:
        texte = input_rss.read()

        # find global categories in header
        if (match := re.search("<channel>.+?<item>", texte, flags=re.DOTALL)) is not None:
            header = match.group(0)
            for submatch in re.finditer("<category.*>(.+?)</category>", header):
                global_categories.add(submatch.group(1))

        for match in re.finditer(r"<item>.*?</item>", texte, flags=re.DOTALL):
            item = match.group(0)

            title = re.search(r"<title.*?>(.+?)</title>", item).group(1)
            title = nettoyage(title)
            description = re.search(r"<description.*?>(.*?)</description>", item, flags=re.DOTALL).group(1)
            description = nettoyage(description)

            local_categories = global_categories.copy()
            if re.finditer(r"<category.*?>(.+?)</category>", item): # s'assurer qu'il y a les catégories dans les item
                for category in re.finditer(r"<category.*?>(.+?)</category>", item): 
                    local_categories.add(category.group(1))

            dataid = re.search(r"<guid.*?>(.+?)</guid>", item).group(1)

            pubdate_element = re.search(r"<pubDate.*?>(.+?)</pubDate>", item)
            if pubdate_element is not None:
                pubdate = nettoyage(pubdate_element.group(1))
            else:
                pubdate = ""
            # use Article class to create an article
            article = Article(
                id=dataid,
                source=name,
                title=title,
                description=description,
                date=pubdate,
                categories=sorted(local_categories),
            )
            articles.append(article)

    return Corpus(articles)

def rss_reader_etree(filename: str | Path) -> Corpus:
    name = Path(filename).name

    if name.lower() in ("flux.xml", "flux rss.xml"): # erreur de parsing
        return Corpus([])

    try:
        root = ET.parse(filename)
    except ET.ParseError:
        return Corpus([])

    articles = []
    global_categories = set(element.text.strip() for element in root.iterfind("./channel/category"))

    for item in root.iterfind(".//item"):
        dataid = item.find("guid").text

        # particularité etree : ne pas vérifier la valeur avec juste if, mais bien avec "is None" ou "is not None"
        # doc : https://docs.python.org/3/library/xml.etree.elementtree.html#xml.etree.ElementTree.Element.remove
        title_element = item.find("title")
        if title_element is not None:
            title = title_element.text or ""
        else:
            title = ""
        title = nettoyage(title)

        description_element = item.find("description")
        if description_element is not None:
            description = description_element.text or ""
        else:
            description = ""
        description = nettoyage(description)

        pubdate_element = item.find("pubDate")
        if pubdate_element is None:
            pubdate_element = item.find("lastpublished")
        if pubdate_element is not None:
            pubdate = pubdate_element.text
        else:
            pubdate = None
        if pubdate is not None:
            pubdate = nettoyage(pubdate)

        local_categories = global_categories.copy()
        for category_element in item.iterfind("category"):
            local_categories.add(category_element.text)

        article = Article(
            id=dataid,
            source=name,
            title=title,
            description=description,
            date=pubdate,
            categories=sorted(local_categories),
        )
        articles.append(article)
        
    return Corpus(articles)

def rss_reader_feedparser(filename: str | Path) -> Corpus:
    name = Path(filename).name
    feed = feedparser.parse(filename)
    articles = []
    global_categories = set([item["term"] for item in feed["feed"].get("tags", [])])
    for item in feed["entries"]:
        pubdate = item.get("published")
        if not pubdate:
            pubdate = item.get("lastpublicationdate")

        categories = global_categories | set(t["term"] for t in item.get("tags", []))

        article = Article(
            id=item.get("id", ""),
            source=name,
            title=item.get("title", ""),
            description=item.get("summary", ""),
            date=pubdate,
            categories=sorted(categories),
        )
        
        articles.append(article)

    return Corpus(articles)

def main():
    parser = argparse.ArgumentParser(description="Process RSS feeds.")
    parser.add_argument("filename", type=str, help="RSS feed file")
    parser.add_argument("-r","--reader", type=str, default="etree", help="reader (re, etree, feedparser)")
    args = parser.parse_args()

    if args.reader == "re":
        articles = rss_reader_re(args.filename)
    elif args.reader == "etree":
        articles = rss_reader_etree(args.filename)
    elif args.reader == "feedparser":
        articles = rss_reader_feedparser(args.filename)
    else:
        print(f"Error: unknown format '{args.reader}'")
        raise KeyError

    for article in articles.articles:
        print(f"ID: {article.id}")
        print(f"Source: {article.source}")
        print(f"Title: {article.title}")
        print(f"Description: {article.description}")
        print(f"Date: {article.date}")
        print(f"Categories: {', '.join(article.categories)}")
        print() 

name_to_reader = {
    "re": rss_reader_re,
    "etree": rss_reader_etree,
    "feedparser": rss_reader_feedparser
}

if __name__ == "__main__":
    main()