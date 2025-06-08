"""
RSS Pipeline Script

This script integrates functionality from rss_reader.py, rss_parcourir.py, analyzers.py, and BERTopic
to create a complete pipeline for processing RSS feeds, analyzing them with NLP tools,
and performing topic modeling.

Usage:
    python rss_pipeline.py /path/to/rss_folder --reader etree --walker pathlib 
                          [--start 01/05/25] [--categories cat1 cat2] [--source source_name]
                          [--analyzer stanza] [--n_topics 20] [--n_components 8] [--min_dist 0.05]
                          [--min_cluster_size 5] [--min_samples 5] [--embeddings_path embeddings.npy]
                          [--save_embeddings] [-p --pos] [--format] [--output output_folder]
Ex: python pipeline.py path/to/rss_folder --reader etree --walker pathlib --start 01/05/25 --categories finance crypto --source "BFM CRYPTO" --analyzer spacy --n_topics 15 --n_components 10 --min_dist 0.1 --min_cluster_size 8 --min_samples 8 --save_embeddings --pos NOUN --format json --output ./complete_results
"""

import argparse
from pathlib import Path
import time
import copy
import xml
import pickle
import numpy as np
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
import os


# Import functionality from existing scripts
from rss_reader import rss_reader_re, rss_reader_etree, rss_reader_feedparser, name_to_reader
from rss_parcourir import (
    walk_os, walk_pathlib, walk_glob, name_to_walker,
    create_filter_start_date, create_filter_categories, create_filter_source,
    build_filters, filtrage
)
from datastructures import Article, Corpus, save_json, save_pickle, save_xml, name_to_save
from analyzers import parsers, models

# Try importing BERTopic-related modules
try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from umap import UMAP
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from bertopic.vectorizers import ClassTfidfTransformer
    HAS_BERTOPIC = True # if the importing has successed, we mark with true
except ImportError:
    print("Warning: BERTopic or its dependencies not imported. Model will be disabled.")
    HAS_BERTOPIC = False


def setup_argparse() -> argparse.ArgumentParser:
    """
    Configure and return the argument parser.
    """
    parser = argparse.ArgumentParser(
        description="RSS Pipeline: Read, filter, analyze, and topic modeling RSS feeds"
    )
    
    # RSS inputs
    parser.add_argument("rss_feed", help="Path to RSS feed file or directory")
    parser.add_argument(
        "-r", "--reader", 
        choices=("re", "etree", "feedparser"),
        default="etree", 
        help="RSS reader to use (default: etree)"
    )
    parser.add_argument(
        "-w", "--walker", 
        choices=("os", "pathlib", "glob"),
        default="pathlib", 
        help="Directory walker to use (default: pathlib)"
    )
    
    # Filtering arguments
    parser.add_argument(
        "-s", "--start", 
        help="Keep articles from this date onward (format: DD/MM/YY)"
    )
    parser.add_argument(
        "-c", "--categories", 
        nargs="*", 
        default=[],
        help="Filter by categories"
    )
    parser.add_argument(
        "--source", 
        help="Filter by source name"
    )
    
    # Analysis arguments
    parser.add_argument(
        "-a", "--analyzer", 
        choices=("stanza", "spacy", "trankit"),
        default="spacy", 
        help="NLP analyzer to use (default: spacy)"
    )
    
    # Topic modeling arguments
    if HAS_BERTOPIC:
        parser.add_argument(
            "--n_topics", 
            type=int, 
            default=20,
            help="Number of topics for BERTopic (default: 20)"
        )
        parser.add_argument(
            "--no_topic_modeling", 
            action="store_true",
            help="Only processe RSS files while skip topic modeling step"
        )
        parser.add_argument(
            "--n_components", 
            type=int, 
            default=8,
            help="Number of components for UMAP dimensionality reduction (default: 8)"
        )
        parser.add_argument(
            "--min_dist", 
            type=float, 
            default=0.05,
            help="Minimum distance between points for UMAP (default: 0.05)"
        )
        parser.add_argument(
            "--min_cluster_size", 
            type=int, 
            default=5,
            help="Minimum cluster size for HDBSCAN (default: 5)"
        )
        parser.add_argument(
            "--min_samples", 
            type=int, 
            default=5,
            help="Minimum samples for HDBSCAN (default: 5)"
        )
        parser.add_argument(
        "--embeddings_path", 
        type=str,
        default=None,
        help="Path to pre-computed embeddings npy file"
        )
        parser.add_argument(
            "--save_embeddings",
            action="store_true",
            help="Save computed embeddings for future use"
        )




    # Pos option
    parser.add_argument(
        "-p", "--pos",
        default=None,
        help="Only keep given pos words during preprocessing"
    )

    # Output arguments
    parser.add_argument(
        "-o", "--output", 
        default="BERTopic_model"
    )
    parser.add_argument(
        "--format", 
        choices=("json", "pickle", "xml"), 
        default="json",
        help="Format of output (default: json)"
    )
    
    return parser


def prepare_output_directory(output_dir) -> Path:
    """
    Create the output directory structure: idealy four directories
    Corpus(extracted data) | Anylyzed(NLP enriched data) | topic_model | visualization
    """
    output_path = Path(output_dir)
    
    # Create main output directory
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Create subdirectories
    (output_path / "corpus").mkdir(exist_ok=True)
    (output_path / "analyzed").mkdir(exist_ok=True)
    
    if HAS_BERTOPIC:
        (output_path / "topic_model").mkdir(exist_ok=True)
        (output_path / "visualizations").mkdir(exist_ok=True)
    
    return output_path


def read_rss_files(args) -> Corpus:
    """
    Read RSS files with specified walker and reader 
    By returning a raw corpus of all extracted articles
    Article has a strucure of
    Id | source | title | description | data | categories | analysis
    """
    print(f"[1/4] Reading RSS files from {args.rss_feed}...")
    
    # Get appropriate walker and reader functions
    walker = name_to_walker.get(args.walker)
    
    reader = name_to_reader.get(args.reader)
    
    # Find all RSS files' path
    files = walker(args.rss_feed)
    print(f"Found {len(files)} RSS files")
    
    # Read all articles from RSS files
    articles = []
    for feed in files:
        try:
            feed_articles = reader(feed).articles
            articles.extend(feed_articles)
            print(f"Read {len(feed_articles)} articles from {Path(feed).name}")
        except Exception:
            print(f"Error reading {Path(feed).name}: {Exception}")
    
    # Create corpus from articles
    corpus = Corpus(articles)
    print(f"Total articles read: {len(corpus.articles)}")
    
    return corpus


def filter_corpus(corpus, args) -> Corpus:
    """
    Filter the corpus based on source, category and published date
    """
    print("[2/4] Filtering articles...")
    
    filtres = build_filters(args)
    
    if not filtres:
        print("No filters applied")
        return corpus
    
    # Apply filters
    filtered_articles = Corpus(filtrage(filtres, corpus.articles))
    
    print(f"Articles after filtering: {len(filtered_articles.articles)}")
    return filtered_articles


def analyze_corpus(corpus, args) -> Corpus:
    """
    Analyze the corpus using given NLP tools: Spacy or Stanza or Trankit
    In the scipt Analyzer we have splited NLP models and active functions 
    Because once the model are imported, python will load related processors while for some tools like Stanza 
    It takes time. So we choose to import only the given model while not importing them in the begining
    And pass the model to its active function
    """
    print(f"[3/4] Analyzing articles with {args.analyzer}...")
    
    # Get analyzer function and model
    analyzer_func = parsers.get(args.analyzer)
    if analyzer_func is None:
        raise ValueError(f"Invalid analyzer: {args.analyzer}")
    
    analyzer_model = models.get(args.analyzer)
    if analyzer_model is None:
        print(f"Loading {args.analyzer} model...")
        if args.analyzer == "stanza":
            from analyzers import load_stanza
            analyzer_model = load_stanza()
        elif args.analyzer == "trankit":
            from trankit import Pipeline
            analyzer_model = Pipeline(lang='french', gpu=False)
    
    # Analyze articles
    analyzed_articles = []
    total = len(corpus.articles)
    
    if args.analyzer == "spacy":
        from analyzers import batch_analyze_with_spacy
        analyzed_articles = batch_analyze_with_spacy(corpus.articles, analyzer_model)
    else:
        for i, article in enumerate(corpus.articles):
            if i % 1000 == 0 or i == total - 1:
                print(f"Analyzing article {i+1}/{total}")
            
            try:
                analyzed = analyzer_func(article, analyzer_model)
                analyzed_articles.append(analyzed)
            except Exception as e:
                print(f"Error analyzing article {article.id}: {e}")
    
    analyzed_corpus = Corpus(analyzed_articles)
    print(f"Articles successfully analyzed: {len(analyzed_corpus.articles)}")
    
    return analyzed_corpus


def train_topic_model(corpus, args, pos:str=None):
    """
    Train a BERTopic model on the corpus.
    Supports using pre-computed embeddings in .npy format for faster processing.
    """
    if not HAS_BERTOPIC or args.no_topic_modeling:
        return None, None
    
    print("[4/4] Training topic model...")
    
    if not corpus.articles:
        print(" No documents to model")
        return None, None
    
    # Prepare data for topic modeling
    if pos:
        docs = []
        for article in corpus.articles:
            text = [word for analysis in article.analysis for word in analysis if word['pos'].lower() == pos.lower()]
            text = " ".join(text)
            docs.append(text)
    else:
        docs = [f"{article.title} {article.description}".lower() for article in corpus.articles]
    
    print(f"Processing {len(docs)} documents for topic modeling")
    
    # Import numpy for embeddings handling
    import numpy as np
    
    # Check if we should use pre-computed embeddings
    embeddings = None
    if args.embeddings_path:
        try:
            print(f"Loading pre-computed embeddings from {args.embeddings_path}")
            embeddings = np.load(args.embeddings_path)
            
            # Check if the embeddings match our current dataset
            if len(embeddings) == len(docs):
                print(f"Loaded embeddings for {len(docs)} documents")
            else:
                print(f"Warning: Embeddings count mismatch. Expected {len(docs)}, found {len(embeddings)}.")
                print("Computing new embeddings...")
                embeddings = None
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            print("Computing new embeddings...")
            embeddings = None
    
    # Initialize models
    print("Initializing models...")
    embedding_model = SentenceTransformer('dangvantuan/sentence-camembert-base')
    
    umap_model = UMAP(
        n_neighbors=20,
        n_components=args.n_components,
        min_dist=args.min_dist,
        metric='cosine',
        random_state=42,
        low_memory=False
    )
    
    hdbscan_model = HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        cluster_selection_method="eom",
        metric="euclidean",
        gen_min_span_tree=True,
        algorithm='best',
        core_dist_n_jobs=-1
    )
    
    # Load stopwords
    try:
        with open("./stopwords-fr.txt") as f:
            stopwords = f.read().split()
    except FileNotFoundError:
        print("Warning: stopwords-fr.txt not found, using empty stopwords list")
        stopwords = []
    
    vectorizer_model = CountVectorizer(stop_words=stopwords)
    ctfidf_model = ClassTfidfTransformer()
    
    # Create BERTopic model
    topic_model = BERTopic(
        n_gram_range=(1, 2),
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        top_n_words=10,
        calculate_probabilities=False,
        verbose=True
    )
    
    # Generate embeddings if not provided
    if embeddings is None:
        print("Generating document embeddings...")
        batch_size = 64
        embeddings_list = []
        
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i+batch_size]
            batch_embeddings = embedding_model.encode(
                batch, 
                show_progress_bar=True,
                batch_size=32
            )
            embeddings_list.extend(batch_embeddings)
            print(f"Encoded {min(i+batch_size, len(docs))}/{len(docs)} documents")
        
        embeddings = np.array(embeddings_list)
        
        # Save embeddings if requested
        if args.save_embeddings:
            embeddings_output = Path(args.output) / "topic_model" / "embeddings.npy"
            print(f"Saving embeddings to {embeddings_output}")
            np.save(embeddings_output, embeddings)
            
            # Also save document count metadata in a separate file
            metadata_output = Path(args.output) / "topic_model" / "embeddings_metadata.txt"
            with open(metadata_output, 'w') as f:
                f.write(f"doc_count: {len(docs)}\n")
                f.write(f"timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"embedding_dimensions: {embeddings.shape[1]}\n")
    
    # Fit the model
    print("Fitting topic model...")
    topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)
    
    # Reduce topics if needed
    if args.n_topics < len(topic_model.get_topic_info()):
        print(f"Reducing to {args.n_topics} topics...")
        topic_model = topic_model.reduce_topics(docs, nr_topics=args.n_topics)
    
    topic_info = topic_model.get_topic_info()
    print(f"Found {len(topic_info)-1} topics (excluding outliers)")
    
    return topic_model, docs


def evaluate_topic_coherence(topic_model, docs, top_n=10):
    """
    评估主题一致性（coherence），返回每个主题的coherence分数和平均分。
    """
    # 1. 获取每个主题的前top_n个关键词
    topics = topic_model.get_topics()
    topic_words = []
    for topic in topics.values():
        words = [word for word, _ in topic[:top_n]]
        topic_words.append(words)
    # 2. 文档分词
    tokenized_docs = [doc.split() for doc in docs]
    # 3. 构建gensim字典和语料
    dictionary = Dictionary(tokenized_docs)
    corpus = [dictionary.doc2bow(text) for text in tokenized_docs]
    # 4. 计算coherence
    cm = CoherenceModel(
        topics=topic_words,
        texts=tokenized_docs,
        corpus=corpus,
        dictionary=dictionary,
        coherence='c_v'  # 也可以用'umass'
    )
    coherence_per_topic = cm.get_coherence_per_topic()
    mean_coherence = cm.get_coherence()
    print(f"主题一致性均值: {mean_coherence:.4f}")
    for idx, score in enumerate(coherence_per_topic):
        print(f"主题{idx}一致性: {score:.4f}")
    return mean_coherence, coherence_per_topic


def save_results(corpus, analyzed_corpus, topic_model, output_dir, format, docs=None):
    """
    Save all results to the output directory.
    """
    print("Saving results...")
    output_path = Path(output_dir)
    
    # Save original corpus
    corpus_path = output_path / "corpus" / f"corpus.{format}"
    save_func = name_to_save.get(format)
    save_func(corpus, corpus_path)
    print(f"Saved corpus to {corpus_path}")
    
    # Save analyzed corpus
    analyzed_path = output_path / "analyzed" / f"analyzed_corpus.{format}"
    save_func(analyzed_corpus, analyzed_path)
    print(f"Saved analyzed corpus to {analyzed_path}")
    
    # Save topic model and visualizations
    if topic_model is not None:
        # Save model
        model_path = output_path / "topic_model" / "bertopic_model"
        topic_model.save(str(model_path))
        print(f"Saved topic model to {model_path}")
        
        # Save topic info as JSON
        topic_info = topic_model.get_topic_info()
        topic_info_path = output_path / "topic_model" / "topic_info.json"
        topic_info.to_json(topic_info_path, orient="records", indent=2)
        
        # Generate and save visualizations
        viz_path = output_path / "visualizations"
        try:
            # Topic visualization
            fig1 = topic_model.visualize_topics()
            fig1.write_html(str(viz_path / "topics.html"))
            
            # Try to generate hierarchical topics visualization
            if docs:
                try:
                    hierarchical_topics = topic_model.hierarchical_topics(docs)
                    fig2 = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
                    fig2.write_html(str(viz_path / "hierarchy.html"))
                except Exception as e:
                    print(f"Warning: Could not generate hierarchical visualization: {e}")
            
            # Barchart visualization
            fig3 = topic_model.visualize_barchart()
            fig3.write_html(str(viz_path / "barchart.html"))
            
            print(f"Saved visualizations to {viz_path}")
        except Exception as e:
            print(f"Warning: Could not generate visualizations: {e}")

        # 主题一致性评估
        if docs is not None:
            evaluate_topic_coherence(topic_model, docs)


def main():
    """
    Main pipeline function.
    """
    # Parse arguments
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Prepare output directory
    output_dir = prepare_output_directory(args.output)
    
    # Process RSS files
    start_time = time.time()
    
    # Step 1: Read RSS files
    corpus = read_rss_files(args)
    # extracted_corpus = corpus.copy() this is not good, its surface copying
    # we can not make sure if the extracted_corpus will be change or not in further processing
    extracted_corpus = copy.deepcopy(corpus)
    
    # Step 2: Filter corpus
    filtered_corpus = filter_corpus(corpus, args)
    
    # Step 3: Analyze corpus
    analyzed_corpus = analyze_corpus(filtered_corpus, args)
    
    # Step 4: Topic modeling
    topic_model = None
    docs = None
    if HAS_BERTOPIC and not getattr(args, 'no_topic_modeling', False): # if user wants topic modeling
        # Prepare docs for topic modeling
        topic_model, docs = train_topic_model(analyzed_corpus, args, args.pos)
    
    # Save results
    save_results(extracted_corpus, analyzed_corpus, topic_model, args.output, args.format, docs)
    
    computed_time = time.time() - start_time
    print(f"Pipeline completed in {computed_time:.2f} seconds") # :.2f formatting spefication for floating-point numbers. 2 means precision
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    main()
