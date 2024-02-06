"""
Module for topic modelling functions. 
Functions starting with _ are not meant to be called directly but rather are helper functions for use inside the module
"""

import gensim
import nltk
from gensim.utils import simple_preprocess
from wordcloud import WordCloud

nltk.download('stopwords')
import gensim.corpora as corpora
import pandas as pd
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import tomotopy as tp
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from pyvis.network import Network
import os
import re
import warnings
warnings.filterwarnings("ignore")


def generate_wordcloud(tweet_list):
    # Join the different processed titles together.
    long_string = ','.join(list(tweet_list))

    # Create a WordCloud object
    wordcloud = WordCloud(background_color="white", max_words=4000, contour_width=3, contour_color='steelblue')

    # Generate a word cloud
    wordcloud.generate(long_string)

    # Visualize the word cloud
    return wordcloud.to_image()
    

def gensim_topic_modelling(tweet_list, num_topics=5, visualize = True, filepath=""):
    data_words = list(_sent_to_words(tweet_list))
    # remove stop words
    data_words = _remove_stopwords(data_words)
    id2word = corpora.Dictionary(data_words)

    # Create Corpus
    texts = data_words

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    
    # Build LDA model
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                        id2word=id2word,
                                        num_topics=num_topics)
    if visualize:
        # Visualize the topics
        pyLDAvis.enable_notebook()
        LDAvis_data_filepath = os.path.join(filepath+"_"+str(num_topics))
        LDAvis_prepared = gensimvis.prepare(lda_model, corpus, id2word)
        pyLDAvis.save_html(LDAvis_prepared, filepath+'_gensim_LDA.html')
        print("html saved to:", filepath+'_gensim_LDA.html') 
    
    return lda_model

def bertopic_topic_modelling(tweet_list, visualize=False, filepath=""):
    # Instantiate a representation model to improve representations of tokens in each topic
    representation_model = KeyBERTInspired()
    # Instantiate a tokenizer model
    vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words="english")
    # Instantiate BERTopic model
    topic_model = BERTopic(representation_model=representation_model, 
                           vectorizer_model=vectorizer_model,
                           language="english") 
    # Fine-tune topic representations
    topics, probs = topic_model.fit_transform(tweet_list)       
    return topic_model, topics, probs   

def tomotopy_LDA(tweet_list, num_topics, num_iterations, step_size=10, print_output = True, visualize=False, filepath=""):
    mdl = tp.LDAModel(k=num_topics)
    for tweet in tweet_list:
        mdl.add_doc(tweet.strip().split())
    for i in range(0, num_iterations, step_size):
        mdl.train(step_size)
        if print_output:
            print('Iteration: {}\tLog-likelihood: {}'.format(i, mdl.ll_per_word))
    if print_output:        
        for k in range(mdl.k):
            print('Top 10 words of topic #{}'.format(k))
            print(mdl.get_topic_words(k, top_n=10))
    if visualize:
        topic_term_dists = np.stack([mdl.get_topic_word_dist(k) for k in range(mdl.k)])
        doc_topic_dists = np.stack([doc.get_topic_dist() for doc in mdl.docs])
        doc_topic_dists /= doc_topic_dists.sum(axis=1, keepdims=True)
        doc_lengths = np.array([len(doc.words) for doc in mdl.docs])
        vocab = list(mdl.used_vocabs)
        term_frequency = mdl.used_vocab_freq

        prepared_data = pyLDAvis.prepare(
            topic_term_dists, 
            doc_topic_dists, 
            doc_lengths, 
            vocab, 
            term_frequency,
            start_index=0, # tomotopy starts topic ids with 0, pyLDAvis with 1
            sort_topics=False # IMPORTANT: otherwise the topic_ids between pyLDAvis and tomotopy are not matching!
        )
        pyLDAvis.save_html(prepared_data, file_path+'_tomotopyLDA_ldavis.html')     
        print("html saved to:", file_path+'_tomotopyLDA_ldavis.html')    
    return mdl    

def tomotopy_CTM(tweet_list, n_topics, min_df=5, rm_top=40):
    corpus = _tomotopy_corpus(tweet_list)
    mdl = tp.CTModel(min_df=min_df, rm_top=rm_top, k=n_topics, corpus=corpus)
    mdl.train(0)

    print('Num docs:{}, Num Vocabs:{}, Total Words:{}'.format(
        len(mdl.docs), len(mdl.used_vocabs), mdl.num_words
    ))
    print('Removed Top words: ', *mdl.removed_top_words)

    # train the model
    mdl.train(1000, show_progress=True)
    mdl.summary()

    # visualize the result
    g = Network(width=800, height=800, font_color="#333")
    correl = mdl.get_correlations().reshape([-1])
    correl.sort()
    top_tenth = mdl.k * (mdl.k - 1) // 10
    top_tenth = correl[-mdl.k - top_tenth]

    for k in range(mdl.k):
        label = "#{}".format(k)
        title= ' '.join(word for word, _ in mdl.get_topic_words(k, top_n=6))
        print('Topic', label, title)
        g.add_node(k, label=label, title=title, shape='ellipse')
        for l, correlation in zip(range(k - 1), mdl.get_correlations(k)):
            if correlation < top_tenth: continue
            g.add_edge(k, l, value=float(correlation), title='{:.02}'.format(correlation))

    g.barnes_hut(gravity=-1000, spring_length=20)
    g.show_buttons()
    g.show("topic_network.html")


def _tomotopy_corpus(tweet_list):
    porter_stemmer = nltk.PorterStemmer().stem
    english_stops = set(porter_stemmer(w) for w in stopwords.words('english'))
    pat = re.compile('^[a-z]{2,}$')
    corpus = tp.utils.Corpus(
        tokenizer=tp.utils.SimpleTokenizer(porter_stemmer), 
        stopwords=lambda x: x in english_stops or not pat.match(x)
    )
    corpus.process(tweet_list)  
    return corpus 
        
def _sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def _remove_stopwords(texts, stop_words_extenstion = ['from', 'subject', 're', 'edu', 'use']):
    stop_words = stopwords.words('english')
    stop_words.extend(stop_words_extenstion)
    return [[word for word in simple_preprocess(str(doc)) 
             if word not in stop_words] for doc in texts]
    
def _further_remove_stopwords(tweet, stopwords_extenstion = ['from', 'subject', 're', 'edu', 'use']):
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    #extend stop words
    stop_words |= set(stopwords_extenstion)
    # Tokenize the tweet into words
    words = word_tokenize(tweet)
    # Remove stopwords and punctuation
    filtered_words = [word.lower() for word in words if word.lower() not in stop_words and word not in string.punctuation]
    s.translate(None, string.punctuation)
    # Join the remaining words to form the processed tweet
    processed_tweet = ' '.join(filtered_words)
    return processed_tweet