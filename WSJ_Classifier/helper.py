# -*- coding: utf-8 -*-
"""Module containing helper functions.

Module contains functions for saving and loading data in and from .pkl files
and a function to download news article summaries together with their news
category from the Wall Street Journal for a specified time range.

"""

from datetime import timedelta
import pickle
from bs4 import BeautifulSoup
import requests
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.rcdefaults()


def save_obj(obj, name):
    """Saves an object in a .pkl file.

    Args:
        obj:        Any python object that you want to save.
        name (str): Path + file name where you want to save your file.

    Use file ending .pkl.

    """
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    """Load an object from a .pkl file.

    Args:
        name (str): Path + file name of the file.

    """
    with open(name, 'rb') as f:
        return pickle.load(f)

def daterange(start_date, end_date):
    """Date range generator.

    From https://stackoverflow.com/questions/1060279/iterating-through-a-range-of-dates-in-python.

    Args:
        start_date (date): Starting date of the date range.
        end_date (date):   Last date of the date range.

    Yields:
        The next date in the range start_date to end_date.

    """
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)

def get_wsj(start_date, end_date):
    """Download news article summaries and belonging news categories.

    Download news article summaries and belonging news categories from
    the website of the Wall Street Journal for a given date range.

    Args:
        start_date (date): First date for which the article summaries
                           and belonging categories are downloaded.
        end_date (date):   Last date in the date range for which the
                           article summaries and belonging categories
                           are downloaded.

    Returns:
        news_articles (list): List of all downloaded news arcticle
                              summaries.
        sections (list):      List of the categories belonging to
                              the news article summaries.

    """
    news_articles = []
    sections = []
    n = 0
    # iterate over each day of the given state range
    for single_date in tqdm(daterange(start_date, end_date)):
        #articles_of_day = []
        date_str = single_date.strftime("%Y-%m-%d")
        # construct url
        url = 'http://www.wsj.com/public/page/archive-' + date_str + '.html'
        r = requests.get(url)
        r.raise_for_status()

        # parse html string
        soup = BeautifulSoup(r.text, 'html.parser')
        articles = soup.find(id="archivedArticles").find_all('p')

        # iterate over all article summaries of the day
        for article in articles:
            news_articles.append(article.text.strip())
            n += 1

        list_of_articles = soup.find("ul", {"class": "newsItem"}).find_all("li")
        links_to_articles = []

        # extract links to the whole articles because
        # only the website with the whole article includes
        # its section/categorie
        for item in list_of_articles:
            links_to_articles.append(item.find(href=True)['href'])

        # iterate over each link, open the website and extract the
        # section / categorie that belongs to the article
        for link in links_to_articles:
            r = requests.get(link)

            # if a link can't be opened use section <UNKNOWN>
            if r.status_code != 200:
                sections.append("<ERROR>")

            # if it can be opened, exctract the section name
            else:
                soup = BeautifulSoup(r.text, 'html.parser')
                sections.append(soup.find("meta", {"name": "article.section"})["content"])

    print("Downloaded {} articles from Wall Street Journal.".format(n))
    return news_articles, sections

def plot_training_history(history, model_name="model"):
    # Plot training and validation accuracy values
    plt.subplot(121)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Prediction accuracy of {}'.format(model_name))
    plt.ylabel('Accuracy')
    plt.xlabel('Training Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    #plt.show()
    
    # Plot training and validation loss values
    plt.subplot(122)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Prediction loss of {}'.format(model_name))
    plt.ylabel('Loss')
    plt.xlabel('Training Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.tight_layout()
    plt.show()
