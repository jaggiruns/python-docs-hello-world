import sys
from flask import Flask, jsonify, request, make_response, abort
import os
import nltk
#import numpy as np
#import pandas as pd
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import time
import logging
import pickle
import re
# Use with Azure Web Apps
#os.environ['PATH'] = r'D:\home\python354x64;' + os.environ['PATH']
#sys.path.append(".")
#sys.path.append("..")
#sys.path.append("webservice/models")
#sys.path.append("wwwroot/models")
app = Flask(__name__)
__location__ = os.path.realpath(os.path.join(
    os.getcwd(), os.path.dirname(__file__), 'models'))

# Download models
#from __future__ import print_function
try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve


def download_file(file_url, folder_path, file_name):
    file_path = os.path.join(folder_path, file_name)
    if not os.path.exists(file_path):
        print('Downloading file from ' + file_url + '...')
        urlretrieve(file_url, file_path)
        print('Done downloading file: ' + file_path)
    else:
        print('File: ' + file_path + ' already exists.')


def download_models():
    print('Downloading models for web service...')
    file_list = ['category.model', 'impact.model', 'ticket_type.model']
    folder_path = os.path.dirname(os.path.abspath(__file__))
    url = "https://privdatastorage.blob.core.windows.net/github/support-tickets-classification/models/"
    for file_name in file_list:
        download_file(url + file_name, folder_path, file_name)


#from models.download_models import download_file, download_models
#download_models()

# Loading models
#model_impact = pickle.load(
    #open(
       # os.path.join(__location__, "impact.model"), "rb"
    #)
#)
#model_ticket_type = pickle.load(
    #open(
        #os.path.join(__location__, "ticket_type.model"), "rb"
    #)
#)
#model_category = pickle.load(
    #open(
       # os.path.join(__location__, "category.model"), "rb"
    #)
#)


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


@app.route('/')
def index():
    return """
        <html>
        <body>
        Hello, World!<br>
        This is a sample web service written in Python using <a href=""http://flask.pocoo.org/"">Flask</a> module.<br>
        </body>
        </html>
        """
