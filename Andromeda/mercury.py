import os
import threading
import sqlite3
import pickle
import numpy as np
from flask import Flask, request, jsonify

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from torch.utils.data import Dataset, DataLoader

from tensorflow import keras

from keras.layers import Embedding, Bidirectional, LSTM, Dense
from keras.models import Sequential

from keras.preprocessing.text import Tokenizer
from keras.models import Sequential

import openai


from gensim.models import Word2Vec
import logging
import gensim

from transformers import GPT2Tokenizer, GPT2Model

from transformers import GPT2LMHeadModel

from diffusers import StableDiffusionPipeline