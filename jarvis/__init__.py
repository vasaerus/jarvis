import os
import sqlite3
import openai
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, Trainer, TrainingArguments
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.corpus import wordnet
import speech_recognition as sr
import pyttsx3
import datetime
import subprocess
import psutil
import requests
import json
from flask import Flask, request, jsonify, render_template

def init():
    # Set up OpenAI API key
    openai.api_key = "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"

    # Set up SQLite database
    conn = sqlite3.connect("jarvis.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS conversations (id INTEGER PRIMARY KEY, prompt TEXT, response TEXT, image_file TEXT, audio_file TEXT, video_file TEXT, date TEXT, trained INT DEFAULT 0)''')
    conn.commit()
    conn.close()

    # Set up GPT-2 model
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    if os.path.isfile("jarvis.h5"):
        model = GPT2LMHeadModel.from_pretrained("jarvis.h5", pad_token_id=tokenizer.eos_token_id)
    else:
        model = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

    return tokenizer, model
