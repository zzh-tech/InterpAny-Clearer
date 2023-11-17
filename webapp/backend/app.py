from flask import Flask, jsonify, request, send_from_directory
from json import *
import os
import subprocess
import shlex
from flask_cors import CORS, cross_origin
import uuid
from urllib.parse import urlparse

import numpy as np
from pprint import pprint
from statistics import median, mode
import time
import random
from embedding import Embedding
from interpolate_ours import Interpolate

# basedir = os.path.abspath(os.path.dirname(__file__))    
baseDir = os.path.abspath(os.path.dirname(__file__))   
uploadDir = f'{baseDir}/data/uploads'
modelDir = f'{baseDir}/data/models'
resultDir = f'{baseDir}/data/results'

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    response = {}
    response['isMusic'] = True
    response['tempo'] = 110
    response['loudness'] = 10
    return jsonify(response)

@app.route('/upload', methods = ['POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        fileName = f'{uuid.uuid4()}.png'
        localUrl = f'{uploadDir}/{fileName}'
        publicUrl = f'http://localhost:5001/uploads/{fileName}'
        f.save(localUrl)
        return jsonify({"url": publicUrl, 'filename': fileName})
    
@app.route('/upload_mask_dict', methods = ['POST'])
def upload_mask_dict():
    if request.method == 'POST':
        f = request.files['file']
        fileName = f'{uuid.uuid4()}.txt'
        localUrl = f'{uploadDir}/{fileName}'
        f.save(localUrl)
        json_dict_path = localUrl

        # do actual interpolate
        interpolate.interpolate(json_dict_path=json_dict_path, iters=3)
        fileName = f'{uuid.uuid4()}.mp4'
        subprocess.run(shlex.split(f'ffmpeg -y -r 10 -f image2 -i data/output_recur/img%d.png -c:v libx264 -pix_fmt yuv420p data/results/{fileName} -q:v 0 -q:a 0'))
        
        # return response
        response = {}
        response['video_url'] = f'http://localhost:5001/results/{fileName}'
        return jsonify(response)
    
@app.route('/embedding', methods = ['POST'])
def get_embedding():
    if request.method == 'POST':
        content = request.json
        fileName = content.get('filename')
        embeddingUrl = embedding.generateEmbedding(fileName)
        # embeddingFile = "f82cbaf6-4800-4158-adbf-7cb4ddf9d7b6"
        # embeddingUrl = f'http://localhost:5001/embeddings/{embeddingFile}.npy'
        response = {}
        response['embedding_url'] = embeddingUrl
        return jsonify(response)

@app.route('/interpolate', methods = ['POST'])
def get_interpolate():
    if request.method == 'POST':
        content = request.json
        # fileName1 = urlparse(content.get('filename1'))[2]
        # fileName2 = urlparse(content.get('filename2'))[2]
        # files = interpolate.interpolate(f'data{fileName1}', f'data{fileName2}')
        # subprocess.run(shlex.split('ffmpeg -y -r 10 -f image2 -i data/output/img%d.png -s 448x256 -c:v libx264 -pix_fmt yuv420p data/results/slomo.mp4 -q:v 0 -q:a 0'))
        json_dict_path = 'data/mask_dict/masks_dict.txt'
        video_path = interpolate.interpolate(json_dict_path=json_dict_path, iters=3)
        subprocess.run(shlex.split('ffmpeg -y -r 10 -f image2 -i data/output_recur/img%d.png -s 448x256 -c:v libx264 -pix_fmt yuv420p data/results/slomo.mp4 -q:v 0 -q:a 0'))

        response = {}
        response['video_url'] = 'http://localhost:5001/results/slomo.mp4'
        return jsonify(response)
    
@app.route('/uploads/<path:path>')
def send_uploads(path):
    return send_from_directory('data/uploads', path)

@app.route('/embeddings/<path:path>')
def send_embeddings(path):
    return send_from_directory('data/embeddings', path)

@app.route('/results/<path:path>')
def send_results(path):
    return send_from_directory('data/results', path)

if __name__ == '__main__':
    print('Loading resources...')
    embedding = Embedding(resultDir, modelDir)
    interpolate = Interpolate()
    print('Resources loaded.')
    print('Running server...')
    app.run(port=5001, host='0.0.0.0', debug=True)
