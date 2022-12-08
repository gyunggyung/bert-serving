import tensorflow as tf
import tensorflow_text
from flask import Flask, request, render_template
from cachelib  import SimpleCache

model = tf.saved_model.load("model/distilkobert_sentence_encoder")
cache = SimpleCache()
app = Flask(__name__)

def find_cache(key):
    if cache.get(key) is None:
        sentence = tf.constant([key])
        cache.set(key, model(sentence), timeout=3000)
    return cache.get(key)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/classify', methods=['POST'])
def classify():
    sentence1 = request.form["sentence1"]
    sentence2 = request.form["sentence2"]
    embeddings1 = find_cache(sentence1)
    embeddings2 = find_cache(sentence2)
    score = tf.tensordot(embeddings1, embeddings2, axes=[1, 1]).numpy()[0][0]
    return "<h3>두 문장의 유사도는: {}%입니다!</h3".format(round(score * 100, 2))

if __name__ == '__main__':
    app.run(host="0.0.0.0", port="8000", debug=True)
