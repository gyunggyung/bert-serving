import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
from flask import Flask, request, render_template

# Load required models
encoder = hub.KerasLayer("https://tfhub.dev/jeongukjae/distilkobert_sentence_encoder/1")
preprocessor = hub.KerasLayer("https://tfhub.dev/jeongukjae/distilkobert_cased_preprocess/1")

# Define sentence encoder model
inputs = tf.keras.Input([], dtype=tf.string)
encoder_inputs = preprocessor(inputs)
sentence_embedding = encoder(encoder_inputs)
normalized_sentence_embedding = tf.nn.l2_normalize(sentence_embedding, axis=-1)
model = tf.keras.Model(inputs, normalized_sentence_embedding)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/classify', methods=['POST'])
def classify():
    sentence1 = tf.constant([request.form["sentence1"]])
    sentence2 = tf.constant([request.form["sentence2"]])
    embeddings1 = model(sentence1)
    embeddings2 = model(sentence2)
    score = tf.tensordot(embeddings1, embeddings2, axes=[1, 1]).numpy()[0][0]
    return "<h3>두 문장의 유사도는: {}%입니다!</h3".format(round(score * 100, 2))

if __name__ == '__main__':
    app.run(host="0.0.0.0", port="8000", debug=True)
