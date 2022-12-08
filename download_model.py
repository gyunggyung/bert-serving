import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text

encoder = hub.KerasLayer("https://tfhub.dev/jeongukjae/distilkobert_sentence_encoder/1")
preprocessor = hub.KerasLayer("https://tfhub.dev/jeongukjae/distilkobert_cased_preprocess/1")

inputs = tf.keras.Input([], dtype=tf.string)
encoder_inputs = preprocessor(inputs)
sentence_embedding = encoder(encoder_inputs)
normalized_sentence_embedding = tf.nn.l2_normalize(sentence_embedding, axis=-1)
model = tf.keras.Model(inputs, normalized_sentence_embedding)

model.save("model/distilkobert_sentence_encoder")
