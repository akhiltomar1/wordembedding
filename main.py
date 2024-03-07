from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential

sentences=[  'the glass of milk',
            'the glass of juice',
            'the cup of tea',
            'I am a good boy',
            'I am a good developer',
            'understand the meaning of words',
            'your videos are good',]

vocabulary_size = 10000;

#onehot creates the matrix represtation of the words woth a weighted value

onehot_representaion = [ one_hot(words,vocabulary_size)for words in sentences]
print(onehot_representaion)


#pad_Sequuences gives padding to matrix in case of different number of words in a sentence (pre and post - padding)
sentence_length=8
embedded_docs=pad_sequences(onehot_representaion,padding='pre',maxlen=sentence_length)
print(embedded_docs)

dimension = 10

model=Sequential()
model.add(Embedding(vocabulary_size,10,input_length=sentence_length))
model.compile('adam','mse')

model.summary()

print(model.predict(embedded_docs))

print(embedded_docs[0])

print(model.predict(embedded_docs)[0])