import pandas as pd
import numpy as np
import re
import pickle
from keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import model_from_json
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D

def fetch_data():
	df = pd.read_csv('data/train.tsv',sep='\t', header=0)
	return df
	
def preprocess_data(df):
	df = df.drop(columns = ['SentenceId','PhraseId'])
	df.loc[df.Sentiment == 4, 'Sentiment'] = 3
	df.loc[df.Sentiment == 0, 'Sentiment'] = 1
	df['Phrase'] = df['Phrase'].apply(lambda x: x.lower())
	df['Phrase'] = df['Phrase'].apply(lambda x: re.sub('[^a-zA-Z0-9\s]','',x))
	return df

def tokenize_data(df, maxlength, max_features):
	tokenizer = Tokenizer(num_words = max_features, split=' ')
	tokenizer.fit_on_texts(df['Phrase'].values)
	max_features = 2000
	filehandler = open('data/tokenizer.obj','wb')
	pickle.dump(tokenizer, filehandler)
	filehandler.close()
	X = tokenizer.texts_to_sequences(df['Phrase'].values)
	Y = pd.get_dummies(df['Sentiment']).values
	return X,Y

def generate_model(max_features, embed_dim, lstm_out, X_shape, Y_shape):
	model = Sequential()
	model.add(Embedding(max_features, embed_dim, input_length = X_shape))
	model.add(SpatialDropout1D(0.4))
	model.add(LSTM(lstm_out, dropout = 0.2, recurrent_dropout = 0.2 ))
	model.add(Dense(Y_shape, activation = 'softmax'))
	model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	return model
    
