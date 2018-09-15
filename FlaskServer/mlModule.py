import pandas as pd
import numpy as np
import functions as f
from keras.models import model_from_json
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
import pickle

def preprocessing():
	df = f.fetch_data()
	df = f.preprocess_data(df)
	maxlength = 50
	max_features = 2000
	X, Y = f.tokenize_data(df, maxlength, max_features)
	X = pad_sequences(X, maxlen = maxlength)
	X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 222)
	return X_train, X_test, Y_train, Y_test

def model_building(X_train, Y_train):
	embed_dim = 128
	lstm_out = 196
	X_shape = X_train.shape[1]
	Y_shape = Y_train.shape[1]
	max_features = 2000
	model = f.generate_model(max_features,embed_dim,lstm_out,X_shape, Y_shape)
	return model
    
def training(model, X_train, Y_train, batch_size):
	model.fit(X_train, Y_train, epochs = 5, batch_size = batch_size, verbose = 2)
	return model
	
def testing(model, X_test, Y_test, batch_size):
	score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
	return score
	
def preprocess_user_review(review):
	review = np.array([review])
	review = np.char.lower(review)
	remove_unwanted_characters = lambda x: re.sub('[^a-zA-Z0-9\s]','',x)
	func = np.vectorize(remove_unwanted_characters)
	review = func(review)
	filehandler = open('data/tokenizer.obj','rb')
	tokenizer = pickle.load(filehandler)
	filehandler.close()
	review_processed = tokenizer.text_to_sequences(review)
	review_processed_padded = pad_sequences(review_processed, maxlen = 50)
	return review_processed_padded
	
def predict(review):
	review = preprocess_user_review(review)
	model_json_file_name = "model/model.json"
	model_weight_file_name = "model/model_weights.h5"
	model = load_model(model_json_file_name, model_weight_file_name)
	predicted_class = model.predict(review)
	return predicted_class

def save_model(model, model_json_file_name, model_weight_file_name):
	model_json = model.to_json()
	with open(model_json_file_name, "w") as json_file:
		json_file.write(model_json)
	json_file.close()
	model.save_weights(model_weight_file_name)
	return

def load_model(model_json_file_name, model_weight_file_name):
	json_file = open(model_json_name, 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights(model_weight_file_name)
	return loaded_model
