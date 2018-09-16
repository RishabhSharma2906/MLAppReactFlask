from flask import Flask, redirect, url_for, request, jsonify
import mlModule as ml

import numpy as np
import re
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

app = Flask(__name__)

@app.route('/home')
def welcome_here():
	return jsonify({"message" : "Welcome Here"}), 200

@app.route('/result', methods = ['POST','GET'])
def giveResult():
	if request.method == 'POST':
		review = request.form['review']
		predicted_class = ml.predict(review, model, tokenizer)
		result = jsonify({'class' : predicted_class}), 200
	else:
		result = jsonify({'Response' : 'No other request except POST requests are accepted on this server'}), 404
	return result
		
if __name__ == '__main__':
	model_json_file_name = "model/model.json"
	model_weight_file_name = "model/model_weights.h5"
	model = ml.load_model(model_json_file_name, model_weight_file_name)
	#model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	filehandler = open('data/tokenizer.obj', 'rb')
	tokenizer = pickle.load(filehandler)
	filehandler.close()
	model2 = load_model('model/first_model.h5')
	
	print('------------------------------------------------Temp--------------------------------------------')
	review = 'ok movie'
	review = np.array([review])
	review = np.char.lower(review)
	remove_unwanted_characters = lambda x: re.sub('[^a-zA-Z0-9\s]','',x)
	func = np.vectorize(remove_unwanted_characters)
	abcd = func(review)
	op = tokenizer.texts_to_sequences(abcd)
	X_new = pad_sequences(op, maxlen = 50)
	ynew = model.predict_classes(X_new)
	#print(ynew[0])
	#filehandler.close()
	print('------------------------------------------------Temp--------------------------------------------')
	
	app.run(debug = True)
	
