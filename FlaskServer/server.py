from flask import Flask, redirect, url_for, request, jsonify
from mlModule import predict

app = Flask(__name__)

@app.route('/home')
def welcome_here():
	return jsonify({"message" : "Welcome Here"}), 200

@app.route('/result', methods = ['POST','GET'])
def giveResult():
	try:
		if request.is_json:
			content = request.get_json()
		else:
			raise Exception('Json','Json not found in body')
		if request.method == 'POST':
			if 'review' not in content or content['review'] is None:
				raise Exception('review', 'Review not found in POST request')
			else:
				review = request.form['review']
			    predicted_class = mlModule.predict(review)
			    result = jsonify({'class' : predicted_class}), 200
		else:
			result = jsonify({'Response' : 'No other request except POST requests are accepted on this server'}), 404
	except Exception as e:
		raise e
	return result
		
if __name__ == '__main__':
	app.run(debug = True)
	
