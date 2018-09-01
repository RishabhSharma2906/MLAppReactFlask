from flask import Flask, redirect, url_for, request, jsonify
from mlModule import predict

app = Flask(__name__)

@app.route('/home')
def welcome_here():
	return jsonify({"message" : "Welcome Here"}), 200

@app.route('/result', methods = ['POST','GET'])
def giveResult():
	if request.method == 'POST':
		statement = request.form['statement']
		print(statement)
		result = predict(statement)
		return jsonify({'class' : result}), 200
	else:
		return jsonify({'Response' : 'No other request except POST requests are accepted on this server'}), 404
		
if __name__ == '__main__':
	app.run(debug = True)
	
