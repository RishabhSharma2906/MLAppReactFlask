import mlModule as ml

class Setup:
	def __init__(self):
		X_train, X_test, Y_train, Y_test = ml.preprocessing()
		print('1')
		barebone_model = ml.model_building(X_train, Y_train)
		batch_size = 32
		print('2')
		built_model = ml.training(barebone_model, X_train, Y_train, batch_size)
		testing_score = ml.testing(built_model, X_test, Y_test, batch_size)
		print(testing_score)
		model_json_file_name =  "model/model.json"
		model_weight_file_name = "model/model_weights.h5"
		ml.save_model(built_model, model_json_file_name, model_weight_file_name)

setup_object = Setup()
