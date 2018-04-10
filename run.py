# -*- coding: utf-8 -*-


import os
import numpy as np
import pandas as pd


from utils.loader import FNC_Data
from utils.split_data import split_train_val
from utils.preprocess import preprocess_dataset, one_hot
from feature_gen import *
from visualise import donut_plot, plot_feature_distribution, plot_confusion_matrix, visualise_learning_rate
from models.linear_regression import Linear_Regression_Model
from models.logistic_regression import Logistic_Regression_Model
from models.additional_models import random_forest, multi_layer_percep
from models.model_evaluation import get_performance_statistics
from sklearn.preprocessing import StandardScaler # Only used for additional ML features


# file names for retrieving data
file_train_instances = "train_stances.csv"
file_train_bodies = "train_bodies.csv"
file_test_instances = "competition_test_stances.csv"
file_test_bodies = "competition_test_bodies.csv"


# Parameters
train_proportion = 0.9
mu = 0.8 # Parameter for Dirichelet smothing
learning_rate_list = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001] # List of learning rates to try for the ML algorithms
# Note: learning rates greater than 0.05 may cause buffer oveflows due to diverging values. 
epochs = 1000 # Number of epochs to use for training
batch_size = 100 # Batch size to use for while training

# Use the validation performance metrics to fine-tune the values of any hyper-parameters
# Those chosen above are the most optimal 
optimum_learning_rate_linear_reg = 0.001
optimum_learning_rate_logistic_reg = 0.001

# Use thse variables to control the flow of the program
load_features = False # Set this to true if the features have already been computed.
gen_plots = True # Use this parameter to specify to generate feature distribution plots
validation_performance = True # If set to true, then iterate through the llist of learning rates,
# and evaluate performance on the validation set, additonally use this to tune hyper parameters



if __name__ == "__main__":

	if load_features == False:

		# Load the data
		print ("Reading Training dataset")
		train_data = FNC_Data(file_train_instances, file_train_bodies)
		print ("Reading Testing dataset")
		test_data = FNC_Data(file_test_instances, file_test_bodies)

		# Split the data into training and validation sets
		train_stances, validation_stances = split_train_val(train_data, train_proportion)

		# Preprocess tha headlines and the bodies for the training/validation set
		headlines_processed = preprocess_dataset(train_data.headlines)
		print ("Finished Prepocessing training and validation headlines")

		bodies_processed = preprocess_dataset(train_data.bodies)
		print ("Finished Prepocessing training and validation article bodies")

		# Preprocess tha headlines and the bodies for the testing set
		test_headlines_processed = preprocess_dataset(test_data.headlines)
		print ("Finished Prepocessing test headlines")
		test_bodies_processed = preprocess_dataset(test_data.bodies)
		print ("Finished Prepocessing testing article bodies")

		print ("Generating Vocabulary")
		# Vocabulary is generated based on the training set only
		vocabulary = generate_vocabulary(train_stances, headlines_processed, bodies_processed)
		
		# Generate term-frequencey vectors for headlines and bodies
		print ("Generating tf vectorised representations for train/validation set")
		headlines_vec, bodies_vec = vectorize_dataset(headlines_processed, bodies_processed, vocabulary)
		print ("Generating tf vectorised representations for test set")
		test_headlines_vec, test_bodies_vec = vectorize_dataset(test_headlines_processed, test_bodies_processed, vocabulary)

		# Generating TF-matrix for training data
		print ("Generating TF-matrix for training data")
		tf_matrix = generate_tf_matrix(train_stances, headlines_vec, bodies_vec)

		# Generate Features

		# Cosine similarity
		print ("Generating feature cosine similarity")
		train_cosine_sim = feature_cosine_similarity(train_stances, headlines_vec, bodies_vec, tf_matrix)
		validation_cosine_sim = feature_cosine_similarity(validation_stances, headlines_vec, bodies_vec, tf_matrix)
		test_cosine_sim = feature_cosine_similarity(test_data.stances, test_headlines_vec, test_bodies_vec, tf_matrix)

		# KL-Divergence
		print ("Generating feature KL-Divergence")
		train_kl_div = feature_kl_divergence(train_stances, headlines_vec, bodies_vec, tf_matrix, mu)
		validation_kl_div = feature_kl_divergence(validation_stances, headlines_vec, bodies_vec, tf_matrix, mu)
		test_kl_div = feature_kl_divergence(test_data.stances, test_headlines_vec, test_bodies_vec, tf_matrix, mu)

		# Jaccard Similarity
		print ("Generating feature Jaccard similarity")
		train_jacc_sim = feature_jaccard_similarity(train_stances, headlines_vec, bodies_vec)
		validation_jacc_sim = feature_jaccard_similarity(validation_stances, headlines_vec, bodies_vec)
		test_jacc_sim = feature_jaccard_similarity(test_data.stances, test_headlines_vec, test_bodies_vec)

		# Negative tokens feature
		print ("Generating feature negative tokens count")
		train_negative = negative_feature(train_stances, headlines_vec, bodies_vec, vocabulary)
		validation_negative = negative_feature(validation_stances, headlines_vec, bodies_vec, vocabulary)
		test_negative = negative_feature(test_data.stances, test_headlines_vec, test_bodies_vec, vocabulary)

		# Latent Semantic Analysis -- Cosine Similarity
		print ("Generating feature LSA cosine similarity")
		train_LSA = feature_LSA(train_stances, headlines_vec, bodies_vec, tf_matrix)
		validation_LSA = feature_LSA(validation_stances, headlines_vec, bodies_vec, tf_matrix)
		test_LSA = feature_LSA(test_data.stances, test_headlines_vec, test_bodies_vec, tf_matrix)

		print ("Feature generation complete")

		# Create feature array
		# At this point it is possible to extract/remove certain features
		train_features = np.array([train_cosine_sim, train_kl_div, train_jacc_sim, train_negative, train_LSA]).T
		validation_features = np.array([validation_cosine_sim, validation_kl_div, validation_jacc_sim, validation_negative, validation_LSA]).T
		test_features = np.array([test_cosine_sim, test_kl_div, test_jacc_sim, test_negative, test_LSA]).T

		# Generate a one-hot encoding for the training, validatoin and test stances
		one_hot_train = one_hot(train_stances)
		one_hot_validation = one_hot(validation_stances)
		one_hot_test = one_hot(test_data.stances)

		# Store the features and one-hot vectors for easy retrieval later
		np.save("resources/train_features.npy", train_features)
		np.save("resources/validation_features.npy", validation_features)
		np.save("resources/test_features.npy", test_features)
		np.save("resources/one_hot_train.npy", one_hot_train)
		np.save("resources/one_hot_validation.npy", one_hot_validation)
		np.save("resources/one_hot_test.npy", one_hot_test)
	
	else:

		# Features are already computed -- load the features directly
		train_features = np.load("resources/train_features.npy")
		validation_features = np.load("resources/validation_features.npy")
		test_features = np.load("resources/test_features.npy")

		# Features can be extracted/removed at this point

		# Load the one-hot encoded labels
		one_hot_train = np.load("resources/one_hot_train.npy")
		one_hot_validation = np.load("resources/one_hot_validation.npy")
		one_hot_test = np.load("resources/one_hot_test.npy")


	if gen_plots == True:

		print ("Generating Visualisations")
		# Visualise the test/validation split for each class
		donut_plot(one_hot_train, one_hot_validation)

		# Visualise the distribution of selected features
		plot_feature_distribution(train_features[:,1], one_hot_train, "KL-Divergence", (0,7), (0,100))
		plot_feature_distribution(train_features[:,5], one_hot_train, "LSI_Cosine-Similarity", (-1,1), (0,100))


	if validation_performance == True:

		# Machine learning using linear regression
		print ("Performing linear regression")
		linear_regression_losses = []
		linear_regression_results_validation = []

		for learning_rate in learning_rate_list:
			# Create a linear regression model object
			linear_model = Linear_Regression_Model(train_features, one_hot_train, validation_features, one_hot_validation, test_features, one_hot_test)
			# train the model
			losses = linear_model.train(batch_size, epochs, learning_rate)
			linear_regression_losses.append(losses)

			# Evaluate the performace of the model on the validation set -- use the results to fine tune any hyper-parameters
			model_performance_validation = linear_model.evaluate_performance(validation=True)

			log_file = open("resources/linear_model_validation.txt","a+")
			log_file.write("Model learning rate: %f,  Model Accuracy: %f, Model F1 Score %f \n" % (learning_rate, model_performance_validation[0], model_performance_validation[1]))
			log_file.close()

		# Generate plot of the learning rates
		visualise_learning_rate(linear_regression_losses, learning_rate_list, "LinearRegression", "MSE loss", (0, 1))


		# Machine learning using logistic regression
		print ("Performing logistic regression")
		logistic_regression_losses = []
		logistic_regression_results_validation = []


		for learning_rate in learning_rate_list:
			# Create a logistic regression model object
			logistic_model = Logistic_Regression_Model(train_features, one_hot_train, validation_features, one_hot_validation, test_features, one_hot_test)
			# train the model
			losses = logistic_model.train(batch_size, epochs, learning_rate)
			logistic_regression_losses.append(losses)

			# Evaluate the performace of the model on the validation set -- use the results to fine tune any parameters
			model_performance_validation = logistic_model.evaluate_performance(validation=True)

			# Append the performace statistics to a file
			log_file = open("resources/logistic_model_validation.txt","a+")
			log_file.write("Model learning rate: %f,  Model Accuracy: %f, Model F1 Score %f \n" % (learning_rate, model_performance_validation[0], model_performance_validation[1]))
			log_file.close()

		# Generate plot of the learning rates
		visualise_learning_rate(logistic_regression_losses, learning_rate_list, "LogisticRegression", "Cross-entropy loss", (30, 150))


	#----------- Train and evaluate final models ----------------

	# Train and evaluate final linear regression model
	print ("Final Linear model")

	final_linear_model = Linear_Regression_Model(train_features, one_hot_train, validation_features, one_hot_validation, test_features, one_hot_test)
	losses = final_linear_model.train(batch_size, epochs, optimum_learning_rate_linear_reg)
	# Evaluate the trained model on the test set
	model_performance = final_linear_model.evaluate_performance(test=True)
	# Write the results to a file
	log_file = open("resources/final_linear_model.txt","w+")
	log_file.write("Final model accuracy: %f, Weighted F1 Score %f \n" % (model_performance[0], model_performance[1]))
	log_file.close()
	# Plot confusion matrix and store
	plot_confusion_matrix(model_performance[2], "Linear_Model")


	# Train and evaluate final logistic regression model
	print ("Final Logistic model")

	final_logistic_model = Logistic_Regression_Model(train_features, one_hot_train, validation_features, one_hot_validation, test_features, one_hot_test)
	losses = final_logistic_model.train(batch_size, epochs, optimum_learning_rate_logistic_reg)
	# Evaluate the trained model on the test set
	model_performance = final_logistic_model.evaluate_performance(test=True)
	# Write the results to a file
	log_file = open("resources/final_logistic_model.txt","w+")
	log_file.write("Final model accuracy: %f, Weighted F1 Score %f \n" % (model_performance[0], model_performance[1]))
	log_file.close()
	# Plot confusion matrix and store
	plot_confusion_matrix(model_performance[2], "Logistic_Model")


	# Train and evaluate random forest model
	print ("Random Forest model")

	# Standardise the input data -- this results in better performance of the ML algorithms
	standardizer = StandardScaler().fit(train_features)

	random_forest.fit(standardizer.transform(train_features), np.argmax(one_hot_train, axis=1))
	# Evaluate the trained model on the test set
	predictions = random_forest.predict(standardizer.transform(test_features))
	model_performance = get_performance_statistics(predictions, np.argmax(one_hot_test, axis=1))
	# Write the results to a file
	log_file = open("resources/random_forest.txt","w+")
	log_file.write("Final model accuracy: %f, Weighted F1 Score %f \n" % (model_performance[0], model_performance[1]))
	log_file.close()
	# Plot confusion matrix and store
	plot_confusion_matrix(model_performance[2], "Random_Forest")


	# Train and evaluate NN model
	print ("Multi-Layer-Perceptron Neural Netrwork model")

	standardizer = StandardScaler().fit(train_features)

	multi_layer_percep.fit(standardizer.transform(train_features), np.argmax(one_hot_train, axis=1))
	# Evaluate the trained model on the test set
	predictions = multi_layer_percep.predict(standardizer.transform(test_features))
	model_performance = get_performance_statistics(predictions, np.argmax(one_hot_test, axis=1))
	# Write the results to a file
	log_file = open("resources/neural_network.txt","w+")
	log_file.write("Final model accuracy: %f, Weighted F1 Score %f \n" % (model_performance[0], model_performance[1]))
	log_file.close()
	# Plot confusion matrix and store
	plot_confusion_matrix(model_performance[2], "Neural_Network")


