# -*- coding: utf-8 -*-

import numpy as np
from models.model_evaluation import get_performance_statistics

class Linear_Regression_Model(object):

    # Define a class for creating a liner regression model

    def __init__(self, train_inputs, train_labels, validation_inputs,\
                 validation_labels, test_inputs, test_labels):
        
        # Add an additional feature with value 1 to all inputs (required for bias parameter)
        self.train_inputs = np.concatenate((train_inputs, np.ones((train_inputs.shape[0],1))), axis=1)
        self.train_labels = train_labels
        self.validation_inputs = np.concatenate((validation_inputs, np.ones((validation_inputs.shape[0],1))), axis=1)
        self.validation_labels = validation_labels
        self.test_inputs = np.concatenate((test_inputs, np.ones((test_inputs.shape[0],1))), axis=1)
        self.test_labels = test_labels
        self.W = np.random.uniform(size=(self.train_inputs.shape[1], 4))
        
        
    def train(self, batch_size, num_epochs, learning_rate):
        losses_during_epoch = []
        epoch_losses = []
        
        for epoch in range(num_epochs):
            for batch in np.arange(0, self.train_inputs.shape[0], batch_size):
                # Compute prediction
                prediction = np.dot(self.train_inputs[batch:batch+batch_size, :], self.W)
                
                derivative_cost = prediction - self.train_labels[batch:batch+batch_size, :]
                
                # Compute the MSE loss and store
                losses_during_epoch.append(np.mean(np.square(derivative_cost))) 
                
                # Compute gradient and update weigth parameters
                gradient = np.dot(self.train_inputs[batch:batch+batch_size, :].T, derivative_cost)
                self.W = self.W - learning_rate*(gradient/float(batch_size))
                
            # After ever epoch store the average loss
            epoch_losses.append(np.mean(losses_during_epoch))
            losses_during_epoch = []
            
        return epoch_losses
        
    def get_predictions(self, input_data):
        prediction_list = []
        # Do this stage in batches of 1 due to limitation of computational resources
        for input_vec in input_data:
            prediction_vec = np.dot(input_vec, self.W)
            prediction = np.argmax(prediction_vec)
            prediction_list.append(prediction)
        return np.array(prediction_list)

    def evaluate_performance(self, train=False, validation=False, test=False):
        if train==True:
            predictions = self.get_predictions(self.train_inputs)
            labels = np.argmax(self.train_labels, axis=1)
        elif validation==True:
            predictions = self.get_predictions(self.validation_inputs)
            labels = np.argmax(self.validation_labels, axis=1)
        elif test==True:
            predictions = self.get_predictions(self.test_inputs)
            labels = np.argmax(self.test_labels, axis=1)
            
        # Get evaluation statistics
        accuracy, f1, confusion_matrix = get_performance_statistics(predictions, labels)
        
        return accuracy, f1, confusion_matrix