# -*- coding: utf-8 -*-

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


random_forest = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=0, class_weight='balanced')


multi_layer_percep = MLPClassifier(hidden_layer_sizes=(50, 100, 100, 100, 50), activation='relu', solver='adam', alpha=0.0001, batch_size=100,\
learning_rate='adaptive', learning_rate_init=0.01, max_iter=2000000, tol=0.0001, momentum=0.9) 