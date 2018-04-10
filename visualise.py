# -*- coding: utf-8 -*-

import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import itertools


def donut_plot(training_gt, validation_gt):

    # get the counts for each class
    train_counts = training_gt.sum(axis=0)
    validation_counts = validation_gt.sum(axis=0)
    
    class_list = ["Agree", "Disagree", "Discuss", "Unrelated"]
    
    for i, title in enumerate(class_list):
        
        sizes = [train_counts[i], validation_counts[i]]
        labels = ['Training\n\n' + str(train_counts[int(i)]), 'Validation\n\n' + str(validation_counts[int(i)])]
        colors = ['blue', 'red']

        fig = plt.figure()
        plt.pie(sizes, labels=labels, colors=colors, pctdistance=0.4, autopct='%1.1f%%')

        #draw a circle at the center of pie to make it look like a donut
        centre_circle = plt.Circle((0,0), 0.75, color='black', fc='white',linewidth=0.25)
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)

        # Set aspect ratio to be equal.
        plt.axis('equal')
        plt.title(title)
        plt.rcParams['patch.linewidth'] = 0.4 
        fig.savefig("visualisations/" + title + "_Split.png")



def plot_feature_distribution(feature, gt_stance, feature_name, xlim, ylim):

    # Filter by stance 
    agree_features = feature[gt_stance[:,0] == 1]
    disagree_features = feature[gt_stance[:,1] == 1]
    discuss_features = feature[gt_stance[:,2] == 1]
    unrelated_features = feature[gt_stance[:,3] == 1]
    
    fig = plt.figure()
    ax = sns.distplot(agree_features, kde=False, bins=300, color='red', label="Agree")
    plt.title("Agree")
    plt.xlabel(feature_name)
    plt.ylabel("density")
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    fig.savefig("visualisations/" + feature_name + "_Agree.png")
    
    fig = plt.figure()
    ax = sns.distplot(disagree_features, kde=False, bins=300, color='blue', label="Disagree")
    plt.title("Disagree")
    plt.xlabel(feature_name)
    plt.ylabel("density")
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    fig.savefig("visualisations/" + feature_name + "_Disagree.png")
    
    fig = plt.figure()
    ax = sns.distplot(discuss_features, kde=False, bins=300, color='yellow', label="Discuss")
    plt.title("Discuss")
    plt.xlabel(feature_name)
    plt.ylabel("density")
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    fig.savefig("visualisations/" + feature_name + "_Discuss.png")
    
    fig = plt.figure()
    ax = sns.distplot(unrelated_features, kde=False, bins=300, color='green', label="Unrelated")
    plt.title("Unrelated")
    plt.xlabel(feature_name)
    plt.ylabel("density")
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    fig.savefig("visualisations/" + feature_name + "_Unrelated.png")



def plot_confusion_matrix(cm, title, cmap=plt.cm.Blues):

    classes = ['Agree', 'Disagree', 'Discuss', 'Unrelated']
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fig.savefig("visualisations/" + title + "_cf_matrix.png")



def visualise_learning_rate(losses, learning_rates, model, loss_type, y_range):

    fig = plt.figure()

    for loss in losses:
        plt.plot(np.arange(len(loss)), loss)
    
    plt.ylim(y_range[0], y_range[1])
    plt.title("Comparison of Learning Rates")
    plt.xlabel("Iteration")
    plt.ylabel(loss_type)
    plt.legend([str(lr) for lr in learning_rates], loc='upper right')
    fig.savefig("visualisations/" + model + "_learning_rate.png")
