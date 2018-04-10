# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import random

def split_train_val(dataset, train_prop):
    # Split the data, ensure that both the train and validation sets have 
    # similar ratios of the four classes
    train_stances = []
    validation_stances = []
    
    stances = ['unrelated', 'discuss', 'agree', 'disagree']
    # Convert the dictionary containing stances to a Pandas Dataframe
    df_stances = pd.DataFrame(dataset.stances)
    
    for stance in stances:
        # Filter to get a particular stance only
        filtered_df = df_stances[df_stances['Stance'] == stance] 
        # Get the count for that stance
        count = filtered_df.count()[0]
        # Split the data of that stance using the training proportion
        filtered_df_split = np.split(filtered_df, [int(round(count*train_prop))], axis=0)
        # Concatenate the lists
        train_stances += filtered_df_split[0].to_dict('records')
        validation_stances += filtered_df_split[1].to_dict('records')
    
    # At this point the elements in both lists are sorted based on the stance
    # Randomly shuffle the elements within each list
    random.shuffle(train_stances)
    random.shuffle(validation_stances)
    return (train_stances, validation_stances)

