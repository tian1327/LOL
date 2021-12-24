# -*- coding: utf-8 -*-
"""ColBERT_Task1a_Tian_BERT_LARGE_UNCASED.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1qfjpA4L9TMZYLUUDI3QSHW1ZJ_av0jY3
"""

#Mount the google drive
from google.colab import drive
drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
# direct to the folder where the data located, change the folder path here if needed
# %cd '/content/drive/MyDrive/CSCE 638 NLP Project/LOL_Data/' 
!ls

import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

# import tensorflow_hub as hub
import tensorflow as tf
# import bert_tokenization as tokenization
import tensorflow.keras.backend as K
from tensorflow import keras 

import os
from scipy.stats import spearmanr
from math import floor, ceil
!pip install transformers
!pip install sentencepiece
from transformers import *

import seaborn as sns
import string
import re    #for regex

np.set_printoptions(suppress=True)
print(tf.__version__)

"""# Prep / tokenizer

#### 1. Read data and tokenizer

Read tokenizer and data, as well as defining the maximum sequence length that will be used for the input to Bert (maximum is usually 512 tokens)
"""

training_sample_count = 8000
training_epochs = 3
dev_count = 1000
test_count = 1000

running_folds = 1

MAX_SENTENCE_LENGTH = 20 # max number of words in a sentence
MAX_SENTENCES = 5 # max number of sentences to encode in a text

MAX_LENGTH = 100 # max words in a text as whole sentences

"""### load dataset"""

df_train = pd.read_csv('train8000.csv')
df_train = df_train[:training_sample_count*running_folds]
display(df_train.head(3))

df_dev = pd.read_csv('dev1000.csv')
df_test = pd.read_csv('test1000.csv')
df_dev = df_dev[:dev_count]
df_test = df_test[:test_count]
display(df_dev.head(3))
display(df_test.head(3))

# check the pos/negative label of the data
print(df_train.describe())
print(df_dev.describe())
print(df_test.describe())

print(sum(df_train['is_humor']==0)/len(df_train['is_humor']))
print(sum(df_dev['is_humor']==0)/len(df_dev['is_humor']))

output_categories = list(df_train.columns[[2]])
input_categories = list(df_train.columns[[1]])

TARGET_COUNT = len(output_categories)

print('\ninput categories:\n\t', input_categories)
print('\noutput categories:\n\t', output_categories)
print('\noutput TARGET_COUNT:\n\t', TARGET_COUNT)

"""## 2. Preprocessing functions

These are some functions that will be used to preprocess the raw text data into useable Bert inputs.<br>

"""

from transformers import BertTokenizer

MODEL_TYPE = 'bert-large-uncased'
tokenizer = BertTokenizer.from_pretrained(MODEL_TYPE)

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

def return_id(str1, str2, truncation_strategy, length):

    inputs = tokenizer.encode_plus(str1, str2,
        add_special_tokens=True,
        max_length=length,
        truncation_strategy=truncation_strategy)

    input_ids =  inputs["input_ids"] #token indices, numerical representations of tokens building the sequences that will be used as input by the model
    input_masks = [1] * len(input_ids) # indicate the ids should be attended
    input_segments = inputs["token_type_ids"] #BERT, also deploy token type IDs (also called segment IDs). They are represented as a binary mask identifying the two types of sequence in the model.

    padding_length = length - len(input_ids)
    padding_id = tokenizer.pad_token_id

    input_ids = input_ids + ([padding_id] * padding_length)
    input_masks = input_masks + ([0] * padding_length)
    input_segments = input_segments + ([0] * padding_length)

    return [input_ids, input_masks, input_segments]


def compute_input_arrays(df, columns, tokenizer):

    model_input = []
    for xx in range((MAX_SENTENCES*3)+3): # +3 for the whole sentences
        model_input.append([])
    
    for _, row in tqdm(df[columns].iterrows()):
        #print(type(row))
        #print(row)
        #print(row.text)
        #print(type(row.text))
        #stop

        i = 0
        
        # sent
        sentences = sent_tokenize(row.text) # separate a long text into sentences

        for xx in range(MAX_SENTENCES): # MAX_SENTENCES = 5

            s = sentences[xx] if xx<len(sentences) else ''

            ids_q, masks_q, segments_q = return_id(s, None, 'longest_first', MAX_SENTENCE_LENGTH) #MAX_SENTENCE_LENGTH = 20

            model_input[i].append(ids_q)
            i+=1
            model_input[i].append(masks_q)
            i+=1
            model_input[i].append(segments_q)
            i+=1
        
        # full row
        ids_q, masks_q, segments_q = return_id(row.text, None, 'longest_first', MAX_LENGTH) # MAX_LENGTH = 100

        model_input[i].append(ids_q)
        i+=1
        model_input[i].append(masks_q)
        i+=1
        model_input[i].append(segments_q)
        
    for xx in range((MAX_SENTENCES*3)+3):
        model_input[xx] = np.asarray(model_input[xx], dtype=np.int32)
        
    print(model_input[0].shape)

    return model_input

inputs      = compute_input_arrays(df_train, input_categories, tokenizer)
dev_inputs = compute_input_arrays(df_dev, input_categories, tokenizer)
test_inputs = compute_input_arrays(df_test, input_categories, tokenizer)

# check the tokenized sentences
print(len(inputs), len(inputs[0]), len(inputs[0][0]))

# check out input for 7th row
xx = 7
print(df_train.iloc[xx,1])
print(sent_tokenize(df_train.iloc[xx,1]))

inputs[0][xx], inputs[3][xx], inputs[6][xx], inputs[15][xx]

def compute_output_arrays(df, columns):
    return np.asarray(df[columns])

outputs = compute_output_arrays(df_train, output_categories)
dev_outputs = compute_output_arrays(df_dev, output_categories)

"""## 3. Create model


"""

#config = BertConfig() # print(config) to see settings
#config.output_hidden_states = False # Set to True to obtain hidden states
#bert_model = TFBertModel.from_pretrained('bert-large-uncased', config=config)
bert_model = TFBertModel.from_pretrained("bert-large-uncased")
#config

def create_model():
    # model structure
    # takes q_ids [max=20*MAX_SENTENCES] and a_ids [max=200]
    import gc
    
    model_inputs = []
    f_inputs=[]

    for i in range(MAX_SENTENCES):
        
        # bert embeddings
        q_id = tf.keras.layers.Input((MAX_SENTENCE_LENGTH,), dtype=tf.int32)
        q_mask = tf.keras.layers.Input((MAX_SENTENCE_LENGTH,), dtype=tf.int32)
        q_atn = tf.keras.layers.Input((MAX_SENTENCE_LENGTH,), dtype=tf.int32)
        q_embedding = bert_model(q_id, attention_mask=q_mask, token_type_ids=q_atn)[0]
        q = tf.keras.layers.GlobalAveragePooling1D()(q_embedding)
        
        # internal model
        hidden1 = keras.layers.Dense(32, activation="relu")(q)
        hidden2 = keras.layers.Dropout(0.3)(hidden1)
        hidden3 = keras.layers.Dense(8, activation='relu')(hidden2)
        
        f_inputs.append(hidden3)
        model_inputs.extend([q_id, q_mask, q_atn])
        
    # whole sentence
    a_id = tf.keras.layers.Input((MAX_LENGTH,), dtype=tf.int32)
    a_mask = tf.keras.layers.Input((MAX_LENGTH,), dtype=tf.int32)
    a_atn = tf.keras.layers.Input((MAX_LENGTH,), dtype=tf.int32)
    a_embedding = bert_model(a_id, attention_mask=a_mask, token_type_ids=a_atn)[0]
    a = tf.keras.layers.GlobalAveragePooling1D()(a_embedding)
    print(a.shape)
    
    # internal model
    hidden1 = keras.layers.Dense(256, activation="relu")(a)
    hidden2 = keras.layers.Dropout(0.2)(hidden1)
    hidden3 = keras.layers.Dense(64, activation='relu')(hidden2)

    f_inputs.append(hidden3)
    model_inputs.extend([a_id, a_mask, a_atn])
    
    # final classifier
    concat_ = keras.layers.Concatenate()(f_inputs)
    hiddenf1 = keras.layers.Dense(512, activation='relu')(concat_)
    hiddenf2 = keras.layers.Dropout(0.2)(hiddenf1)
    hiddenf3 = keras.layers.Dense(256, activation='relu')(hiddenf2)
    
    output = keras.layers.Dense(TARGET_COUNT, activation='sigmoid')(hiddenf3) # softmax
    model = keras.Model(inputs=model_inputs, outputs=[output] )
    
    gc.collect()
    return model

model = create_model()
model.summary()

from tensorflow.keras.utils import plot_model
plot_model(model, to_file='./Results/ColBERT_Large_model_plot.png', show_shapes=True, show_layer_names=True)

"""## 5. Training, validation and testing

Loops over the folds in gkf and trains each fold for 3 epochs --- with a learning rate of 3e-5 and batch_size of 6. A simple binary crossentropy is used as the objective-/loss-function. 
"""

# Evaluation Metrics
import sklearn

def print_evaluation_metrics(y_true, y_pred, label='', is_regression=True, label2=''):

    print('==================', label2)
    ### For regression
    if is_regression:
        print("Regression task returns: MSE")
        print('mean_absolute_error',label,':', sklearn.metrics.mean_absolute_error(y_true, y_pred))
        print('mean_squared_error',label,':', sklearn.metrics.mean_squared_error(y_true, y_pred))
        print('r2 score',label,':', sklearn.metrics.r2_score(y_true, y_pred))
        #     print('max_error',label,':', sklearn.metrics.max_error(y_true, y_pred))
        return sklearn.metrics.mean_squared_error(y_true, y_pred)
    else:
        ### FOR Classification
#         print('balanced_accuracy_score',label,':', sklearn.metrics.balanced_accuracy_score(y_true, y_pred))
#         print('average_precision_score',label,':', sklearn.metrics.average_precision_score(y_true, y_pred))
#         print('balanced_accuracy_score',label,':', sklearn.metrics.balanced_accuracy_score(y_true, y_pred))
#         print('accuracy_score',label,':', sklearn.metrics.accuracy_score(y_true, y_pred))
        
        print("Classification returns: Acc")

        print('f1_score',label,':', sklearn.metrics.f1_score(y_true, y_pred))
        
        matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)
        print(matrix)
        TP,TN,FP,FN = matrix[1][1],matrix[0][0],matrix[0][1],matrix[1][0]
        Accuracy = (TP+TN)/(TP+FP+FN+TN)
        Precision = TP/(TP+FP)
        Recall = TP/(TP+FN)
        F1 = 2*(Recall * Precision) / (Recall + Precision)
        print('Acc', Accuracy, 'Prec', Precision, 'Rec', Recall, 'F1',F1)

        return sklearn.metrics.accuracy_score(y_true, y_pred)

# test
print_evaluation_metrics([1,0], [0.9,0.1], '', True)
print_evaluation_metrics([1,0], [1,1], '', False)

"""### Loss function selection
Regression problem between 0 and 1, so binary_crossentropy and mean_absolute_error seem good.

Here are the explanations: https://www.dlology.com/blog/how-to-choose-last-layer-activation-and-loss-function/
"""

min_acc = 100
min_test = []

dev_preds = []
test_preds = []
best_model = False

for BS in [6]:
    LR = 1e-5
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print('LR=', LR)

    gkf = GroupKFold(n_splits=2).split(X=df_train.text, groups=df_train.text)

    for fold, (train_idx, valid_idx) in enumerate(gkf):

        if fold not in range(running_folds):
            continue

        train_inputs = [(inputs[i][:])[:training_sample_count] for i in range(len(inputs))]
        train_outputs = (outputs[:])[:training_sample_count]

        #train_inputs = [(inputs[i][train_idx])[:training_sample_count] for i in range(len(inputs))]
        #train_outputs = (outputs[train_idx])[:training_sample_count]

        #valid_inputs = [inputs[i][valid_idx] for i in range(len(inputs))]
        #valid_outputs = outputs[valid_idx]

        #print(len(train_idx), len(train_outputs))

        model = create_model()
        K.clear_session()
        optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
        model.compile(loss='binary_crossentropy', optimizer=optimizer)
        print('model compiled')
        
        model.fit(train_inputs, train_outputs, epochs=training_epochs, batch_size=BS, verbose=1,
#                 validation_split=0.2, 
#                  validation_data=(x_val, y_val)
                 )
        # model.save_weights(f'bert-{fold}.h5')
        # valid_preds.append(model.predict(valid_inputs))

        #test_preds.append(model.predict(test_inputs))
        dev_preds.append(model.predict(dev_inputs))
        
        acc = print_evaluation_metrics(np.array(dev_outputs), np.array(dev_preds[-1]))

        if acc < min_acc:
            print('new acc >> ', acc)
            min_acc = acc
            best_model = model

"""## Regression submission"""

# check the dev set results
min_test = best_model.predict(dev_inputs)
df_dev['is_humor_pred'] = min_test

print_evaluation_metrics(df_dev['is_humor'], df_dev['is_humor_pred'], '', True)
df_dev.head()

"""## Binary submission"""

# try different splits to pick optimal split
for split in np.arange(0.1, 0.99, 0.1).tolist():
  df_dev['pred_bi'] = (df_dev['is_humor_pred'] > split)
  print_evaluation_metrics(df_dev['is_humor'], df_dev['pred_bi'], '', False, 'SPLIT on '+str(split))
  df_dev.head()

# check optimal split
split = 0.4
df_dev['pred_bi'] = (df_dev['is_humor_pred'] > split)
print_evaluation_metrics(df_dev['is_humor'], df_dev['pred_bi'], '', False, 'SPLIT on '+str(split))
df_dev.head()

# show wrong prediction examples
df_dev[df_dev['pred_bi']!=df_dev['is_humor']]

"""### Get Test Set Results for Submission"""

min_test = best_model.predict(test_inputs)
df_test['is_humor'] = min_test

df_test['is_humor'] = (df_test['is_humor'] > split)
print(df_test.head())

df_test['is_humor'] = df_test['is_humor'].astype(int)
print(df_test.head())

# drop the text column for submission
df_sub = df_test.drop('text',axis = 1)
print(df_sub.head())

df_sub.to_csv('./Results/ColBERT_LargeUncased_Task1a.csv', index=False)