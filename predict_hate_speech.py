
print('\n\nLoading Libraries...\n\n')

import os
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
sys.stderr = stderr


from keras.models import model_from_json
from keras.preprocessing import sequence
import pickle
from string import punctuation

from nltk.stem.porter import *

stemmer = PorterStemmer()

def load_model(model_fname, weights_fname):
	json_file = open(model_fname, 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights(weights_fname)
	loaded_model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	return loaded_model

def load_tokenizer(fname):
	with open(fname, 'rb') as handle:
		tokenizer = pickle.load(handle)
	return tokenizer

print('\n\nLoading Model and Weights...\n\n')

model = load_model('model.json', 'model.h5')
tokenizer = load_tokenizer('tokenizer.pickle')

def predict_comment(comment):
    comment = "".join([c for c in comment if c not in punctuation]).strip().lower()
    comment = " ".join([stemmer.stem(w) for w in comment.split()])
    comment = [comment]
    #print(comment)
    tokenized = tokenizer.texts_to_sequences(comment)
    padded = sequence.pad_sequences(tokenized, 1000)
    pred = model.predict(padded)[0][0]
    if 0.9 > pred > 0.7:
        label = 'Might be Hate Speech'
    elif pred > 0.9: 
        label =  'Most Likely Hate Speech'
    else: label = 'Probably Not Hate Speech'
    return (label, pred)


def run():

	print('\n\nBegin Entering Comments\n\n')

	cont = True
	txt = ""
	while cont != False:

		txt = input('Comment: ')
		if txt == 'end':
			print('--Closing--')
			cont = False
		else:
			label, prob = predict_comment(txt)

			print("\n" + label + ".\nConfidence: " + str(prob) + '\n')

if __name__ == '__main__':
	print('\n\n-----Instructions:')
	print('-----Type a comment below for the model to predict.')
	print("-----Type 'close' to close program.")
	run()