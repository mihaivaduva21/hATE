from keras.models import model_from_json
from keras.preprocessing import sequence
import pickle
from string import punctuation
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from nltk.stem.porter import *

import os
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
sys.stderr = stderr

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

def safe_search(word):
	try:
		idx = vocab_to_int[word]
	except KeyError:
		#print('Not in vocabulary')
		idx = 0
	return idx

def find_closest(word, num = 10):
	#w1idx = vocab_to_int[word]
	w1idx = safe_search(word)
	sims = enumerate(cosines[w1idx,:])
	sorted_sims = sorted(sims, key = lambda x: x[1], reverse = True)
	sorted_sims = [sim for sim in sorted_sims if sim[0] != w1idx]
	words = []

	words = [int_to_vocab[sim[0]] for sim in sorted_sims][:num]
	return words

model = load_model('model.json', 'model.h5')
tokenizer = load_tokenizer('tokenizer.pickle')

vocab_to_int = tokenizer.word_index
vocab_to_int['__PADDING__'] = 0

int_to_vocab = dict(zip(vocab_to_int.values(), vocab_to_int.keys()))

vecs = model.layers[0].get_weights()[0]
word_embeds = {w:vecs[idx] for w, idx in vocab_to_int.items()}

from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
import numpy as np

cosines = 1 - pairwise_distances(vecs, metric = 'cosine')

def find_furthest(word, num = 10):
	#w1idx = vocab_to_int[word]
	w1idx = safe_search(word)
	sims = enumerate(cosines[w1idx,:])
	sorted_sims = sorted(sims, key = lambda x: x[1])
	sorted_sims = [sim for sim in sorted_sims if sim[0] != w1idx]
	words = []
	words = [int_to_vocab[sim[0]] for sim in sorted_sims][:num]
	return words

hate_words = ['nigger', 'queer', 'fag', 'raghead', 'wetback', 'chink', 'bean', 'coon', 'negro']

def generate_clusters(keys):
	far_words = []
	for key in keys:
		far_words += find_furthest(key, 3)
	sample = np.random.choice(far_words, size = len(keys))
	#keys = keys + list(set(sample))
	#print(keys)
	embedding_clusters = []
	word_clusters = []
	#keys = keys + 
	unique_words = set()
	for word in keys:
		most_similar_words = find_closest(word, 100)
		words_no_dups = [word for word in most_similar_words if word not in unique_words]
		unique_words = unique_words.union(set(words_no_dups))
		embeddings = np.array([word_embeds[w2] for w2 in most_similar_words])
		embedding_clusters.append(embeddings)
		word_clusters.append(most_similar_words)

	embedding_clusters = np.array(embedding_clusters)

	n, m, k = embedding_clusters.shape
	tsne_model_2d = TSNE(perplexity = 30, n_components = 2, init = 'pca', n_iter = 3500)
	embeddings_2d = np.array(tsne_model_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)
	
	return embeddings_2d, word_clusters

def tsne_plot_similar_words(labels, embedding_clusters, word_clusters, a):

	plt.figure(figsize = (16,9))
	colors = cm.PuBuGn(np.linspace(0,1,len(labels)))
	unique_words = set()
	for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
		x = embeddings[:,0]
		y = embeddings[:,1]
		if label in hate_words:
			label = label[0] + '*'*(len(label) - 1)
			plt.scatter(x,y, c = 'r', alpha = a, label = label)

			for i, word in enumerate(words):
				if word in unique_words:
					continue
				else:
					if prof_pred([word]) == 1 or word in hate_words:
					    word = word[0] + '*'*(len(word) - 1)
					plt.annotate(word, alpha = 0.5, xy = (x[i], y[i]), xytext = (5,2),
								textcoords = 'offset points', ha = 'right', va = 'bottom', size = 8)
				unique_words.add(word)
			else:
				plt.scatter(x,y, c = [color], alpha = a, label = label)
			for i, word in enumerate(words):
				if word in unique_words:
					continue
				else:
					if prof_pred([word]) == 1 or word in hate_words:
						word = word[0] + '*'*(len(word) - 1)
					plt.annotate(word, alpha = 0.5, xy = (x[i], y[i]), xytext = (5,2),
								textcoords = 'offset points', ha = 'right', va = 'bottom', size = 8)
					unique_words.add(word)
	plt.legend(loc = 4)
    #plt.title(title)
	plt.grid(True)
	plt.show()

def run():


	stemmer = PorterStemmer()

	cont = True

	#ax = plt.axes()

	print("\n\n-----This is a tool for visualizing how the model builds relationships between words.\n")
	print("-----Word clusters corresponding to hate speech words should be geometrically different than those of neutral words.\n\n")
	print("-----NOTE: Similar words are stemmed.")

	print('\n---Instructions---')
	print('------Type Words to Generate Clusters.')
	print('------Make sure words are separated by a single space')
	print('------Example: "We are going to be alright"')
	print('------If you type in a hate word, all words similar to this will be red')
	print('------Note: Not all possible hate words will be recognized at this step.')
	print('------Resulting plot will be clusters of words most similar to your inputted words.')

	

	while cont == True:

		txt = input('\n\nType words: ')

		keys = txt.split()
		#print(keys)
		keys_stem = [stemmer.stem(w) for w in txt.split()]

		print('\n Generating Clusters')
		embedding_clusters, word_clusters = generate_clusters(keys_stem)
		#tsne_plot_similar_words(keys, embedding_clusters, word_clusters, 0.7)
		colors = cm.PuBuGn(np.linspace(0,1,len(keys)))
		
		plt.figure(figsize = (16,9))

		for cluster, word, color, close_words in zip(embedding_clusters, keys, colors, word_clusters):
			
			if stemmer.stem(word) in hate_words:
				c = ['r']
			else: c = [color]

			x = cluster[:,0]
			y = cluster[:,1]
			
			plt.scatter(x,y, c = c, label = word, s = 8)

			for i, close_word in enumerate(close_words):
				if len(close_word) > 15:
					close_word = close_word[:15]
				plt.annotate(close_word, xy = (x[i], y[i]), xytext = (5,2), alpha = 0.5,
								textcoords = 'offset points', ha = 'right', va = 'bottom', size = 8)

		plt.legend()
		plt.grid(True)
		plt.title('Most Similar Words')
		plt.show()

if __name__ == '__main__':

	run()