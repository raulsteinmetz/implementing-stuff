''' 
    Data loading and cleaning functions for the corpus
'''

import unicodedata

def get_corpus(fpath):
	'''
		Extracts text from a .txt file into a string
	'''
	with open(fpath, 'r') as f:
		return f.read()

def clean_text(corpus):
	'''
		Processes a string so that:
			- only english letters remain
			- only lower-case letters
			- [...] will add more as corpus increases
	'''

	# Take out accentuation from latin languages
	corpus = unicodedata.normalize('NFKD', corpus)
	corpus = ''.join(c for c in corpus if unicodedata.category(c) != 'Mn')

	# Convert into lower-case only
	corpus = corpus.lower()

	return corpus