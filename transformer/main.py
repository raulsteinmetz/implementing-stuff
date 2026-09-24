import unicodedata
import hashlib
import os
import json
import torch


# ---- Hyperparameters ----
CONTEXT_LENGTH = 4
EMBED_DIM = 32
VOCAB_SIZE = 128

# ---- Data preparation ----

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

# ---- Tokenization ----

class Tokenizer:
	def __init__(self, base_chars, merge_order):
		pass

	def tokenize(self, text):
		pass

	def detokenize(self, tokens):
		pass

def build_tokenizer(corpus):
	'''
		Builds and stores in disk a BPE tokenizer for the corpus if it doesn't exist already;
		Returns 'Tokenizer' object.
	'''

	# ---- Check if tokenizer was already built for this corpus ----

	# Hash current corpus, so we do not need to run this code again for a processed corpus
	corpus_hash = hashlib.sha256(corpus.encode("utf-8")).hexdigest()
	
	if os.path.exists('./tokenizer.json'):
		with open('./tokenizer.json') as f:
			old_tokenizer_data = json.load(f)
			if old_tokenizer_data['corpus_hash'] == corpus_hash:
				print('[tokenizer] A tokenizer which matches the corpus was found in this directory - Skipping tokenizer creation and returning the existing tokenizer...')
				return Tokenizer(
					old_tokenizer_data['base_chars'],
					old_tokenizer_data['merge_order'])
			else:
				print('[tokenizer] A tokenizer was found in this directory, but the corpus with which is was built does not match - Overwriting old tokenizer...')
	else:
		print('No tokenizer was found in this directory. Building it...')

	# ---- Build tokenizer (Byte-Pair Encoding)----

	# Vocabulary starts as a list of unique chars in the corpus mapped to token indexes (integers) -
	# and we add new tokens based on what sequences of chars appear more frequently
	base_chars = sorted(set(corpus))
	text_to_token = {char: i for i, char in enumerate(base_chars)}

	# Map corpus into the correspondant token list
	tokens = [text_to_token[char] for char in corpus] 

	# Keep creating new tokens until we reach the desired vocabulary size
	merge_order = [] # save order for the function that tokenizes text
	while len(base_chars) + len(merge_order) < VOCAB_SIZE:
		# 1. Check which pair of tokens appears together more frequently
		frequency_map = {}
		for i in range(len(tokens) - 1):
			pair = (tokens[i], tokens[i + 1])
			frequency_map[pair] = frequency_map.get(pair, 0) + 1
		most_frequent = max(frequency_map, key=frequency_map.get)

		# 2. Most frequent pair of tokens becomes a new token
		new_token = len(base_chars) + len(merge_order)

		# 3. Replace pair occurance with new token 
		old_tokens = tokens
		tokens = []
		i = 0
		while i < len(old_tokens):
			if (i < len(old_tokens) - 1 and (old_tokens[i], old_tokens[i + 1]) == most_frequent):
				tokens.append(new_token)
				i += 2
			else:
				tokens.append(old_tokens[i])
				i += 1

		merge_order.append((new_token, most_frequent))

	# ---- Save tokenizer ----

	tokenizer_data = {
		'corpus_hash': corpus_hash,
		'base_chars': base_chars, 
		'merge_order': merge_order
	}

	with open('./tokenizer.json', 'w') as f:
		json.dump(tokenizer_data, f)
	return Tokenizer(
		tokenizer_data['base_chars'],
		tokenizer_data['merge_order']
	)


def main():
	'''
		Loads and processes text data, creates tokenizer. [will do more]
	'''

	# ---- Data Processing ----

	corpus = get_corpus('corpus.txt')
	print('[main] Loaded corpus.')
	corpus = clean_text(corpus)
	print('[main] Cleaned corpus.')

	# ---- Tokenizer creation / loading ----
	tokenizer = build_tokenizer(corpus)
	print('[main] Tokenizer retrieved.')

	# tokenizer test
	# [...]

if __name__ == '__main__':
	main()

