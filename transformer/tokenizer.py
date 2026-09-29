''' BPE tokenizer '''

import os
import hashlib
import json

def _merge_one(tokens, merge):
	old_tokens = tokens
	tokens = []
	i = 0
	while i < len(old_tokens):
		if (i < len(old_tokens) - 1 and (old_tokens[i], old_tokens[i + 1]) == merge[1]):
			tokens.append(merge[0])
			i += 2
		else:
			tokens.append(old_tokens[i])
			i += 1
	return tokens

class Tokenizer:
	def __init__(self, base_chars, merge_order):
		# 1. Create dictionaries
		self.text_to_token = {char: token for token, char in enumerate(base_chars)} # for tokenize
		self.token_to_text = {token:char for token, char in enumerate(base_chars)} # for detokenize, start with base
		for merged_id, to_merge in merge_order: # add merged_ids and their correspondance in text
			self.token_to_text[merged_id] = self.token_to_text[to_merge[0]] + self.token_to_text[to_merge[1]]

		# 2. Save merge order as an attribute of the class 
		self.merge_order = [(merged_id, tuple(list_pair)) for merged_id, list_pair in merge_order] # json converts tuples to lists, so we are reversing it here

	def tokenize(self, text):
		# Repeat the tokenizer creation process, except we already know which tokens to merge in which order
		tokens = [self.text_to_token[char] for char in text] # initial translation using base chars
		for merge in self.merge_order: # follow the merge order, explanation in the build_tokenizer() function 
				tokens = _merge_one(tokens, merge)

		return tokens

	def detokenize(self, tokens):
		return ''.join([self.token_to_text[token] for token in tokens])

def build_tokenizer(corpus, vocab_size):
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
	while len(base_chars) + len(merge_order) < vocab_size:
		# 1. Check which pair of tokens appears together more frequently
		frequency_map = {}
		for i in range(len(tokens) - 1):
			pair = (tokens[i], tokens[i + 1])
			frequency_map[pair] = frequency_map.get(pair, 0) + 1
		most_frequent = max(frequency_map, key=frequency_map.get)

		# 2. Most frequent pair of tokens becomes a new token
		new_token = len(base_chars) + len(merge_order)

		# 3. Replace pair occurance with new token 
		tokens = _merge_one(tokens, (new_token, most_frequent))
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
