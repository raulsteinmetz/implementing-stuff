from data import get_corpus, clean_text
from tokenizer import build_tokenizer

# temp - transformer development, will move to separate file after
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim


# ---- Hyperparameters ----
CONTEXT_LENGTH = 4
EMBED_DIM = 32
VOCAB_SIZE = 128


class TransformerBlock(nn.Module):
	def __init__(self, ):
		pass
	def forward(self, ):
		pass


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
	tokenizer = build_tokenizer(corpus, VOCAB_SIZE)
	print('[main] Tokenizer retrieved.')

	# ---- Transformer creation/loading ----
	
	# temp - test transformer during development
	# 1. 


	# ---- Transfomer training ----

	# ---- Checkpoint selection ----

	# ---- Eval ----
	
	# ---- Text generation ----


if __name__ == '__main__':
	main()

