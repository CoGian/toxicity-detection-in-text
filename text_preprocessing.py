# Load Dataset and Parameter Initialization
import gc
import os.path
import time
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize.treebank import TreebankWordTokenizer
import pickle
import operator
import string
import argparse

"""
# Preprocessing
## Word Embeddings
"""

nltk.download('punkt')
nltk.download('stopwords')

parser = argparse.ArgumentParser()
parser.add_argument(
	"--data_path",
	"-d",
	help="path of the datasets",
)

args = parser.parse_args()
data_path = args.data_path

TRAIN_DATASET_PATH = os.path.join(data_path, 'train.csv')
TEST_PUBLIC_DATASET_PATH = os.path.join(data_path, 'test_public_expanded.csv')
TEST_PRIVATE_DATASET_PATH = os.path.join(data_path, 'test_private_expanded.csv')
GLOVE_EMBEDDING_PATH = os.path.join(data_path, 'glove.840B.300d.pkl')

"""Read the datasets """

train_df = pd.read_csv(TRAIN_DATASET_PATH)
test_public_df = pd.read_csv(TEST_PUBLIC_DATASET_PATH)
test_private_df = pd.read_csv(TEST_PRIVATE_DATASET_PATH)


def load_embeddings(path):
	with open(path, 'rb') as f:
		embedding_index = pickle.load(f)
	return embedding_index


def build_matrix(word_index, path):
	embedding_index = load_embeddings(path)
	embedding_matrix = np.zeros((len(word_index) + 1, 300))
	unknown_words = []

	for word, i in word_index.items():
		try:
			embedding_matrix[i] = embedding_index[word]
		except KeyError:
			unknown_words.append(word)
	return embedding_matrix, unknown_words


def build_vocab(sentences, verbose=True):
	"""
	build_vocab builds a ordered dictionary of words and their frequency in your text corpus.
	:param sentences: list of list of words
	:return: dictionary of words and their count
	"""
	vocab = {}
	for sentence in sentences:
		for word in sentence:
			try:
				vocab[word] += 1
			except KeyError:
				vocab[word] = 1
	return vocab


def check_coverage(vocab, embeddings_index):
	"""
	goes through a given vocabulary and tries to find word vectors in your embedding matrix
	"""
	known_words = {}
	unknown_words = {}
	num_known_words = 0
	num_unknown_words = 0
	for word in vocab.keys():
		try:
			known_words[word] = embeddings_index[word]
			num_known_words += vocab[word]
		except:
			unknown_words[word] = vocab[word]
			num_unknown_words += vocab[word]
			pass

	print('Found embeddings for {:.2%} of vocab'.format(len(known_words) / len(vocab)))
	print('Found embeddings for  {:.2%} of all text'.format(num_known_words / (num_known_words + num_unknown_words)))
	unknown_words = sorted(unknown_words.items(), key=operator.itemgetter(1))[::-1]

	return unknown_words


tic = time.time()
glove_embeddings = load_embeddings(GLOVE_EMBEDDING_PATH)
print(f'loaded {len(glove_embeddings)} word vectors in {time.time() - tic}s')
vocab = build_vocab(list(train_df['comment_text'].apply(lambda x: x.split())))
unknown_words = check_coverage(vocab, glove_embeddings)
print("Top 10 Unknown words:")
print(unknown_words[:10])
del vocab
gc.collect()

"""Seems like ' and other punctuation directly on or in a word is an issue. We could simply delete punctuation to fix that words, but there are better methods. Lets explore the embeddings, in particular symbols a bit. For that we first need to define "what is a symbol" in contrast to a regular letter. I nowadays use the following list for "regular" letters. And symbols are all characters not in that list.
## Delete symbols that we have no embeddings and split contractions
"""

latin_similar = "’'‘ÆÐƎƏƐƔĲŊŒẞÞǷȜæðǝəɛɣĳŋœĸſßþƿȝĄƁÇĐƊĘĦĮƘŁØƠŞȘŢȚŦŲƯY̨Ƴąɓçđɗęħįƙłøơşșţțŧųưy̨ƴÁÀÂÄǍĂĀÃÅǺĄÆǼǢƁĆĊĈČÇĎḌĐƊÐÉÈĖÊËĚĔĒĘẸƎƏƐĠĜǦĞĢƔáàâäǎăāãåǻąæǽǣɓćċĉčçďḍđɗðéèėêëěĕēęẹǝəɛġĝǧğģɣĤḤĦIÍÌİÎÏǏĬĪĨĮỊĲĴĶƘĹĻŁĽĿʼNŃN̈ŇÑŅŊÓÒÔÖǑŎŌÕŐỌØǾƠŒĥḥħıíìiîïǐĭīĩįịĳĵķƙĸĺļłľŀŉńn̈ňñņŋóòôöǒŏōõőọøǿơœŔŘŖŚŜŠŞȘṢẞŤŢṬŦÞÚÙÛÜǓŬŪŨŰŮŲỤƯẂẀŴẄǷÝỲŶŸȲỸƳŹŻŽẒŕřŗſśŝšşșṣßťţṭŧþúùûüǔŭūũűůųụưẃẁŵẅƿýỳŷÿȳỹƴźżžẓ"
white_list = string.ascii_letters + string.digits + latin_similar + ' ' + "'"

"""Print all symbols that we have an embedding vector for."""

glove_chars = ''.join([c for c in glove_embeddings if len(c) == 1])
glove_symbols = ''.join([c for c in glove_chars if not c in white_list])

"""Print symbols in our comments """

jigsaw_chars = build_vocab(list(train_df["comment_text"]))
jigsaw_symbols = ''.join([c for c in jigsaw_chars if not c in white_list])


"""Delete all symbols we have no embeddings for"""

symbols_to_delete = ''.join([c for c in jigsaw_symbols if not c in glove_symbols])
symbols_to_isolate = ''.join([c for c in jigsaw_symbols if c in glove_symbols])  # we are keeping them

isolate_dict = {ord(c): f' {c} ' for c in symbols_to_isolate}
remove_dict = {ord(c): f'' for c in symbols_to_delete}


def handle_punctuation(x):
	x = x.translate(remove_dict)
	x = x.translate(isolate_dict)
	return x


train_df['comment_text'] = train_df['comment_text'].apply(lambda x: handle_punctuation(x))
test_public_df['comment_text'] = test_public_df['comment_text'].apply(lambda x: handle_punctuation(x))
test_private_df['comment_text'] = test_private_df['comment_text'].apply(lambda x: handle_punctuation(x))

"""Check Coverage"""

vocab = build_vocab(list(train_df['comment_text'].apply(lambda x: x.split())))
unknown_words = check_coverage(vocab, glove_embeddings)
print("Top 10 Unknown words:")
print(unknown_words[:10])
del vocab
gc.collect()
"""Now lets split standard contraction that will fix the issue with the ' punctuation"""

tokenizer = TreebankWordTokenizer()


def handle_contractions(x):
	x = tokenizer.tokenize(x)
	x = ' '.join(x)
	return x


train_df['comment_text'] = train_df['comment_text'].apply(lambda x: handle_contractions(x))
test_public_df['comment_text'] = test_public_df['comment_text'].apply(lambda x: handle_contractions(x))
test_private_df['comment_text'] = test_private_df['comment_text'].apply(lambda x: handle_contractions(x))

"""Check Coverage"""

vocab = build_vocab(list(train_df['comment_text'].apply(lambda x: x.split())))
unknown_words = check_coverage(vocab, glove_embeddings)
print("Top 10 Unknown words:")
print(unknown_words[:10])
del vocab
gc.collect()
"""## Check if lowercase/uppercase a word without embedding , find embedding  """


def check_case(comment, embeddings_index):
	comment = comment.split()

	comment = [
		word if word in embeddings_index else word.lower() if word.lower() in embeddings_index else word.title() if word.title() in embeddings_index else word
		for word in comment]

	comment = ' '.join(comment)
	return comment


train_df['comment_text'] = train_df['comment_text'].apply(lambda x: check_case(x, glove_embeddings))
test_public_df['comment_text'] = test_public_df['comment_text'].apply(lambda x: check_case(x, glove_embeddings))
test_private_df['comment_text'] = test_private_df['comment_text'].apply(lambda x: check_case(x, glove_embeddings))

vocab = build_vocab(list(train_df['comment_text'].apply(lambda x: x.split())))
unknown_words = check_coverage(vocab, glove_embeddings)
print("Top 10 Unknown words:")
print(unknown_words[:10])
del vocab
gc.collect()

"""## More cleaning of the contractions """

contraction_mapping = {
	"daesh": "isis", "Qur'an": "quran",
	"Trump's": 'trump is', "'cause": 'because', ',cause': 'because', ';cause': 'because', "ain't": 'am not',
	'ain,t': 'am not',
	'ain;t': 'am not', 'ain´t': 'am not', 'ain’t': 'am not', "aren't": 'are not',
	'aren,t': 'are not', 'aren;t': 'are not', 'aren´t': 'are not', 'aren’t': 'are not', "can't": 'cannot',
	"can't've": 'cannot have', 'can,t': 'cannot', 'can,t,ve': 'cannot have',
	'can;t': 'cannot', 'can;t;ve': 'cannot have',
	'can´t': 'cannot', 'can´t´ve': 'cannot have', 'can’t': 'cannot', 'can’t’ve': 'cannot have',
	"could've": 'could have', 'could,ve': 'could have', 'could;ve': 'could have', "couldn't": 'could not',
	"couldn't've": 'could not have', 'couldn,t': 'could not', 'couldn,t,ve': 'could not have', 'couldn;t': 'could not',
	'couldn;t;ve': 'could not have', 'couldn´t': 'could not',
	'couldn´t´ve': 'could not have', 'couldn’t': 'could not', 'couldn’t’ve': 'could not have', 'could´ve': 'could have',
	'could’ve': 'could have', "didn't": 'did not', 'didn,t': 'did not', 'didn;t': 'did not', 'didn´t': 'did not',
	'didn’t': 'did not', "doesn't": 'does not', 'doesn,t': 'does not', 'doesn;t': 'does not', 'doesn´t': 'does not',
	'doesn’t': 'does not', "don't": 'do not', 'don,t': 'do not', 'don;t': 'do not', 'don´t': 'do not',
	'don’t': 'do not',
	"hadn't": 'had not', "hadn't've": 'had not have', 'hadn,t': 'had not', 'hadn,t,ve': 'had not have',
	'hadn;t': 'had not',
	'hadn;t;ve': 'had not have', 'hadn´t': 'had not', 'hadn´t´ve': 'had not have', 'hadn’t': 'had not',
	'hadn’t’ve': 'had not have', "hasn't": 'has not', 'hasn,t': 'has not', 'hasn;t': 'has not', 'hasn´t': 'has not',
	'hasn’t': 'has not',
	"haven't": 'have not', 'haven,t': 'have not', 'haven;t': 'have not', 'haven´t': 'have not', 'haven’t': 'have not',
	"he'd": 'he would',
	"he'd've": 'he would have', "he'll": 'he will',
	"he's": 'he is', 'he,d': 'he would', 'he,d,ve': 'he would have', 'he,ll': 'he will', 'he,s': 'he is',
	'he;d': 'he would',
	'he;d;ve': 'he would have', 'he;ll': 'he will', 'he;s': 'he is', 'he´d': 'he would', 'he´d´ve': 'he would have',
	'he´ll': 'he will',
	'he´s': 'he is', 'he’d': 'he would', 'he’d’ve': 'he would have', 'he’ll': 'he will', 'he’s': 'he is',
	"how'd": 'how did', "how'll": 'how will',
	"how's": 'how is', 'how,d': 'how did', 'how,ll': 'how will', 'how,s': 'how is', 'how;d': 'how did',
	'how;ll': 'how will',
	'how;s': 'how is', 'how´d': 'how did', 'how´ll': 'how will', 'how´s': 'how is', 'how’d': 'how did',
	'how’ll': 'how will',
	'how’s': 'how is', "i'd": 'i would', "i'll": 'i will', "i'm": 'i am', "i've": 'i have', 'i,d': 'i would',
	'i,ll': 'i will',
	'i,m': 'i am', 'i,ve': 'i have', 'i;d': 'i would', 'i;ll': 'i will', 'i;m': 'i am', 'i;ve': 'i have',
	"isn't": 'is not',
	'isn,t': 'is not', 'isn;t': 'is not', 'isn´t': 'is not', 'isn’t': 'is not', "it'd": 'it would', "it'll": 'it will',
	"It's": 'it is',
	"it's": 'it is', 'it,d': 'it would', 'it,ll': 'it will', 'it,s': 'it is', 'it;d': 'it would', 'it;ll': 'it will',
	'it;s': 'it is', 'it´d': 'it would', 'it´ll': 'it will', 'it´s': 'it is',
	'it’d': 'it would', 'it’ll': 'it will', 'it’s': 'it is',
	'i´d': 'i would', 'i´ll': 'i will', 'i´m': 'i am', 'i´ve': 'i have', 'i’d': 'i would', 'i’ll': 'i will',
	'i’m': 'i am',
	'i’ve': 'i have', "let's": 'let us', 'let,s': 'let us', 'let;s': 'let us', 'let´s': 'let us',
	'let’s': 'let us', "ma'am": 'madam', 'ma,am': 'madam', 'ma;am': 'madam', "mayn't": 'may not', 'mayn,t': 'may not',
	'mayn;t': 'may not',
	'mayn´t': 'may not', 'mayn’t': 'may not', 'ma´am': 'madam', 'ma’am': 'madam', "might've": 'might have',
	'might,ve': 'might have', 'might;ve': 'might have', "mightn't": 'might not', 'mightn,t': 'might not',
	'mightn;t': 'might not', 'mightn´t': 'might not',
	'mightn’t': 'might not', 'might´ve': 'might have', 'might’ve': 'might have', "must've": 'must have',
	'must,ve': 'must have', 'must;ve': 'must have',
	"mustn't": 'must not', 'mustn,t': 'must not', 'mustn;t': 'must not', 'mustn´t': 'must not', 'mustn’t': 'must not',
	'must´ve': 'must have',
	'must’ve': 'must have', "needn't": 'need not', 'needn,t': 'need not', 'needn;t': 'need not', 'needn´t': 'need not',
	'needn’t': 'need not', "oughtn't": 'ought not', 'oughtn,t': 'ought not', 'oughtn;t': 'ought not',
	'oughtn´t': 'ought not', 'oughtn’t': 'ought not', "sha'n't": 'shall not', 'sha,n,t': 'shall not',
	'sha;n;t': 'shall not', "shan't": 'shall not',
	'shan,t': 'shall not', 'shan;t': 'shall not', 'shan´t': 'shall not', 'shan’t': 'shall not', 'sha´n´t': 'shall not',
	'sha’n’t': 'shall not',
	"she'd": 'she would', "she'll": 'she will', "she's": 'she is', 'she,d': 'she would', 'she,ll': 'she will',
	'she,s': 'she is', 'she;d': 'she would', 'she;ll': 'she will', 'she;s': 'she is', 'she´d': 'she would',
	'she´ll': 'she will',
	'she´s': 'she is', 'she’d': 'she would', 'she’ll': 'she will', 'she’s': 'she is', "should've": 'should have',
	'should,ve': 'should have', 'should;ve': 'should have',
	"shouldn't": 'should not', 'shouldn,t': 'should not', 'shouldn;t': 'should not', 'shouldn´t': 'should not',
	'shouldn’t': 'should not', 'should´ve': 'should have',
	'should’ve': 'should have', "that'd": 'that would', "that's": 'that is', 'that,d': 'that would',
	'that,s': 'that is', 'that;d': 'that would',
	'that;s': 'that is', 'that´d': 'that would', 'that´s': 'that is', 'that’d': 'that would', 'that’s': 'that is',
	"there'd": 'there had',
	"there's": 'there is', 'there,d': 'there had', 'there,s': 'there is', 'there;d': 'there had', 'there;s': 'there is',
	'there´d': 'there had', 'there´s': 'there is', 'there’d': 'there had', 'there’s': 'there is',
	"they'd": 'they would', "they'll": 'they will', "they're": 'they are', "they've": 'they have',
	'they,d': 'they would', 'they,ll': 'they will', 'they,re': 'they are', 'they,ve': 'they have',
	'they;d': 'they would', 'they;ll': 'they will', 'they;re': 'they are',
	'they;ve': 'they have', 'they´d': 'they would', 'they´ll': 'they will', 'they´re': 'they are',
	'they´ve': 'they have', 'they’d': 'they would', 'they’ll': 'they will',
	'they’re': 'they are', 'they’ve': 'they have', "wasn't": 'was not', 'wasn,t': 'was not', 'wasn;t': 'was not',
	'wasn´t': 'was not',
	'wasn’t': 'was not', "we'd": 'we would', "we'll": 'we will', "we're": 'we are', "we've": 'we have',
	'we,d': 'we would', 'we,ll': 'we will',
	'we,re': 'we are', 'we,ve': 'we have', 'we;d': 'we would', 'we;ll': 'we will', 'we;re': 'we are',
	'we;ve': 'we have',
	"weren't": 'were not', 'weren,t': 'were not', 'weren;t': 'were not', 'weren´t': 'were not', 'weren’t': 'were not',
	'we´d': 'we would', 'we´ll': 'we will',
	'we´re': 'we are', 'we´ve': 'we have', 'we’d': 'we would', 'we’ll': 'we will', 'we’re': 'we are',
	'we’ve': 'we have', "what'll": 'what will', "what're": 'what are', "what's": 'what is',
	"what've": 'what have', 'what,ll': 'what will', 'what,re': 'what are', 'what,s': 'what is', 'what,ve': 'what have',
	'what;ll': 'what will', 'what;re': 'what are',
	'what;s': 'what is', 'what;ve': 'what have', 'what´ll': 'what will',
	'what´re': 'what are', 'what´s': 'what is', 'what´ve': 'what have', 'what’ll': 'what will', 'what’re': 'what are',
	'what’s': 'what is',
	'what’ve': 'what have', "where'd": 'where did', "where's": 'where is', 'where,d': 'where did',
	'where,s': 'where is', 'where;d': 'where did',
	'where;s': 'where is', 'where´d': 'where did', 'where´s': 'where is', 'where’d': 'where did', 'where’s': 'where is',
	"who'll": 'who will', "who's": 'who is', 'who,ll': 'who will', 'who,s': 'who is', 'who;ll': 'who will',
	'who;s': 'who is',
	'who´ll': 'who will', 'who´s': 'who is', 'who’ll': 'who will', 'who’s': 'who is', "won't": 'will not',
	'won,t': 'will not', 'won;t': 'will not',
	'won´t': 'will not', 'won’t': 'will not', "wouldn't": 'would not', 'wouldn,t': 'would not', 'wouldn;t': 'would not',
	'wouldn´t': 'would not',
	'wouldn’t': 'would not', "you'd": 'you would', "you'll": 'you will', "you're": 'you are', 'you,d': 'you would',
	'you,ll': 'you will',
	'you,re': 'you are', 'you;d': 'you would', 'you;ll': 'you will',
	'you;re': 'you are', 'you´d': 'you would', 'you´ll': 'you will', 'you´re': 'you are', 'you’d': 'you would',
	'you’ll': 'you will', 'you’re': 'you are',
	'´cause': 'because', '’cause': 'because', "you've": "you have", "could'nt": 'could not',
	"havn't": 'have not', "here’s": "here is", 'i""m': 'i am', "i'am": 'i am', "i'l": "i will", "i'v": 'i have',
	"wan't": 'want', "was'nt": "was not", "who'd": "who would",
	"who're": "who are", "who've": "who have", "why'd": "why would", "would've": "would have", "y'all": "you all",
	"y'know": "you know", "you.i": "you i",
	"your'e": "you are", "arn't": "are not", "agains't": "against", "c'mon": "common", "doens't": "does not",
	'don""t': "do not", "dosen't": "does not",
	"dosn't": "does not", "shoudn't": "should not", "that'll": "that will", "there'll": "there will",
	"there're": "there are",
	"this'll": "this all", "u're": "you are", "ya'll": "you all", "you'r": "you are", "you’ve": "you have",
	"d'int": "did not", "did'nt": "did not", "din't": "did not", "dont't": "do not", "gov't": "government",
	"i'ma": "i am", "is'nt": "is not", "‘I": 'I',
	'ᴀɴᴅ': 'and', 'ᴛʜᴇ': 'the', 'ʜᴏᴍᴇ': 'home', 'ᴜᴘ': 'up', 'ʙʏ': 'by', 'ᴀᴛ': 'at', '…and': 'and',
	'civilbeat': 'civil beat', \
	'TrumpCare': 'Trump care', 'Trumpcare': 'Trump care', 'OBAMAcare': 'Obama care', 'ᴄʜᴇᴄᴋ': 'check', 'ғᴏʀ': 'for',
	'ᴛʜɪs': 'this', 'ᴄᴏᴍᴘᴜᴛᴇʀ': 'computer', \
	'ᴍᴏɴᴛʜ': 'month', 'ᴡᴏʀᴋɪɴɢ': 'working', 'ᴊᴏʙ': 'job', 'ғʀᴏᴍ': 'from', 'Sᴛᴀʀᴛ': 'start', 'gubmit': 'submit',
	'CO₂': 'carbon dioxide', 'ғɪʀsᴛ': 'first', \
	'ᴇɴᴅ': 'end', 'ᴄᴀɴ': 'can', 'ʜᴀᴠᴇ': 'have', 'ᴛᴏ': 'to', 'ʟɪɴᴋ': 'link', 'ᴏғ': 'of', 'ʜᴏᴜʀʟʏ': 'hourly',
	'ᴡᴇᴇᴋ': 'week', 'ᴇɴᴅ': 'end', 'ᴇxᴛʀᴀ': 'extra', \
	'Gʀᴇᴀᴛ': 'great', 'sᴛᴜᴅᴇɴᴛs': 'student', 'sᴛᴀʏ': 'stay', 'ᴍᴏᴍs': 'mother', 'ᴏʀ': 'or', 'ᴀɴʏᴏɴᴇ': 'anyone',
	'ɴᴇᴇᴅɪɴɢ': 'needing', 'ᴀɴ': 'an', 'ɪɴᴄᴏᴍᴇ': 'income', \
	'ʀᴇʟɪᴀʙʟᴇ': 'reliable', 'ғɪʀsᴛ': 'first', 'ʏᴏᴜʀ': 'your', 'sɪɢɴɪɴɢ': 'signing', 'ʙᴏᴛᴛᴏᴍ': 'bottom',
	'ғᴏʟʟᴏᴡɪɴɢ': 'following', 'Mᴀᴋᴇ': 'make', \
	'ᴄᴏɴɴᴇᴄᴛɪᴏɴ': 'connection', 'ɪɴᴛᴇʀɴᴇᴛ': 'internet', 'financialpost': 'financial post', 'ʜaᴠᴇ': ' have ',
	'ᴄaɴ': ' can ', 'Maᴋᴇ': ' make ', 'ʀᴇʟɪaʙʟᴇ': ' reliable ', 'ɴᴇᴇᴅ': ' need ',
	'ᴏɴʟʏ': ' only ', 'ᴇxᴛʀa': ' extra ', 'aɴ': ' an ', 'aɴʏᴏɴᴇ': ' anyone ', 'sᴛaʏ': ' stay ', 'Sᴛaʀᴛ': ' start',
	'SHOPO': 'shop',
}


def clean_contr(x, dic):
	x = x.split()
	x = [dic[word] if word in dic else dic[word.lower()] if word.lower() in dic else word for word in x]
	x = ' '.join(x)
	return x


train_df['comment_text'] = train_df['comment_text'].apply(lambda x: clean_contr(x, contraction_mapping))
test_public_df['comment_text'] = test_public_df['comment_text'].apply(lambda x: clean_contr(x, contraction_mapping))
test_private_df['comment_text'] = test_private_df['comment_text'].apply(lambda x: clean_contr(x, contraction_mapping))

vocab = build_vocab(list(train_df['comment_text'].apply(lambda x: x.split())))
unknown_words = check_coverage(vocab, glove_embeddings)
print("Top 10 Unknown words:")
print(unknown_words[:10])
del vocab
gc.collect()

"""## Correct mispelled words
Correct spelling for a range of known mispelled words taking by https://www.kaggle.com/nz0722/simple-eda-text-preprocessing-jigsaw
"""

mispell_dict = {'SB91': 'senate bill', 'tRump': 'trump', 'utmterm': 'utm term', 'FakeNews': 'fake news',
				'Gʀᴇat': 'great', 'ʙᴏᴛtoᴍ': 'bottom', 'washingtontimes': 'washington times', 'garycrum': 'gary crum',
				'htmlutmterm': 'html utm term', 'RangerMC': 'car', 'TFWs': 'tuition fee waiver',
				'SJWs': 'social justice warrior', 'Koncerned': 'concerned', 'Vinis': 'vinys', 'Yᴏᴜ': 'you',
				'Trumpsters': 'trump', 'Trumpian': 'trump', 'bigly': 'big league', 'Trumpism': 'trump', 'Yoyou': 'you',
				'Auwe': 'wonder', 'Drumpf': 'trump', 'utmterm': 'utm term', 'Brexit': 'british exit',
				'utilitas': 'utilities', 'ᴀ': 'a', '😉': 'wink', '😂': 'joy', '😀': 'stuck out tongue',
				'theguardian': 'the guardian', 'deplorables': 'deplorable', 'theglobeandmail': 'the globe and mail',
				'justiciaries': 'justiciary', 'creditdation': 'Accreditation', 'doctrne': 'doctrine',
				'fentayal': 'fentanyl', 'designation-': 'designation', 'CONartist': 'con-artist',
				'Mutilitated': 'Mutilated', 'Obumblers': 'bumblers', 'negotiatiations': 'negotiations', 'dood-': 'dood',
				'irakis': 'iraki', 'cooerate': 'cooperate', 'COx': 'cox', 'racistcomments': 'racist comments',
				'envirnmetalists': 'environmentalists', }


def correct_spelling(x, dic):
	x = x.split()
	x = [dic[word] if word in dic else dic[word.lower()] if word.lower() in dic else word for word in x]
	x = ' '.join(x)
	return x


train_df['comment_text'] = train_df['comment_text'].apply(lambda x: correct_spelling(x, mispell_dict))
test_public_df['comment_text'] = test_public_df['comment_text'].apply(lambda x: correct_spelling(x, mispell_dict))
test_private_df['comment_text'] = test_private_df['comment_text'].apply(lambda x: correct_spelling(x, mispell_dict))

vocab = build_vocab(list(train_df['comment_text'].apply(lambda x: x.split())))
unknown_words = check_coverage(vocab, glove_embeddings)
print("Top 10 Unknown words:")
print(unknown_words[:10])
del vocab
gc.collect()

"""## Need to fix the punctuation ' in the start """


def del_punctuation_from_start(x, punc):
	x = [word[1:] if word.startswith(punc) else word for word in x.split()]
	x = ' '.join(x)
	return x


train_df['comment_text'] = train_df['comment_text'].apply(lambda x: del_punctuation_from_start(x, "'"))
test_public_df['comment_text'] = test_public_df['comment_text'].apply(lambda x: del_punctuation_from_start(x, "'"))
test_private_df['comment_text'] = test_private_df['comment_text'].apply(lambda x: del_punctuation_from_start(x, "'"))

vocab = build_vocab(list(train_df['comment_text'].apply(lambda x: x.split())))
unknown_words = check_coverage(vocab, glove_embeddings)
print("Top 10 Unknown words:")
print(unknown_words[:10])
del vocab
gc.collect()

"""# Save cleared data sets"""

train_df.to_csv(os.path.join(data_path, 'train_cleared.csv'), index=False)
test_public_df.to_csv(os.path.join(data_path, 'test_public_cleared.csv'), index=False)
test_private_df.to_csv(os.path.join(data_path, 'test_private_cleared.csv'), index=False)