# Load Dataset and Parameter Initialization
import os.path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# import wordcloud
from wordcloud import WordCloud
# import NLTK mainly for stopwords
import nltk
from nltk.corpus import stopwords
from PIL import Image
import emoji
import tensorflow as tf
import argparse

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

"""
We will use only train dataset for EDA and pretrained word vectors from GloVe Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors)."""

TRAIN_DATASET_PATH = os.path.join(data_path, 'train.csv')
TEST_PUBLIC_DATASET_PATH = os.path.join(data_path, 'test_public_expanded.csv')
TEST_PRIVATE_DATASET_PATH = os.path.join(data_path, 'test_private_expanded.csv')
GLOVE_EMBEDDING_PATH = os.path.join(data_path, 'glove.840B.300d.pkl')

"""Read the datasets """

train_df = pd.read_csv(TRAIN_DATASET_PATH)
test_public_df = pd.read_csv(TEST_PUBLIC_DATASET_PATH)
test_private_df = pd.read_csv(TEST_PRIVATE_DATASET_PATH)

"""Create lists with column names """

TARGET_COLUMN = 'target'
GENDER_IDENTITIES = ['male', 'female', 'transgender', 'other_gender']
SEXUAL_ORIENTATION_IDENTITIES = ['heterosexual', 'homosexual_gay_or_lesbian', 'bisexual',
'other_sexual_orientation']
RELIGION_IDENTINTIES = ['christian', 'jewish', 'muslim', 'hindu', 'buddhist',
'atheist', 'other_religion']
RACE_IDENTINTIES = ['black', 'white', 'latino', 'asian',
'other_race_or_ethnicity']
DISABILITY_IDENTINTIES = ['physical_disability','intellectual_or_learning_disability',
                          'psychiatric_or_mental_illness', 'other_disability']
IDENTITY_COLUMNS = GENDER_IDENTITIES + SEXUAL_ORIENTATION_IDENTITIES + RELIGION_IDENTINTIES + RACE_IDENTINTIES + DISABILITY_IDENTINTIES
IDENTITY_COLUMNS = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness'
  ]
AUX_COLUMNS = ['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']
REACTION_COLUMNS = ['funny', 'wow',	'sad',	'likes',	'disagree']

"""  # Exploratory Data Analysis
## Understanding the Data scheme
Print the first 5 samples
"""

train_df = train_df[['id']+["comment_text"]+AUX_COLUMNS+IDENTITY_COLUMNS]
train_df.head()

num_of_samples = len(train_df)
print('Train size: {:d}'.format(num_of_samples))

# check for amount of missing(null) values in every column αnd print the precentage of them.
null_columns=train_df.columns[train_df.isnull().any()]
print(train_df[null_columns].isnull().sum() / num_of_samples * 100)

"""
Check the frequency of lengths of training sentences
"""

# tokenize
tokenizer = tf.keras.preprocessing.text.Tokenizer(lower=False)
tokenizer.fit_on_texts(list(train_df["comment_text"].astype(str)))
x_train = tokenizer.texts_to_sequences(list(train_df["comment_text"].astype(str)))

# count lengths
training_sentence_lengths = [len(tokens) for tokens in x_train]
print("Max sentence length is %s" % max(training_sentence_lengths))
targets_n_lens = pd.DataFrame(data=train_df[TARGET_COLUMN],columns=[TARGET_COLUMN])
targets_n_lens['lens'] = training_sentence_lengths

# plot hist
plt.clf()
plt.hist(training_sentence_lengths, density=True, bins=10)  # arguments are passed to np.histogram
plt.title("Histogram of lenghts of comments")
plt.ylabel('Frequency')
plt.xlabel('Lengths of training comments')
plt.savefig(os.path.join(data_path, "plots/Lengths of training comments.png"))

plt.clf()
plt.hist(targets_n_lens[targets_n_lens[TARGET_COLUMN] >= .5]['lens'].values, density=True, bins=10)  # arguments are passed to np.histogram
plt.title("Histogram of lenghts of toxic comments")
plt.ylabel('Frequency')
plt.xlabel('Lengths of training toxic comments')
plt.savefig(os.path.join(data_path, "plots/Lengths of training toxic comments.png"))

plt.clf()
plt.hist(targets_n_lens[targets_n_lens[TARGET_COLUMN] < .5]['lens'].values, density=True, bins=10)  # arguments are passed to np.histogram
plt.title("Histogram of lenghts of non-toxic comments")
plt.ylabel('Frequency')
plt.xlabel('Lengths of training non-toxic comments')
plt.savefig(os.path.join(data_path, "plots/Lengths of training non-toxic comments.png"))


"""  ## Understanding the Toxic Comments with Identity"""

num_of_non_toxic_samples = int(train_df[train_df.target < 0.5][TARGET_COLUMN].count())
num_of_toxic_samples = num_of_samples - num_of_non_toxic_samples

print('Nummer of samples: ', num_of_samples)
print('Nummer of non-toxic samples: {:d} ,percentage: {:.2f}%'.format(num_of_non_toxic_samples, (
			num_of_non_toxic_samples / num_of_samples) * 100))
print('Nummer of toxic samples: {:d} ,percentage: {:.2f}%'.format(num_of_toxic_samples,
																  (num_of_toxic_samples / num_of_samples) * 100))

"""Let's drop the samples without identity and calculate the previous stats """

train_df_with_identity = train_df.loc[:, AUX_COLUMNS + IDENTITY_COLUMNS].dropna()
num_of_samples_with_identity = len(train_df_with_identity)
num_of_toxic_samples_with_identity = int(
	train_df_with_identity[train_df_with_identity.target >= 0.5][TARGET_COLUMN].count())
num_of_non_toxic_samples_with_identity = num_of_samples_with_identity - num_of_toxic_samples_with_identity

print('Number of samples with identity: {:d}'.format(num_of_samples_with_identity))
print('Nummer of non-toxic samples with identity: {:d} ,percentage: {:.2f}%'.format(
	num_of_non_toxic_samples_with_identity,
	(num_of_non_toxic_samples_with_identity / num_of_samples_with_identity) * 100))
print('Nummer of toxic samples with identity: {:d} ,percentage: {:.2f}%'.format(num_of_toxic_samples_with_identity, (
			num_of_toxic_samples_with_identity / num_of_samples_with_identity) * 100))

"""Count the samples in every subgroup"""

num_of_samples_in_subgroups = train_df_with_identity[(train_df_with_identity[IDENTITY_COLUMNS] > 0)][
	IDENTITY_COLUMNS].count()
num_of_samples_in_subgroups = num_of_samples_in_subgroups.to_frame(name='count')

"""Count the toxic samples in every subgroup"""

# first get all samples that have an identity and are toxic
toxic_samples_with_identity = train_df_with_identity[train_df_with_identity[TARGET_COLUMN] >= 0.5]
num_of__toxic_samples_in_subgroups = toxic_samples_with_identity[toxic_samples_with_identity[IDENTITY_COLUMNS] > 0][
	IDENTITY_COLUMNS].count()
num_of__toxic_samples_in_subgroups = num_of__toxic_samples_in_subgroups.to_frame(name='toxic_count')

subgroup_stats = (num_of__toxic_samples_in_subgroups.toxic_count / num_of_samples_in_subgroups['count']) * 100
subgroup_stats = subgroup_stats.to_frame(name='toxic percentage')
subgroup_stats['toxic'] = num_of__toxic_samples_in_subgroups.toxic_count
subgroup_stats['non_toxic'] = num_of_samples_in_subgroups['count'] - num_of__toxic_samples_in_subgroups.toxic_count
subgroup_stats['total'] = num_of_samples_in_subgroups['count']

subgroup_stats.sort_values(by='toxic', ascending=False, inplace=True)

subgroup_stats[['toxic', 'non_toxic']].plot(kind='bar', stacked=True, figsize=(20, 8), fontsize=15).legend(
	prop={'size': 15})

# multiply each identity with the target
weighted_toxic_percentage = train_df_with_identity[IDENTITY_COLUMNS].apply(
	lambda x: np.asarray(x) * np.asarray(train_df_with_identity[TARGET_COLUMN])).sum()
# devide the number of samples in each subgroup
weighted_toxic_percentage = (weighted_toxic_percentage / num_of_samples_in_subgroups['count']) * 100
weighted_toxic_percentage.sort_values(ascending=False, inplace=True)

plt.clf()
plt.figure(figsize=(15, 8))
sns.set(font_scale=1)
ax = sns.barplot(x=weighted_toxic_percentage.values, y=weighted_toxic_percentage.index)
plt.ylabel('Subgroups')
plt.xlabel('Weighted Toxicity(%)')
plt.savefig(os.path.join(data_path, "plots/toxic samples in every subgroup.png"))


"""## Correlation and Heatmap of Identities"""

rows = [{c: train_df_with_identity[f].corr(train_df_with_identity[c]) for c in [TARGET_COLUMN] + AUX_COLUMNS} for f in
		IDENTITY_COLUMNS]
poptoxicity_correlations = pd.DataFrame(rows, index=IDENTITY_COLUMNS)
plt.clf()
plt.figure(figsize=(12, 8))
sns.set(font_scale=1)
ax = sns.heatmap(poptoxicity_correlations, vmin=-0.1, vmax=0.1, center=0.0)
plt.savefig(os.path.join(data_path, "plots/heatmap.png"))

# Compute the correlation matrix
corr = train_df_with_identity[IDENTITY_COLUMNS].corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Set up the matplotlib figure
sns.set(font_scale=1)
plt.clf()
f, ax = plt.subplots(figsize=(11, 9))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
			square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.savefig(os.path.join(data_path, "plots/corr.png"))
"""The heatmap plot of the correlation between the identities is very insightful. I will summarize my observations below. As always, if you see something interesting please mention it to me in the comment section.
*   It is interesting to see that strong correlations form triangular area at the edge of diagonal.
*   This basically means that there is a strong correlation between the groups of the identity (gender, religion, races, disabilities). This means, the comments where male identity is the target, female identity is also very likely to be the target.
*   In another words, in toxic and non-toxic comments, people tend to make it about one group vs another quiet frequently.
## Word Clouds
"""


# we will write a simple function to generate the wordcloud per identity group
def generate_word_cloud(identity, toxic_comments, non_toxic_comments):
	# convert stop words to sets as required by the wordcloud library
	stop_words = set(stopwords.words("english"))
	# create toxic wordcloud
	toxic_picture = os.path.join(data_path, 'images/toxic_sign.png')
	toxic_mask = np.array(Image.open(toxic_picture))
	toxic_mask = toxic_mask[:, :]
	wordcloud_toxic = WordCloud(max_words=1000, background_color="black", mask=toxic_mask,
								stopwords=stop_words).generate(toxic_comments)
	# create non-toxic wordcloud
	peace_picture = os.path.join(data_path, 'images/peace_sign.png')
	peace_mask = np.array(Image.open(peace_picture))
	wordcloud_non_toxic = WordCloud(max_words=1000, background_color="black", mask=peace_mask,
									stopwords=stop_words).generate(non_toxic_comments)
	# draw the two wordclouds side by side using subplot
	plt.clf()
	fig = plt.figure(figsize=[20, 5])
	fig.add_subplot(1, 2, 1).set_title("Toxic Wordcloud", fontsize=10)
	plt.imshow(wordcloud_toxic, interpolation="bilinear")
	plt.axis("off")
	fig.add_subplot(1, 2, 2).set_title("Non Toxic Wordcloud", fontsize=10)
	plt.imshow(wordcloud_non_toxic, interpolation="bilinear")
	plt.axis("off")
	plt.subplots_adjust(top=0.85)
	plt.suptitle('Word Cloud - {} Identity'.format(identity), size=16)
	plt.savefig(os.path.join(data_path, 'plots/Word Cloud - {} Identity.png'.format(identity)))


"""### White and Black """

identity = 'white'
# get the comments for the given identity
identity_comments = train_df[train_df[identity] > 0][[TARGET_COLUMN, 'comment_text']]
# lets convert the comments as one long string (as needed by wordcloud)
toxic_comments = ' '.join(identity_comments[identity_comments.target >= 0.5]['comment_text'])
non_toxic_comments = ' '.join(identity_comments[identity_comments.target < 0.5]['comment_text'])
# draw the wordcloud using the function we created earlier
generate_word_cloud(identity, toxic_comments, non_toxic_comments)

identity = 'black'
# get the comments for the given identity
identity_comments = train_df[train_df[identity] > 0][[TARGET_COLUMN, 'comment_text']]
# lets convert the comments as one long string (as needed by wordcloud)
toxic_comments = ' '.join(identity_comments[identity_comments.target >= 0.5]['comment_text'])
non_toxic_comments = ' '.join(identity_comments[identity_comments.target < 0.5]['comment_text'])
# draw the wordcloud using the function we created earlier
generate_word_cloud(identity, toxic_comments, non_toxic_comments)

"""### Homosexual """

identity = 'homosexual_gay_or_lesbian'
# get the comments for the given identity
identity_comments = train_df[train_df[identity] > 0][[TARGET_COLUMN, 'comment_text']]
# lets convert the comments as one long string (as needed by wordcloud)
toxic_comments = ' '.join(identity_comments[identity_comments.target >= 0.5]['comment_text'])
non_toxic_comments = ' '.join(identity_comments[identity_comments.target < 0.5]['comment_text'])
# draw the wordcloud using the function we created earlier
generate_word_cloud(identity, toxic_comments, non_toxic_comments)

"""### Muslim and Jewish """

identity = 'muslim'
# get the comments for the given identity
identity_comments = train_df[train_df[identity] > 0][[TARGET_COLUMN, 'comment_text']]
# lets convert the comments as one long string (as needed by wordcloud)
toxic_comments = ' '.join(identity_comments[identity_comments.target >= 0.5]['comment_text'])
non_toxic_comments = ' '.join(identity_comments[identity_comments.target < 0.5]['comment_text'])
# draw the wordcloud using the function we created earlier
generate_word_cloud(identity, toxic_comments, non_toxic_comments)

identity = 'jewish'
# get the comments for the given identity
identity_comments = train_df[train_df[identity] > 0][[TARGET_COLUMN, 'comment_text']]
# lets convert the comments as one long string (as needed by wordcloud)
toxic_comments = ' '.join(identity_comments[identity_comments.target >= 0.5]['comment_text'])
non_toxic_comments = ' '.join(identity_comments[identity_comments.target < 0.5]['comment_text'])
# draw the wordcloud using the function we created earlier
generate_word_cloud(identity, toxic_comments, non_toxic_comments)

"""### Female and Male"""

identity = 'female'
# get the comments for the given identity
identity_comments = train_df[train_df[identity] > 0][[TARGET_COLUMN, 'comment_text']]
# lets convert the comments as one long string (as needed by wordcloud)
toxic_comments = ' '.join(identity_comments[identity_comments.target >= 0.5]['comment_text'])
non_toxic_comments = ' '.join(identity_comments[identity_comments.target < 0.5]['comment_text'])
# draw the wordcloud using the function we created earlier
generate_word_cloud(identity, toxic_comments, non_toxic_comments)

identity = 'male'
# get the comments for the given identity
identity_comments = train_df[train_df[identity] > 0][[TARGET_COLUMN, 'comment_text']]
# lets convert the comments as one long string (as needed by wordcloud)
toxic_comments = ' '.join(identity_comments[identity_comments.target >= 0.5]['comment_text'])
non_toxic_comments = ' '.join(identity_comments[identity_comments.target < 0.5]['comment_text'])
# draw the wordcloud using the function we created earlier
generate_word_cloud(identity, toxic_comments, non_toxic_comments)

"""### Psychiatric or mental illness"""

identity = 'psychiatric_or_mental_illness'
# get the comments for the given identity
identity_comments = train_df[train_df[identity] > 0][[TARGET_COLUMN, 'comment_text']]
# lets convert the comments as one long string (as needed by wordcloud)
toxic_comments = ' '.join(identity_comments[identity_comments.target >= 0.5]['comment_text'])
non_toxic_comments = ' '.join(identity_comments[identity_comments.target < 0.5]['comment_text'])
# draw the wordcloud using the function we created earlier
generate_word_cloud(identity, toxic_comments, non_toxic_comments)


"""The wordcloud above is really interesting. Looking at it, I have made the following observations:
* Although the sentiment within the sentences (probably) varies in toxic and non-toxic comments, looking at it from top word frequencies, the differences are not that big.
* Between comments about White and Black identity, there is a huge overlap!
* Comments towards homosexual have more unique set of words (as imagined) from the other identity groups. However, between toxic and non-toxic comment there isn't a big variation in terms of the high frequenty words.
* For comments about Jewish identity, the word Muslim appears frequently. After reviewing a lot of the samples of such comments I noticed that a large number of comments about Jewish identity is toxic towards Muslim identity.
* Ironically, Trump is a very frequent common topic of discussion in toxic and non-toxic comments. However, frequency of Trump appearing is more in toxic comments. "Trump" or "Trump Supporters" could be a identitity in itself =)
Do you see other interesting patterns in the visualization above? Did I make a mistake? Can I do something better? Write them down in the comment section if possible :)
## Emojis
"""


# we will use this simple function to process a string and return all the emojis as a list
def extract_emojis(str):
	return [c for c in str if c in emoji.UNICODE_EMOJI]


# create a new column to state if  a row / comment has emojis
train_df['emoji_count'] = train_df['comment_text'].apply(lambda x: 1 if len(extract_emojis(x)) > 0 else 0)

emoji_mean_per_identity = []

for identity in IDENTITY_COLUMNS:
	toxic_emoji_mean = train_df[(train_df[identity] > 0) & (train_df[TARGET_COLUMN] >= .5)]['emoji_count'].mean()
	non_toxic_emoji_mean = train_df[(train_df[identity] > 0) & (train_df[TARGET_COLUMN] < .5)]['emoji_count'].mean()
	emoji_mean_per_identity.append([identity, toxic_emoji_mean, non_toxic_emoji_mean])

emoji_mean_per_identity_df = pd.DataFrame(emoji_mean_per_identity,
										  columns=['identity', 'toxic', 'non toxic']).set_index('identity')

# now we can plot our dataframe and see what we have
plt.clf()
emoji_mean_per_identity_df.plot.bar(figsize=(15, 5))
plt.ylabel('mean emojis per comment')
plt.title('Emojis usage in comments for different identities - Normalized')
plt.savefig(os.path.join(data_path, 'plots/Emojis usage in comments for different identities - Normalized.png'))

"""This clears the picture up much better. First of all, the overall use of emoji is pretty low compared to what I see nowaydays. Furthermore, there are definetly a few comments with a rediculous number of emojis which are responsible for skewing our data in the last plot. Finally, as you can imagine; the use of emoji varies and doesn't really differentiate toxic comments and non-toxic comments."""