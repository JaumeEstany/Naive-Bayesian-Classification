import file_input as fi
import probability as pb
import methods as met
import numpy as np
import metrics as metr
import time

path = "res/FinalStemmedSentimentAnalysisDataset.csv"

starting_time = time.time()

print 'Started'
tweets, tweet_classif = fi.load_dataset(path)
print 'Read database'

train_ratio = 0.9

tweets_train, classif_train, tweets_test, classif_test = met.holdout_data(tweets, tweet_classif, train_ratio)
print 'Holdout with train ratio = ', train_ratio
n_pos, n_neg = pb.count_pos_neg_tweets(classif_train)
dictionary, total_word_app = pb.word_dictionary(tweets_train, classif_train)

print 'Created Dict'

limiting_num = len(dictionary)             # para NO limitar el diccionario a N palabras
#limiting_num = 2000                        # para limitar el diccionario

limiting_num = int(limiting_num)
if limiting_num != len(dictionary):
    print 'Dictionary Limited to ', limiting_num, ' most frequent words.'
    dictionary = met.get_limited_dictionary_with_n_most_commons(dictionary, limiting_num)

determined_classif = np.zeros(classif_test.shape[0]).astype(int)

acc = 0
i = 0
for curr_tweet in tweets_test:
    prob = pb.tweet_pos_probability(curr_tweet, dictionary, total_word_app, float(n_pos), float(n_neg))
    determined_classif[i] = np.argmax(prob)
    i += 1

resulting_metrics = metr.calculate_all_metrics(classif_test, determined_classif)

metr.print_all_metrics(resulting_metrics)

print ''
print 'Elapsed Time: ', (time.time() - starting_time), 's'