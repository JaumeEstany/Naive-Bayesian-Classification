import numpy as np

def count_pos_neg_tweets(classif):
    n_pos = 0
    n_neg = 0
    for classif_value in classif:
        if classif_value == 1:
            n_pos +=1
        else:
            n_neg +=1
    
    return n_pos, n_neg
    
def word_dictionary(tweets, classif):
    #dictionary = np.zeros((0,2))
    dictionary = {}
    total_word_app = 0
    for i in xrange(tweets.shape[0]):
        tweet_words = tweets[i].split()  #No es necesario si tenemos los tweets como tuplas de indices
        for word_id in tweet_words:
            aux = dictionary.get(word_id, [0,0])
            aux[classif[i]] += 1
            dictionary[word_id] = aux
            total_word_app += 1
    return dictionary, total_word_app

def word_probability(counted_pos_neg, total_word_app):
    n_positive = counted_pos_neg[1]
    n_negative = counted_pos_neg[0]
    laplace = 1.0       # Podria ser un parametro, pero siempre el mismo para todas las palabras
    prob_positive = (n_positive + laplace)/(total_word_app*(1 + laplace))
    prob_negative = (n_negative + laplace)/(total_word_app*(1 + laplace))
    
    if prob_positive + prob_negative != 1.0:      # Para que sea una probabilidad proporcionada
        alpha = prob_positive + prob_negative
        prob_positive /= alpha
        prob_negative /= alpha
        
    return prob_negative, prob_positive

def tweet_pos_probability(tweet, dictionary, total_word_app, n_pos, n_neg):
    tweet_probability = np.array([np.log(n_neg/(n_pos + n_neg)),np.log(n_pos/(n_pos + n_neg))])
    tweet_words = tweet.split()
    for tweet_word in tweet_words:

        counted_pos_neg = dictionary.get(tweet_word, None)
        if counted_pos_neg is not None:
            tweet_probability += np.log(word_probability(counted_pos_neg, total_word_app))
        else:
            tweet_probability += (1/total_word_app)

    tweet_probability = np.exp(tweet_probability)
    if tweet_probability[0] + tweet_probability[1] != 1:
        alpha = tweet_probability[0] + tweet_probability[1]
        tweet_probability /= alpha
    return tweet_probability
    
