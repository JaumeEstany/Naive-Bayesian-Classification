import numpy as np

max_word_encounters = 10000000000

def holdout_data(tweets, classif, train_ratio=0.8):

    indices = np.arange(tweets.shape[0])
    np.random.shuffle(indices)
    n_train = int(np.floor(tweets.shape[0] * train_ratio))
    indices_train = indices[:n_train]
    indices_test = indices[n_train:]
    tweets_train = tweets[indices_train]
    classif_train = classif[indices_train]
    tweets_test = tweets[indices_test]
    classif_test = classif[indices_test]

    return tweets_train, classif_train, tweets_test, classif_test


# a partir de aqui "best" significa mas frecuente
# si pone "min_best" significa el menos frecuente de los N mas frecuentes
def search_new_min_best_value_index(best_array):
    index = 0
    value = max_word_encounters

    for i in xrange(best_array.shape[0]):
        curr_val = best_array[i][1]
        if curr_val < value:
            index = i
            value = curr_val

    return index

def get_limited_dictionary_with_n_most_commons(dictionary, size):

    if size > len(dictionary):
        size = len(dictionary)
        print 'Size to limit too large, limited to ',len(dictionary),'.'

    best_array = np.empty(size, dtype=object)
    for i in xrange(best_array.shape[0]):
        best_array[i] = ('', -1)

    min_best_value = 0
    min_best_value_index = -1

    i = 0
    for word, apparition_tuple in dictionary.iteritems():
        curr_sum = apparition_tuple[0]+apparition_tuple[1]
        if curr_sum > min_best_value:

            best_array[min_best_value_index] = (word, curr_sum)
            min_best_value_index = search_new_min_best_value_index(best_array)
            min_best_value = best_array[min_best_value_index][1]
        i += 1

    ret_dict = {}
    for elem in best_array:
        curr_word = elem[0]
        ret_dict[curr_word] = dictionary[curr_word]

    print 'Limited dict'
    return ret_dict

