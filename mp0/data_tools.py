"""Processing data tools for mp0.
"""
import re
import numpy as np


def title_cleanup(data):
    """Remove all characters except a-z, A-Z and spaces from the title,
       then convert all characters to lower case.

    Args:
        data(dict): Key: article_id(int),
                    Value: [title(str), positivity_score(float)](list)
    """
    for key, val in data.items():
        old_title = val[0]
        middle_result = re.sub("[^a-zA-Z ]", "", old_title)
        data[key][0] = middle_result.lower()
    return data


def most_frequent_words(data):
    """Find the more frequeny words (including all ties), returned in a list.

    Args:
        data(dict): Key: article_id(int),
                    Value: [title(str), positivity_score(float)](list)
    Returns:
        max_words(list): List of strings containing the most frequent words.
    """
    max_words = []
    all_words = [word for line in data.values() for word in line[0].split()]
    unique_words = list(set(all_words))
    counts = [all_words.count(word) for word in unique_words]
    maximum_count = max(counts)
    just_words = [word for i, word in enumerate(unique_words) if counts[i] == maximum_count]
    max_words = [value[0] for value in data.values() \
                 for word in just_words \
                 if word in value[0].split()]
    
    return max_words


def most_positive_titles(data):
    """Computes the most positive titles.
    Args:
        data(dict): Key: article_id(int),
                    Value: [title(str), positivity_score(float)](list)
    Returns:
        titles(list): List of strings containing the most positive titles,
                      include all ties.
    """
    titles = []
    maximum_score = float("-inf")
    for value in data.values():
        if value[1] > maximum_score:
            titles = [value[0]]
            maximum_score = value[1]
        elif value[1] == maximum_score:
            titles.append(value[0])
    
    return titles


def most_negative_titles(data):
    """Computes the most negative titles.
    Args:
        data(dict): Key: article_id(int),
                    Value: [title(str), positivity_score(float)](list)
     Returns:
        titles(list): List of strings containing the most negative titles,
                      include all ties.
    """
    titles = []
    minimum_score = float("inf")
    for value in data.values():
        if value[1] < minimum_score:
            titles = [value[0]]
            minimum_score = value[1]
        elif value[1] == minimum_score:
            titles.append(value[0])

    return titles


def compute_word_positivity(data):
    """Computes average word positivity.
    Args:
        data(dict): Key: article_id(int),
                    Value: [title(str), positivity_score(float)](list)
    Returns:
        word_dict(dict): Key: word(str), value: word_index(int)
        word_avg(numpy.ndarray): numpy array where element
                                 #word_dict[word] is the
                                 average word positivity for word.
    """
    word_dict = {}
    
    dict_ = {}
    word_avg = None
    for line in data.values():
        words = line[0].split()
        for word in words:
            if not dict_.get(word):
                dict_[word] = [line[1]]
            else:
                dict_[word].append(line[1])
   
    idx, word_score_list, word_count_list = 0, [], []
    for key, val in dict_.items():
        word_dict[key] = idx
        idx += 1
        word_score_list.append(sum(val))
        word_count_list.append(len(val))

    word_score = np.array(word_score_list)
    word_count = np.array(word_count_list)

    word_avg = word_score / word_count
    
    return word_dict, word_avg


def most_postivie_words(word_dict, word_avg):
    """Computes the most positive words.
    Args:
        word_dict(dict): output from compute_word_positivity.
        word_avg(numpy.ndarray): output from compute_word_positivity.
    Returns:
        words(list):
    """
    words = []
    most = np.argwhere(word_avg == max(word_avg))
    most = most.flatten().tolist()
    words = [key for key, val in word_dict.items() if val in most]

    return words


def most_negative_words(word_dict, word_avg):
    """Computes the most negative words.
    Args:
        word_dict(dict): output from compute_word_positivity.
        word_avg(numpy.ndarray): output from compute_word_positivity.
    Returns:
        words(list):
    """
    words = []
    most = np.argwhere(word_avg == min(word_avg))
    most = most.flatten().tolist()
    words = [key for key, val in word_dict.items() if val in most]

    return words
