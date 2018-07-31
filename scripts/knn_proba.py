"""
Implements kNN classifier that uses longest common subsequence to find k-nearest
training sentences for an unknown sentence to predict its label.
Instead of just returning the predicted label, the classifier
returns confidence probabilities for each class.

Probabilities are calculated according to:
https://stats.stackexchange.com/questions/83600/how-to-obtain-the-class-conditional-probability-when-using-knn-classifier

That means for an unknown instance x, we get the k nearest labeled neighbors
and the probability we assign for each class c (given there are C classes):

proba(x) = (n + s) / (k+C)

where n is the number of the k instances that share the majority label, s being
a smoothing factor to avoid probabilities to be 0.
Probabilities sum up to 1.

"""
import string
import warnings
from collections import Counter
from operator import itemgetter

import editdistance

import pseudo_db


# Value representing a missing label
# NA = " "


def longest_common_substring(s1, s2):
    m = [[0] * (1 + len(s2)) for _ in xrange(1 + len(s1))]
    longest, x_longest = 0, 0
    for x in xrange(1, 1 + len(s1)):
        for y in xrange(1, 1 + len(s2)):
            if s1[x - 1] == s2[y - 1]:
                m[x][y] = m[x - 1][y - 1] + 1
                if m[x][y] > longest:
                    longest = m[x][y]
                    x_longest = x
            else:
                m[x][y] = 0
    return s1[x_longest - longest: x_longest]


def longest_common_substring_sentence(s1, s2):
    """
    Computes the longest common SUBSTRING (= subsequent words that occur at
    contiguous positions in both sentences) between two sentences s1 and s2.
    IMPORTANT: returns normalized number of matches, so that string length
    doesn't affect result, e.g. s1 = "hello hello foo bar", s2 =
    "hello foo bar", s3 = "hello hello foo foo bar bar"; LCS is always
    "hello foo bar", but s1 = s2, so it should have a higher score than
    s1 with s3 or s2 with s3.
    Thus, divide by number of words in longer sentence.
    https://stackoverflow.com/questions/22726177/longest-common-substring-without-cutting-a-word-python

    Returns
    -------
    List of str, float.
    List of shared words, normalized LCS distance. Normalization is implemented
    by dividing len(list of shared words) / max(len(s1, s2), i.e. distance is
    within (and including) [0,1].

    """
    s1_words = s1.split(' ')
    s2_words = s2.split(' ')
    res = " ".join(longest_common_substring(s1_words, s2_words))
    words = 0
    # If there's a shared word (.split(" ")) would yield 1 even if there's no
    # shared word
    if len(res) > 0:
        words = len(res.split(" "))
    normalized = 1.0 * words / max(len(s1_words), len(s2_words))
    return res, normalized


def lcs(s1, s2):
    """https://stackoverflow.com/questions/24547641/python-length-of-longest-common-subsequence-of-lists"""
    table = [[0] * (len(s2) + 1) for _ in xrange(len(s1) + 1)]
    for i, ca in enumerate(s1, 1):
        for j, cb in enumerate(s2, 1):
            table[i][j] = (
                table[i - 1][j - 1] + 1 if ca == cb else
                max(table[i][j - 1], table[i - 1][j]))
    return table[-1][-1]


def longest_common_sentence(s1, s2):
    """
    Computes the longest common subsequence (= words that occur in both
    sentences in the same order, but not necessarily in contiguous order)
    between two sentences s1 and s2.
    IMPORTANT: returns normalized number of matches, so that string length
    doesn't affect result, e.g. s1 = "hello hello foo bar", s2 =
    "hello foo bar", s3 = "hello hello foo foo bar bar"; LCS is always
    "hello foo bar", but s1 = s2, so it should have a higher score than
    s1 with s3 or s2 with s3.
    Thus, divide by number of words in longer sentence.

    Returns
    -------
    float.
    Normalized LCS distance. Normalization is implemented by dividing
    LCS / max(len(s1, s2), i.e. distance is within (and including) [0,1].

    """
    s1_words = s1.split(' ')
    s2_words = s2.split(' ')
    words = lcs(s1_words, s2_words)
    normalized = 1.0 * words / max(len(s1_words), len(s2_words))
    return normalized


def edit_sentence(s1, s2):
    """
    Computes the edit distance between two sentences s1 and s2.
    IMPORTANT: returns normalized number of matches, so that string length
    doesn't affect result.
    Thus, divide by number of words in longer sentence.

    Returns
    -------
    float.
    Normalized edit distance. Normalization is implemented by dividing
    edit_value / max(len(s1, s2), i.e. distance is within (and including) [0,1].

    """
    s1_words = s1.split(' ')
    s2_words = s2.split(' ')
    words = editdistance.eval(s1_words, s2_words)
    normalized = 1.0 * words / max(len(s1_words), len(s2_words))
    return normalized


def preprocess_text(s):
    """
    Removes punctuation from a text as well as converts everything to lower
    case. Also removes newlines and tabs.
    https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string-in-python

    Parameters
    ----------
    s: str - text.

    Returns
    -------
    str.
    Preprocessed text.

    """
    # Replace newlines with whitespace
    s = s.replace("\n", " ").replace("\r", "")
    # Replace tabs with whitespace
    s = s.replace("\t", " ")
    # Make sure there's only 1 whitespace between each word
    s = " ".join(s.split())
    return s.translate(None, string.punctuation)


def knn(x, y, x_test, k=1, weights="uniform", recency=False, keep_na=False,
        similarity="substring"):
    """
    Implements a kNN classifier.

    Scenarios to be tested:
    -----------------------
    1. Your idea: Annotation by similarity
    When an annotator sees a tweet, it assigns to it the majority label of the
    kNN tweets labeled thus far.
    I suggest to set k=1 to begin with. This is not unnatural, I do not believe
    that anyone is able to remember e.g the k=10 most similar tweets.

    2. Alternative: Annotation by recency
    When an annotator sees a tweet, it assigns to it the label of the most
    recent similar tweet.
    The algo is:
    - sort all past tweets on similarity and rank them, smaller numbers for
    higher similarity, e.g. 3, 2, 5, 4, 1
    - rank all past tweets on recency, smaller numbers for higher recency,
    e.g. 1, 2, 3, 4, 5
    - choose the most similar tweet where the recency rank is not larger than
    the similarity rank, ie. 2 in the concrete case.

    Parameters
    ----------
    x: list of str - training texts.
    y: list of str - labels of <x>.
    x_test: list of str - test texts without labels.
    k: int - number of neighbors to consider for predicting labels.
    weights: str - weight function used in prediction.  Possible values:
        - "uniform" : uniform weights.  All points in each neighborhood
          are weighted equally.
        - "distance" : weight points by the inverse of their distance.
          in this case, closer neighbors of a query point will have a
          greater influence than neighbors which are further away.
    recency: bool - True if scenario 2 should be tested. Otherwise scenario 1
    is tested.
    keep_na: bool - True if neighbors with no label should be considered when
    predicting the label of an unknown instance. If False, such instances with
    unavailable labels are discarded.
    similarity: str - similarity function to be used:
        - "substring": longest common SUBSTRING (= contiguously shared words)
        - "subsequence": longest common SUBSEQUENCE (= shared words in relative
        order, not necessarily contiguous)
        - "edit": edit distance

    Returns
    -------
    List of list of tuples.
    Inner lists represent a single tweet, and each tuple contains the class
    label and the probability that the instance belongs to that class.
    Labels for <x_test> in same order.

    Raises
    ------
    ValueError if number of training instances != number of labels.

    """
    # Store most similar neighbors for a  tweet
    max_dist = 1
    t = ""
    if len(x) != len(y):
        raise ValueError("Different number of training sentences and labels!")
    # Discard all training texts with NA labels
    if not keep_na:
        x, y = _remove_na_texts(x, y)
    y_ts = []
    # Predict label for each test instance
    for x_t in x_test:
        # Compute distances to all points and retrieve
        neighbor_idxs, dists = _get_neighbors(x, x_t, k, recency, similarity)
        neighbor_labels = []
        # Print labels of k-nearest neighbors
        # print "Labels of neighbors:"
        # Get labels of neighbors
        for idx in neighbor_idxs:
            neighbor_labels.append(y[idx])
            # print y[idx]
        # Store most similar tweets
        if dists[0] < max_dist:
            neighs = []
            for idx in neighbor_idxs:
                neighs.append(x[idx])
            max_dist = dists[0]
        y_t = _predict_label(neighbor_labels, dists, weights)
        y_ts.append(y_t)
    # We only want some samples for k=3
    # if max_dist < 1 < k < 5:
    #     print "{} most similar: {}".format(k, max_dist)
    #     print "for tweet:", t
    #     print "neighbors:", neighs
    #     print "labels:", labels
    return y_ts


def _remove_na_texts(x, y):
    """
    Removes all training texts for which no label exists (=NA).

    Parameters
    ----------
    x: list of str - training texts.
    y: list of str - labels for <x>, in same order.

    Returns
    -------
    list of str, list of str.
    Returns filtered <x> and <y>.

    """
    x_ = []
    y_ = []
    for t, l in zip(x, y):
        if l != pseudo_db.NA:
            x_.append(t)
            y_.append(l)
    return x_, y_


def _get_neighbors(x, test, k, recency, similarity):
    """
    Retrieve the k closest neighbors from the training set for a test instance.
    NOTE: if recency == True, the number of neighbors might be smaller than k!

    Parameters
    ----------
    x: list of str - list of training instances.
    test: str - test instance.
    k: number of neighbors to retrieve.
    recency: bool - True if scenario 2 should be tested. Otherwise scenario 1
    is tested. See knn() for explanation of both scenarios.
    similarity: str - similarity function to be used:
        - "substring": longest common SUBSTRING (= contiguously shared words)
        - "subsequence": longest common SUBSEQUENCE (= shared words in relative
        order, not necessarily contiguous)
        - "edit": edit distance

    Returns
    -------
    List of int, list of float.
    Indices of the k nearest training instances, distances of those k nearest
    instances sorted ascendingly, s.t. nearest tweet has minimum distance.

    """
    n = len(x)
    if n < k:
        warnings.warn("Only {} neighbors are used for predicting the label, "
                      "but {} neighbors were supposed to be used.".
                      format(n, k))
    dists = []
    # Compute pairwise distances from test sentence with all training sentences
    for inst in x:
        if similarity == "substring":
            _, words = longest_common_substring_sentence(inst, test)
        elif similarity == "subsequence":
            words = longest_common_sentence(inst, test)
        else:
            words = edit_sentence(inst, test)
            # Large value for word indicate that many words had to be switched,
            # implying dissimilarity. Small value for word means similarity.
            # This is opposite to the 2 other distance functions where large
            # values indicate similarity and low values dissimilarity. Thus,
            # we subtract the normalized value from 1. If the result is 0, it
            # now indicates that there was no similarity between words.
            words = 1.0 - words
        # words is [0,1] and 1 indicates perfect similarity, so the distance
        # should be 0.
        dist = 1 - words
        # If many words are shared among two sentences, their distance is small
        # if words > 0:
        #     # dist = 1.0 / words
        # else:
        #     # DO NOT set it to 0, but a really small value instead because
        #     # otherwise it won't have any contribution in the weighting scheme
        #     # "distance".
        #     dist = 0.01
        # print "compare '{}' with '{}' = {} {}".format(inst, test,
        #                                               shared, words)
        dists.append(dist)
    # Sort w.r.t. distances, s.t. closest (smallest distance) sentences come
    # first, but we just need the indices
    sorted_idxs = [i[0] for i in sorted(enumerate(dists), key=itemgetter(1))]
    sorted_dists = []
    if recency:
        # {sentence_id1: recency rank}
        recency_mapping = {}
        # More recently labeled sentences have lower ranks
        recency_rank = n - 1
        for idx, sentence in enumerate(x):
            recency_mapping[idx] = recency_rank
            recency_rank -= 1
        # print "recency ranks", recency_mapping
        j = 0
        # Assign similarity rank, more similar sentences have lower values
        # {sim_rank1: sentence_id, }
        sim_mapping = {}
        for idx, sentence_id in enumerate(sorted_idxs):
            sim_mapping[sentence_id] = idx
        # print "sim rank", sim_mapping
        recency_idxs = []
        # Select the j most similar sentences that were labeled sufficiently
        # recently
        for sentence_idx in sorted_idxs:
            # Stop if we already found the k most similar sentences
            if j == k:
                break
            recency_rank = recency_mapping[sentence_idx]
            sim_rank = sim_mapping[sentence_idx]
            # Choose the most similar sentence where its recency rank is not
            # larger than the similarity rank
            if sim_rank >= recency_rank:
                recency_idxs.append(sentence_idx)
                j += 1
        # print "recency idxs", recency_idxs
        for idx in recency_idxs:
            sorted_dists.append(dists[idx])
        return recency_idxs, sorted_dists
    else:
        # Print infsci2017_results (by uncommenting)to get an intuitive feeling about
        # neighbors
        # print "{}NN for '{}':".format(k, repr(test))
        # for idx in sorted_idxs[:k]:
        #     print x[idx]
        for idx in sorted_idxs[:k]:
            sorted_dists.append(dists[idx])
        return sorted_idxs[:k], sorted_dists


def _predict_label(ys, dists, weights, C=2):
    """
    Predicts the probabilities of all classes for an unknown sentence given
    its neighbor labels, the
    neighbors' distances to that point and the weighting scheme how to aggregate
    the labels. Predicted label is determined by majority voting. Ignores labels
    that are NA, i.e. not available.

    Probabilities are calculated according to:
    https://stats.stackexchange.com/questions/83600/how-to-obtain-the-class-conditional-probability-when-using-knn-classifier

    That means for an unknown instance x, we get the k nearest labeled neighbors
    and the probability we assign for each class c (given there are C classes):

    proba(x) = (n + s) / (k+C)

    where n is the number of the k instances that share the majority label, s
    being a smoothing factor to avoid probabilities to be 0.
    Probabilities sum up to 1.

    Parameters
    ----------
    ys: List of str - labels of neighbors, first entry is most similar neighbor.
    dists: List of float - distances to neighbors w.r.t. unknown sentence,
    same order as <ys>. Higher values mean higher distance and lower similarity.
    weights: str - "uniform" or "distance" to determine how to aggregate <ys>.
    C - int - number of classes that could be predicted.

    Returns
    -------
    List of tuples.
    Each tuple contains a class label and its probability:
    (class_label, probability). The list is sorted, s.t. the most likely class
    is at the 0-th index.

    Raises
    ------
    ValueError if unknown label aggregation scheme is selected.

    """
    k = len(ys)
    probas = []
    if weights == "uniform":
        probas_ = []
        # Variable to count probabilities - used to make sure that they sum
        # up to 1 over all classes
        total = 0.0
        counts = Counter(ys)
        # Sort descendingly w.r.t. occurrences of class labels, i.e. first
        # entry occurs most frequently
        common = counts.most_common()
        if len(common) > 1:
            first, second = common[0][1], common[1][1]
            # If the two most occurring neighbor labels have the same frequency
            if first == second:
                msg = "Tie: {} vs. {}, so the assigned label is random"\
                    .format(common[0], common[1])
                warnings.warn(msg)
                # print "chosen", common[0][0]
        # Compute probabilities for each class label
        for entry in common:
            res = compute_proba(k, C, entry)
            probas_.append(res)
            total += res[1]
        # Make sure that probabilities sum up to 1 (not guaranteed until now)
        for label, prob in probas_:
            probas.append((label, 1.0*prob/total))
        return probas
    elif weights == "distance":
        counts = {}
        # k is the sum of all weighted counts
        k = 0
        for y, dist in zip(ys, dists):
            if y not in counts:
                counts[y] = 0
            weight = 1.0 / dist
            counts[y] += weight
            k += weight
        sorted_counts = sorted(counts.items(), key=itemgetter(1), reverse=True)
        if len(sorted_counts) > 1:
            first, second = sorted_counts[0][1], sorted_counts[1][1]
            # If the two most occurring neighbor labels have the same frequency
            if first == second:
                msg = "Tie: {} vs. {}, so the assigned label is random"\
                    .format(sorted_counts[0], sorted_counts[1])
                warnings.warn(msg)
                # print "chosen", sorted_counts[0][1]
        # Compute probabilities for each class label
        for entry in sorted_counts:
            probas.append(compute_proba(k, C, entry))
        return probas
    else:
        raise ValueError("Label aggregation scheme must be either 'uniform' or "
                         "'distance'!")


def compute_proba(k, C, entry):
    """
    Computes probability for a class label.

    Parameters
    ----------
    k: int - number of neighbors.
    C: int - number of classes that can be predicted
    entry: - tuple - (class label, frequency of how many neighbors have
    that label).

    Returns
    -------
    tuple.
    (class label, probability). Probability represents the certainty of kNN
    assigning the given class label to the unknown tweet.

    """
    # Smoothing factor
    s = 1
    n = entry[1]
    proba = 1.0*(n + s) / (k + C)
    res = (entry[0], proba)
    return res


######################
# Evaluation metrics #
######################
def accuracy(predicted, truth):
    """
    Computes accuracy for predicted labels.

    Parameters
    ----------
    predicted: List of str - list of predicted labels.
    truth: List of str - list of true labels.

    Returns
    -------
    float.
    Accuracy of predictions.

    Raises
    ------
    ValueError if not the same number of entries exist in both input lists.

    """
    if len(predicted) != len(truth):
        raise ValueError("Different number of entries in the input lists!")
    correct = 0
    for y_pred, y_truth in zip(predicted, truth):
        if y_pred == y_truth:
            correct += 1
    return 1.0*correct / len(predicted)


def precision_recall_f1(predicted, truth, label):
    """
    Computes precision, recall, and F1 score for predicted labels. Assumes
    binary classification (because otherwise, the denominator must be changed).
    Applies smoothing to calculations to avoid 0 as outcome for recall and
    precision.

    Parameters
    ----------
    predicted: List of str - list of predicted labels.
    truth: List of str - list of true labels.
    label: str - the true positive label.

    Returns
    -------
    float, float, float.
    Precision, recall, and F1 score of predictions.

    Raises
    ------
    ValueError if not the same number of entries exist in both input lists.
    ValueError if a TP label isn't given.

    """
    # https://stats.stackexchange.com/questions/83600/how-to-obtain-the-class-conditional-probability-when-using-knn-classifier
    # Smoothing value
    s = 1
    if len(predicted) != len(truth):
        raise ValueError("Different number of entries in the input lists!")
    if len(label) == 0:
        raise ValueError("No true positive label specified!")
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for y_pred, y_truth in zip(predicted, truth):
        if y_pred == y_truth and label == y_pred:
            tp += 1
        if y_pred == y_truth and label != y_pred:
            tn += 1
        if y_pred != y_truth and label == y_pred:
            fp += 1
        if y_pred != y_truth and label != y_pred:
            fn += 1
    recall = 1.0*(tp+s) / (tp + fn + s)
    precision = 1.0*(tp+s) / (tp + fp + s)
    f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1


##############################################################################
# Metrics for hierarchical evaluation, see "Learning and Evaluation in
# the Presence of Class Hierarchies: Application to Text Categorization" by
# Kiritchenko et al., 2006
# We use hierarchical precision (hP)/recall (HR)/f1:
# a) Predicted class: "R.1", True class: "R.1.2"
# hP = 1/1, hR = 1/2
# b) Predicted class: "R.1", True class: "R.1.2.1"
# hP = 1/1, hR = 1/3
# c) Predicted class: "R.2.2", True class: "R.2"
# hP = 1/2, hR = 1/1
# d) Predicted class: "R.2.2.1", True class: "R.2"
# hP = 1/3, hR = 1/1
###############################################################################
def hierarchical_precision_recall_f1(predicted, truth):
    """
    Computes hierarchical precision, recall, and F1 score for predicted labels
    given the true labels.
    IMPORTANT: Unavailable labels represented by NA don't affect the infsci2017_results.
    Uses the definitions of "Learning and Evaluation in
    # the Presence of Class Hierarchies: Application to Text Categorization" by
    # Kiritchenko et al., 2006

    Parameters
    ----------
    predicted: List of list of str - each inner list holds all predicted labels
    over all hierarchy levels for a tweet.
    truth: List of list of str - each inner list holds all true labels
    over all hierarchy levels for a tweet.

    Returns
    -------
    float, float, float, Counter.
    Hierarchical precision, recall, and F1 score of predictions. The dictionary
    contains the statistics to calculate micro-averaged scores later on.

    """
    micro_stats = Counter({
        "correct": 0,
        "total_preds": 0,
        "total_truths": 0
    })
    # We have true and predicted labels for each tweet, so it doesn't matter
    # over which one we iterate
    for i, y_pred in enumerate(predicted):
        # Get the list of correct labels for that tweet
        y_truth = truth[i]
        # Get the list of predicted labels for that tweet. If there's no label
        # or only more labels for <truth>, indicate it's missing so that it
        # won't affect the score
        # y_pred = utility.NA
        # if i < len(predicted):
        #     y_pred = predicted[i]
        # Number of predicted labels that match with true labels
        correct = 0
        # Total number of predicted labels
        total_preds = 0
        # Total number of true labels
        total_truths = 0
        # Iterate either over <y_pred> or <y_true> depending on which one has
        # more labels that are not missing
        labels = max(len(y_pred), len(y_truth))
        # For each instance (tweet), we have 3 labels, but they could
        # potentially be missing
        # Check each predicted label with the true label for each hierarchy
        # level
        for level in xrange(labels):
            # Get the true label for that level. If there's no label or
            # only more labels for <y_pred>, indicate it's missing so that it
            # won't affect the score
            y_t = pseudo_db.NA
            if level < len(y_truth):
                y_t = y_truth[level]
            # Get the predicted label for that level. If there's no label or
            # only more labels for <y_truth>, indicate it's missing so that it
            # won't affect the score
            y_p = pseudo_db.NA
            if level < len(y_pred):
                y_p = y_pred[level]
            # print "pred at lvl {}: {} correct: {}".format(level, y_p, y_t)
            # Correct prediction and it's not a missing label
            if y_p == y_t and y_p != pseudo_db.NA:
                correct += 1
            # Increase count if the predicted label isn't missing
            if y_p != pseudo_db.NA:
                total_preds += 1
            # Increase count if true label isn't missing
            if y_t != pseudo_db.NA:
                total_truths += 1
        # total_preds += len(y_pred)
        # total_truths += len(y_truth)
        # Update micro statistics
        micro_stats["correct"] += correct
        micro_stats["total_preds"] += total_preds
        micro_stats["total_truths"] += total_truths
    rec = 0.0
    prec = 0.0
    f1 = 0.0
    # If no label is assigned/predicted, denominator could be 0
    if micro_stats["total_truths"] > 0:
        rec = 1.0*micro_stats["correct"] / micro_stats["total_truths"]
    if micro_stats["total_preds"] > 0:
        prec = 1.0*micro_stats["correct"] / micro_stats["total_preds"]
    # If all predicted labels are wrong, precision and recall would be 0
    if prec > 0 or rec > 0:
        f1 = 2 * (prec * rec) / (prec + rec)
    return prec, rec, f1, micro_stats


def compute_micro_metrics(counts):
    """
    Computes micro-averaged hierarchical precision, recall, and F1-score from
    given counts.

    Parameters
    ----------
    counts: list of Counter - each dictionary contains as keys "correct",
    "total_preds", "total_truths" used for the computation of the metrics.

    Returns
    -------
    float, float, float.
    Micro-averaged hierarchical precision, recall, and F1-score.

    """
    # Sum up the counters, i.e. the statistics, across all dicts
    final = sum(counts, Counter())
    prec = 1.0*final["correct"] / final["total_preds"]
    rec = 1.0*final["correct"] / final["total_truths"]
    f1 = 2 * (prec * rec) / (prec + rec)
    return prec, rec, f1


#########
# Tests #
#########
def test_normalized_lcsubstring():
    """tests that normalization of LCSubstring works"""
    s1 = "hello hello foo bar"
    s2 = "hello foo bar"
    s3 = "hello hello foo foo bar bar"
    s4 = "blub bla blum"
    shared1, words1 = longest_common_substring_sentence(s1, s2)
    # print "s1 vs. s2:", shared1, words1
    shared2, words2 = longest_common_substring_sentence(s1, s3)
    # print "s1 vs. s3:", shared2, words2
    shared3, words3 = longest_common_substring_sentence(s2, s3)
    # print "s2 vs. s3:", shared3, words3
    shared4, words4 = longest_common_substring_sentence(s1, s4)
    assert(words1 == 0.75 and words2 == 0.5 and 0.33 < words3 < 0.34 and words4 == 0)
    # s1 = "Could give a SHIT what his tax returns are...I care more about MY tax returns...and the lack of a refund in the last four yrs #debatenight"
    # s2 = "There are things we can do &amp; we should find a way to do it'\n\n'What do you suggest?'\n\n'That\'s not my job, fuck you'\n\n#debates #debates2016"
    # shared, words = longest_common_substring_sentence(s1, s2)
    # print "shared", shared, words
    # assert(words == 26.0)

def test_normalized_lcsubsequence():
    """tests that normalization of LCSubsequence works"""
    s1 = "hi ho yo boho bub"
    s2 = "hi yo ho x y bub is zu"
    s3 = "foo my name is"
    words1 = longest_common_sentence(s1, s2)
    print "s1 vs. s2:", words1
    words2 = longest_common_sentence(s1, s3)
    print "s1 vs. s3:", words2
    words3 = longest_common_sentence(s2, s3)
    print "s2 vs. s3:", words3
    assert(words1 == 0.375 and words2 == 0 and words3 == 0.125)


def test_normalized_edit():
    """tests that normalization of edit distance works"""
    s1 = "ab cd ef gh"
    s2 = "xx ab cd ef gh"
    s3 = "ab x y cd ef gh"
    s4 = "ab x y cd ef gh"
    words1 = edit_sentence(s1, s2)
    print "s1 vs. s2:", words1
    words2 = edit_sentence(s1, s3)
    print "s1 vs. s3:", words2
    words3 = edit_sentence(s2, s3)
    print "s2 vs. s3:", words3
    words4 = edit_sentence(s3, s4)
    print "s3 vs. s4:", words4
    assert(words1 == 0.2 and 0.33 < words2 < 0.34 and words3 == 0.5 and
           words4 == 0)


def test_preprocessing_lcs():
    """tests that removing punctuation, newlines, tabs from sentences works"""
    s1 = "this is,; a\n\n\r foo bar\t sentence.\"!"
    s2 = "this is a foo bar sentence"
    s1 = preprocess_text(s1)
    s2 = preprocess_text(s2)
    common_sentence, shared_words = longest_common_substring_sentence(s1, s2)
    assert(shared_words == 1)
    s2 = ""
    s2 = preprocess_text(s2)
    common_sentence, shared_words = longest_common_substring_sentence(s1, s2)
    assert(shared_words == 0)


def test_recency_knn():
    """test that k nearest sentences are found for scenario 2"""
    s1 = "this is,; a\n\n\r foo bar\t sentence.\"!"
    s2 = "what a kappa foo bar black sheep?"
    s3 = "rt what a kappa foo bar black sheep?"
    s4 = "ho bar ho ho!"
    s_test = "is a foo bar"
    s1 = preprocess_text(s1)
    s2 = preprocess_text(s2)
    s3 = preprocess_text(s3)
    s4 = preprocess_text(s4)
    idxs, _ = _get_neighbors([s3, s1, s4, s2], s_test, k=2, recency=True,
                             similarity="substring")
    assert(idxs == [3, 2])
    idxs, _ = _get_neighbors([s2, s1, s3, s4], s_test, k=2, recency=True,
                             similarity="substring")
    assert(idxs == [2, 3])
    idxs, _ = _get_neighbors([s3, s2, s1, s4], s_test, k=2, recency=True,
                             similarity="substring")
    assert(idxs == [3])
    idxs, _ = _get_neighbors([s4, s3, s2, s1], s_test, k=2, recency=True,

                             similarity="substring")
    assert(idxs == [3, 2])
    idxs, _ = _get_neighbors([s2, s1, s4, s3], s_test, k=2, recency=True,

                             similarity="substring")
    assert(idxs == [2, 3])


def test_similarity_knn():
    """test that k nearest sentences are found for scenario 1"""
    s1 = "this is,; a\n\n\r foo bar\t sentence.\"!"
    s2 = "what a kappa foo bar black sheep?"
    s3 = "rt what a kappa foo bar black sheep?"
    s4 = "ho bar ho ho!"
    s_test = "is a foo bar"
    s1 = preprocess_text(s1)
    s2 = preprocess_text(s2)
    s3 = preprocess_text(s3)
    s4 = preprocess_text(s4)
    idxs, _ = _get_neighbors([s2, s1, s3, s4], s_test, k=2, recency=False,
                             similarity="substring")
    assert(idxs == [1, 0])
    idxs, _ = _get_neighbors([s2, s1, s3, s4], s_test, k=2, recency=False,
                             similarity="substring")
    assert(idxs == [1, 0])
    idxs, _ = _get_neighbors([s3, s2, s1, s4], s_test, k=2, recency=False,
                             similarity="substring")
    assert(idxs == [2, 1])
    idxs, _ = _get_neighbors([s4, s3, s2, s1], s_test, k=2, recency=False,
                             similarity="substring")
    assert(idxs == [3, 2])
    idxs, _ = _get_neighbors([s2, s1, s4, s3], s_test, k=2, recency=False,
                             similarity="substring")
    assert(idxs == [1, 0])


def test_neighbor_recency_dist():
    """test that correct distance is calculated for scenario 2"""
    s1 = "this is,; a\n\n\r foo bar\t sentence.\"!"
    s2 = "what a kappa foo bar black sheep?"
    s3 = "rt what a kappa foo bar black sheep?"
    s4 = "ho bar ho ho!"
    s_test = "is a foo bar"
    s1 = preprocess_text(s1)
    s2 = preprocess_text(s2)
    s3 = preprocess_text(s3)
    s4 = preprocess_text(s4)
    idxs, dists = _get_neighbors([s3, s1, s4, s2], s_test, k=2, recency=True,
                                 similarity="substring")
    print "dists", dists
    # s2 and s4 are neighbors
    _, shared1 = longest_common_substring_sentence(s_test, s2)
    _, shared2 = longest_common_substring_sentence(s_test, s4)
    dist1 = 1.0 - shared1
    dist2 = 1.0 - shared2
    assert(dists[0] == dist1 and dists[1] == dist2)
    idxs, dists = _get_neighbors([s2, s1, s3, s4], s_test, k=2, recency=True,
                                 similarity="substring")
    _, shared1 = longest_common_substring_sentence(s_test, s3)
    _, shared2 = longest_common_substring_sentence(s_test, s4)
    dist1 = 1.0 - shared1
    dist2 = 1.0 - shared2
    assert(dists[0] == dist1 and dists[1] == dist2)
    idxs, dists = _get_neighbors([s3, s2, s1, s4], s_test, k=2, recency=True,
                                 similarity="substring")
    _, shared1 = longest_common_substring_sentence(s_test, s4)
    dist1 = 1.0 - shared1
    assert(dists[0] == dist1)
    idxs, dists = _get_neighbors([s4, s3, s2, s1], s_test, k=2, recency=True,
                                 similarity="substring")
    _, shared1 = longest_common_substring_sentence(s_test, s1)
    _, shared2 = longest_common_substring_sentence(s_test, s2)
    dist1 = 1.0 - shared1
    dist2 = 1.0 - shared2
    assert(dists[0] == dist1 and dists[1] == dist2)
    idxs, dists = _get_neighbors([s2, s1, s4, s3], s_test, k=2, recency=True,
                                 similarity="substring")
    _, shared1 = longest_common_substring_sentence(s_test, s4)
    _, shared2 = longest_common_substring_sentence(s_test, s3)
    dist1 = 1.0 - shared1
    dist2 = 1.0 - shared2
    assert(dists[0] == dist1 and dists[1] == dist2)


def test_similarity_dist():
    """test that correct distance is calculated for scenario 1"""
    s1 = "this is,; a\n\n\r foo bar\t sentence.\"!"
    s2 = "what a kappa foo bar black sheep?"
    s3 = "rt what a kappa foo bar black sheep?"
    s4 = "ho bar ho ho!"
    s5 = "this is,; a\n\n\r foo bar\t sentence.\"!"
    s_test = "is a foo bar"
    s1 = preprocess_text(s1)
    s2 = preprocess_text(s2)
    s3 = preprocess_text(s3)
    s4 = preprocess_text(s4)
    s5 = preprocess_text(s5)
    _, shared1 = longest_common_substring_sentence(s_test, s1)
    _, shared2 = longest_common_substring_sentence(s_test, s2)
    dist1 = 1.0 - shared1
    dist2 = 1.0 - shared2
    _, dists = _get_neighbors([s2, s1, s3, s4], s_test, k=2, recency=False,
                              similarity="substring")
    print dists[0], dist1, dists[1], dist2
    assert(dists[0] == dist1 and dists[1] == dist2)
    _, dists = _get_neighbors([s2, s1, s3, s4], s_test, k=2, recency=False,
                              similarity="substring")
    assert(dists[0] == dist1 and dists[1] == dist2)
    _, dists = _get_neighbors([s3, s2, s1, s4], s_test, k=2, recency=False,
                              similarity="substring")
    assert(dists[0] == dist1 and dists[1] == dist2)
    _, dists = _get_neighbors([s4, s3, s2, s1], s_test, k=2, recency=False,
                              similarity="substring")
    assert(dists[0] == dist1 and dists[1] == dist2)
    _, dists = _get_neighbors([s2, s1, s4, s3], s_test, k=2, recency=False,
                              similarity="substring")
    assert(dists[0] == dist1 and dists[1] == dist2)
    _, dists = _get_neighbors([s2, s1, s4, s3, s5], s_test, k=2, recency=False,
                              similarity="substring")


def test_neighbor_uniform_weight():
    """test that  distances are correctly aggregated"""
    s1 = "this is,; a\n\n\r foo bar\t sentence.\"!"
    s2 = "what a kappa foo bar black sheep?"
    s3 = "rt what a kappa foo bar black sheep?"
    s4 = "ho bar ho ho!"
    s5 = "is a foo bar"
    s_test = "is a foo bar"
    s1 = preprocess_text(s1)
    s2 = preprocess_text(s2)
    s3 = preprocess_text(s3)
    s4 = preprocess_text(s4)
    s5 = preprocess_text(s5)
    train = [s3, s1, s4, s2, s5]
    labels = ["a", "b", "b", "b", pseudo_db.NA]
    idxs, dists = _get_neighbors(train, s_test, k=3, recency=False,
                                 similarity="substring")
    neighbor_labels = []
    # Get labels of neighbors
    for idx in idxs:
        neighbor_labels.append(labels[idx])
    y_t = _predict_label(neighbor_labels, dists, weights="uniform", C=2)
    print y_t
    assert(y_t[0][0] == "b")

    train = [s3, s1, s5, s4, s2]
    labels = ["a", "a", pseudo_db.NA, "b", "b"]
    idxs, dists = _get_neighbors(train, s_test, k=3, recency=False,
                                 similarity="substring")
    neighbor_labels = []
    # Get labels of neighbors
    for idx in idxs:
        neighbor_labels.append(labels[idx])
    y_t = _predict_label(neighbor_labels, dists, weights="uniform", C=3)
    print y_t
    assert(y_t[0][0] == "a")

    train = [s5, s2, s1, s3, s4]
    labels = [pseudo_db.NA, "a", "d", "b", "c"]
    idxs, dists = _get_neighbors(train, s_test, k=3, recency=False,
                                 similarity="substring")
    neighbor_labels = []
    # Get labels of neighbors
    for idx in idxs:
        neighbor_labels.append(labels[idx])
    y_t = _predict_label(neighbor_labels, dists, weights="uniform", C=4)
    print y_t
    assert(y_t[0][0] == "a")

    # Handle NA labels - prediction yields NA
    train = [s3, s1, s4, s2, s5]
    labels = ["a", "a", pseudo_db.NA, pseudo_db.NA, pseudo_db.NA]
    idxs, dists = _get_neighbors(train, s_test, k=3, recency=False,
                                 similarity="substring")
    neighbor_labels = []
    # Get labels of neighbors
    for idx in idxs:
        neighbor_labels.append(labels[idx])
    y_t = _predict_label(neighbor_labels, dists, weights="uniform", C=2)
    print y_t
    assert(y_t[0][0] == pseudo_db.NA)

    # Remove NA labels and have too few neighbors
    train = [s3, s1, s4, s2, s5]
    labels = ["a", "a", pseudo_db.NA, pseudo_db.NA, pseudo_db.NA]
    # Discard any training instances with NA labels
    train, labels = _remove_na_texts(train, labels)
    idxs, dists = _get_neighbors(train, s_test, k=3, recency=False,
                                 similarity="substring")
    print "NA distances", idxs, dists
    neighbor_labels = []
    # Get labels of neighbors
    for idx in idxs:
        neighbor_labels.append(labels[idx])
    y_t = _predict_label(neighbor_labels, dists, weights="uniform", C=2)
    print y_t
    assert(y_t[0][0] == "a")

    # Test that NA is predicted
    train = [s3, s1, s4, s2, s5]
    labels = ["a", "a", pseudo_db.NA, "a", pseudo_db.NA]
    idxs, dists = _get_neighbors(train, s_test, k=3, recency=False,
                                 similarity="substring")
    print "NA distances", idxs, dists
    neighbor_labels = []
    # Get labels of neighbors
    for idx in idxs:
        neighbor_labels.append(labels[idx])
    y_t = _predict_label(neighbor_labels, dists, weights="uniform", C=2)
    print y_t
    assert(y_t[0][0] == "a")


def test_neighbor_distance_weight():
    """test that  distances are correctly aggregated"""
    s1 = "this is,; a\n\n\r foo bar\t sentence.\"!"
    s2 = "what a kappa foo bar black sheep?"
    s3 = "rt what a kappa foo bar black sheep?"
    s4 = "ho bar ho ho!"
    s_test = "is a foo bar"
    s1 = preprocess_text(s1)
    s2 = preprocess_text(s2)
    s3 = preprocess_text(s3)
    s4 = preprocess_text(s4)
    train = [s3, s1, s4, s2]
    labels = ["a", "b", "b", "b"]
    idxs, dists = _get_neighbors(train, s_test, k=3, recency=False,
                                 similarity="substring")
    print idxs, dists
    neighbor_labels = []
    # Get labels of neighbors
    for idx in idxs:
        neighbor_labels.append(labels[idx])
    print neighbor_labels
    y_t = _predict_label(neighbor_labels, dists, weights="distance", C=2)
    print y_t
    assert(y_t[0][0] == "b")

    train = [s3, s1, s4, s2]
    labels = ["b", "a", "b", "b"]
    idxs, dists = _get_neighbors(train, s_test, k=4, recency=False,
                                 similarity="substring")
    neighbor_labels = []
    # Get labels of neighbors
    for idx in idxs:
        neighbor_labels.append(labels[idx])
    y_t = _predict_label(neighbor_labels, dists, weights="distance", C=2)
    print y_t
    assert(y_t[0][0] == "b")

    train = [s2, s1, s3, s4]
    labels = ["a", "d", "b", "c"]
    idxs, dists = _get_neighbors(train, s_test, k=3, recency=False,
                                 similarity="substring")
    neighbor_labels = []
    # Get labels of neighbors
    for idx in idxs:
        neighbor_labels.append(labels[idx])
    y_t = _predict_label(neighbor_labels, dists, weights="distance", C=3)
    print y_t
    assert(y_t[0][0] == "d")


def test_accuracy():
    """test accuracy calculations"""
    truth = ["a", "b", "c"]

    predicted = ["b", "b", "b"]
    acc = accuracy(predicted, truth)
    assert(0.33 < acc < 0.34)

    truth = ["a", "b", "c"]
    predicted = ["a", "b", "c"]
    acc = accuracy(predicted, truth)
    assert(acc == 1)

    truth = ["a", "b", "c"]
    predicted = ["a", "b", "a"]
    acc = accuracy(predicted, truth)
    assert(0.66 < acc < 0.67)


def test_precision_recall_f1():
    """test precision, recall, and f1 score calculations"""
    truth = ["a", "b", "a", "b", "a", "a", "b", "b"]
    labels = "a"
    predicted = ["b", "b", "b", "a", "a", "b", "a", "b"]

    # Some predictions are correct
    prec, rec, f1 = precision_recall_f1(predicted, truth, labels)
    # assert(prec == 0.4 and 0.33 < rec < 0.34 and 0.36 < f1 < 0.37)
    assert(prec == 0.5 and rec == 0.4 and 0.44 < f1 < 0.45)

    # All predictions are correct
    predicted = ["a", "b", "a", "b", "a", "a", "b", "b"]
    prec, rec, f1 = precision_recall_f1(predicted, truth, labels)
    assert(prec == 1 and rec == 1 and f1 == 1)

    # All predictions are wrong
    predicted = ["b", "a", "b", "a", "b", "b", "a", "a"]
    prec, rec, f1 = precision_recall_f1(predicted, truth, labels)
    # WTF: although f1 = 0.2, f1 == 0.2 yields False, thus rounding applied
    assert(prec == 0.2 and rec == 0.2 and round(f1, 1) == 0.2)


def test_hierarchical_precision_recall_f1():
    """test hierarchical precision, recall, and f1 score calculations"""
    # Examples from Sila, "A Survey of Hierarchical Classification Across
    # Different Application Domains", 2009, page 28
    # Varying recall with a single instance
    truth = [["1", "2"]]
    predicted = [["1"]]
    prec, rec, f1, _ = hierarchical_precision_recall_f1(predicted, truth)
    assert(prec == 1 and rec == 0.5 and 0.66 < f1 < 0.67)
    truth = [["1", "2", "1"]]
    predicted = [["1"]]
    prec, rec, f1, _ = hierarchical_precision_recall_f1(predicted, truth)
    assert(prec == 1 and 0.33 < rec < 0.34 and f1 == 0.5)
    truth = [["1", "2", "1", "1"]]
    predicted = [["1"]]
    prec, rec, f1, _ = hierarchical_precision_recall_f1(predicted, truth)
    assert(prec == 1 and rec == 0.25 and f1 == 0.4)
    # Check that unavailable labels don't affect metrics
    truth = [["1", "2", "1", "1", pseudo_db.NA]]
    predicted = [["1"]]
    prec1, rec1, f11, _ = hierarchical_precision_recall_f1(predicted, truth)
    assert(prec == prec1 and rec == rec1 and f1 == f11)
    truth = [["1", "2", "1", "1", pseudo_db.NA]]
    predicted = [["1", pseudo_db.NA, pseudo_db.NA]]
    prec1, rec1, f11, _ = hierarchical_precision_recall_f1(predicted, truth)
    assert(prec == prec1 and rec == rec1 and f1 == f11)

    # Varying precision with a single instance
    truth = [["2"]]
    predicted = [["2", "2"]]
    prec, rec, f1, _ = hierarchical_precision_recall_f1(predicted, truth)
    assert(prec == 0.5 and rec == 1 and 0.66 < f1 < 0.67)
    truth = [["2"]]
    predicted = [["2", "2", "1"]]
    prec, rec, f1, _ = hierarchical_precision_recall_f1(predicted, truth)
    assert(0.33 < prec < 0.34 and rec == 1 and f1 == 0.5)
    truth = [["2"]]
    predicted = [["2", "2", "1", "3"]]
    prec1, rec1, f11, _ = hierarchical_precision_recall_f1(predicted, truth)
    assert(prec1 == 0.25 and rec1 == 1 and f11 == 0.4)
    # Check that unavailable labels don't affect metrics
    truth = [["2"]]
    predicted = [["2", "2", "1", "3", pseudo_db.NA]]
    prec, rec, f1, _ = hierarchical_precision_recall_f1(predicted, truth)
    assert(prec == prec1 and rec == rec1 and f1 == f11)

    # Test what happens if all predictions are wrong
    truth = [["1"]]
    predicted = [["2", "2", "1", "3", pseudo_db.NA]]
    prec, rec, f1, _ = hierarchical_precision_recall_f1(predicted, truth)
    assert(prec == 0 and rec == 0 and f1 == 0)

    # Test what happens if all labels are missing
    truth = [[pseudo_db.NA]]
    predicted = [[pseudo_db.NA, pseudo_db.NA]]
    prec, rec, f1, _ = hierarchical_precision_recall_f1(predicted, truth)
    assert(prec == 0 and rec == 0 and f1 == 0)


def test_hierarchical_metrics_na():
    """test how metrics behave when some labels are not assigned by annos"""
    s1 = "this is,; a\n\n\r foo bar\t sentence.\"!"
    s2 = "what a kappa foo bar black sheep?"
    s3 = "rt what a kappa foo bar black sheep?"
    s4 = "ho bar ho ho!"
    s5 = "is a foo bar"
    s_test = "is a foo bar"
    s1 = preprocess_text(s1)
    s2 = preprocess_text(s2)
    s3 = preprocess_text(s3)
    s4 = preprocess_text(s4)
    s5 = preprocess_text(s5)
    train = [s3, s1, s4, s2, s5]
    labels = ["a", "b", "b", "b", pseudo_db.NA]
    idxs, dists = _get_neighbors(train, s_test, k=3, recency=False,
                                 similarity="substring")
    neighbor_labels = []
    # Get labels of neighbors
    for idx in idxs:
        neighbor_labels.append(labels[idx])
    y_t = _predict_label(neighbor_labels, dists, weights="uniform")
    print y_t
    assert(y_t[0][0] == "b")

    train = [s3, s1, s5, s4, s2]
    labels = ["a", "a", pseudo_db.NA, "b", "b"]
    idxs, dists = _get_neighbors(train, s_test, k=3, recency=False,
                                 similarity="substring")
    neighbor_labels = []
    # Get labels of neighbors
    for idx in idxs:
        neighbor_labels.append(labels[idx])
    y_t = _predict_label(neighbor_labels, dists, weights="uniform")
    assert(y_t[0][0] == "a")

    train = [s5, s2, s1, s3, s4]
    labels = [pseudo_db.NA, "a", "d", "b", "c"]
    idxs, dists = _get_neighbors(train, s_test, k=3, recency=False,
                                 similarity="substring")
    neighbor_labels = []
    # Get labels of neighbors
    for idx in idxs:
        neighbor_labels.append(labels[idx])
    y_t = _predict_label(neighbor_labels, dists, weights="uniform")
    assert(y_t[0][0] == "a")

    # Handle NA labels - prediction yields NA
    train = [s3, s1, s4, s2, s5]
    labels = ["a", "a", pseudo_db.NA, pseudo_db.NA, pseudo_db.NA]
    idxs, dists = _get_neighbors(train, s_test, k=3, recency=False,
                                 similarity="substring")
    neighbor_labels = []
    # Get labels of neighbors
    for idx in idxs:
        neighbor_labels.append(labels[idx])
    y_t = _predict_label(neighbor_labels, dists, weights="uniform")
    assert(y_t[0][0] == pseudo_db.NA)

    # Remove NA labels and have too few neighbors
    train = [s3, s1, s4, s2, s5]
    labels = ["a", "a", pseudo_db.NA, pseudo_db.NA, pseudo_db.NA]
    # Discard any training instances with NA labels
    train, labels = _remove_na_texts(train, labels)
    idxs, dists = _get_neighbors(train, s_test, k=3, recency=False,
                                 similarity="substring")
    neighbor_labels = []
    # Get labels of neighbors
    for idx in idxs:
        neighbor_labels.append(labels[idx])
    y_t = _predict_label(neighbor_labels, dists, weights="uniform")
    assert(y_t[0][0] == "a")

    # Test that NA is predicted
    train = [s3, s1, s4, s2, s5]
    labels = ["a", "a", pseudo_db.NA, "a", pseudo_db.NA]
    idxs, dists = _get_neighbors(train, s_test, k=3, recency=False,
                                 similarity="substring")
    neighbor_labels = []
    # Get labels of neighbors
    for idx in idxs:
        neighbor_labels.append(labels[idx])
    y_t = _predict_label(neighbor_labels, dists, weights="uniform")
    assert(y_t[0][0] == "a")


def test_micro_macro_metrics():
    """test if micro/macro averaged precision, recall, f1 score are correct"""
    # Vary recall
    truth = [["1", "2"], ["1", "2", "1"], ["1", "2", "1", "1"]]
    predicted = [["1"], ["1"], ["1"]]
    macro_rec = []
    macro_prec = []
    macro_f1 = []
    prec, rec, f1, micro1 = hierarchical_precision_recall_f1(predicted, truth)
    print prec, rec, f1, micro1
    macro_prec.append(prec)
    macro_rec.append(rec)
    macro_f1.append(f1)
    truth = [["1", "2"], ["1", "2", "1"], ["1", "2", "1", "1"]]
    predicted = [["1", "2"], ["1"], ["1"]]
    prec, rec, f1, micro2 = hierarchical_precision_recall_f1(predicted, truth)
    print prec, rec, f1, micro2
    macro_prec.append(prec)
    macro_rec.append(rec)
    macro_f1.append(f1)
    # Compute macro metrics over 2 annotators
    prec = 1.0*sum(macro_prec) / len(macro_prec)
    rec = 1.0*sum(macro_rec) / len(macro_rec)
    f1 = 1.0*sum(macro_f1) / len(macro_f1)
    print "macro", prec, rec, f1

    # Compute micro metrics over 2 annotators
    prec = 1.0*(micro1["correct"] + micro2["correct"]) / \
           (micro1["total_preds"] + micro2["total_preds"])
    rec = 1.0*(micro1["correct"] + micro2["correct"]) / \
          (micro1["total_truths"] + micro2["total_truths"])
    f1 = 2 * (prec * rec) / (prec + rec)
    print "micro", prec, rec, f1
    assert(prec == 1 and 0.38 < rec < 0.39 and f1 == 0.56)
    p, r, f = compute_micro_metrics([micro1, micro2])
    assert(p == prec and r == rec and f == f1)

    # Vary precision
    truth = [["2"], ["2"], ["2"]]
    predicted = [["2", "2"], ["2", "2", "1"], ["2", "2", "1", "3"]]
    macro_rec = []
    macro_prec = []
    macro_f1 = []
    prec, rec, f1, micro1 = hierarchical_precision_recall_f1(predicted, truth)
    print prec, rec, f1, micro1
    macro_prec.append(prec)
    macro_rec.append(rec)
    macro_f1.append(f1)
    truth = [["2", "2"], ["2"], ["2"]]
    predicted = [["2", "2"], ["2", "2", "1"], ["2", "2", "1", "3"]]
    prec, rec, f1, micro2 = hierarchical_precision_recall_f1(predicted, truth)
    print prec, rec, f1, micro2
    macro_prec.append(prec)
    macro_rec.append(rec)
    macro_f1.append(f1)
    # Compute macro metrics over 2 annotators
    prec = 1.0*sum(macro_prec) / len(macro_prec)
    rec = 1.0*sum(macro_rec) / len(macro_rec)
    f1 = 1.0*sum(macro_f1) / len(macro_f1)
    print "macro", prec, rec, f1

    # Compute micro metrics over 2 annotators
    prec = 1.0*(micro1["correct"] + micro2["correct"]) / \
           (micro1["total_preds"] + micro2["total_preds"])
    rec = 1.0*(micro1["correct"] + micro2["correct"]) / \
          (micro1["total_truths"] + micro2["total_truths"])
    f1 = 2 * (prec * rec) / (prec + rec)
    print "micro", prec, rec, f1
    assert(0.38 < prec < 0.39 and rec == 1 and f1 == 0.56)
    p, r, f = compute_micro_metrics([micro1, micro2])
    assert(p == prec and r == rec and f == f1)


def test_na():
    """tests that knn discards any instances with NA labels"""
    s1 = "this is,; a\n\n\r foo bar\t sentence.\"!"
    s2 = "what a kappa foo bar black sheep?"
    s3 = "rt what a kappa foo bar black sheep?"
    s4 = "ho bar ho ho!"
    s5 = "this is,; a\n\n\r foo bar\t sentence.\"!"
    s_test = "is a foo bar"
    s1 = preprocess_text(s1)
    s2 = preprocess_text(s2)
    s3 = preprocess_text(s3)
    s4 = preprocess_text(s4)
    s5 = preprocess_text(s5)
    _, dist1 = longest_common_substring_sentence(s_test, s1)
    _, dist2 = longest_common_substring_sentence(s_test, s2)
    x, y = _remove_na_texts([s1, s2, s3, s4, s5], ["a", "b", "a", "b",
                                                   pseudo_db.NA])
    assert(x == [s1, s2, s3, s4] and pseudo_db.NA not in y)


if __name__ == "__main__":
    debug = False
    if debug:
        test_normalized_lcsubstring()
        test_normalized_lcsubsequence()
        test_normalized_edit()
        test_preprocessing_lcs()
        test_recency_knn()
        test_similarity_knn()
        test_neighbor_recency_dist()
        test_similarity_dist()
        test_neighbor_uniform_weight()
        test_neighbor_distance_weight()
        test_accuracy()
        test_precision_recall_f1()
        test_na()
        test_hierarchical_precision_recall_f1()
        test_micro_macro_metrics()
        test_hierarchical_metrics_na()
    else:
        s1 = "this is,; a\n\n\r foo bar\t sentence.\"!"
        s2 = "what a kappa foo bar black sheep?"
        s3 = "rt what a kappa foo bar black sheep?"
        s4 = "ho bar ho ho!"
        s_test = "is a foo bar"
        s1 = preprocess_text(s1)
        s2 = preprocess_text(s2)
        s3 = preprocess_text(s3)
        s4 = preprocess_text(s4)
        common_sentence, shared_words = \
            longest_common_substring_sentence(s1, s2)
        print common_sentence, shared_words
        common_sentence, shared_words = \
            longest_common_substring_sentence(s1, s4)
        print common_sentence, shared_words
        common_sentence, shared_words = \
            longest_common_substring_sentence(s_test, s4)
        print common_sentence, shared_words
        print knn([s2, s1, s3, s4], ["a", "b", "b", "b"], [s_test], k=2,
                  recency=False)
        print knn([s2, s1, s3, s4], ["a", "b", "b", "b"], [s_test], k=2,
                  recency=True)
