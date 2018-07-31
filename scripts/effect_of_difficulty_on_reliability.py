"""
Run simulation to test how labeling reliability is affected by tweet difficulty:
- perform a hierarchical classification experiment on sentiment analysis
- we build 6 classifiers per annotator due to no ground truth: 2 classifiers per
level and we have 3 hierarchy levels (relevant/irrelevant, factual/non-factual,
positive/negative)
- Each annotator labeled 50 tweets in total (well, we only consider her first 50
 tweets) - 25 tweets in early and 25 tweets in late stage
- STAGE could be either early or late
- for training sets of size i:
    - build classifier 1, called EASY:
        - use first i easy tweets from STAGE to train EASY
        - use remaining 25-i tweets for testing
    - build classifier 2, called DIFFICULT:
        - use first i difficult tweets from STAGE to train DIFFICULT
        - use remaining 25-i tweets for testing (i.e. from all STAGE)
    - evaluation according to hierarchical F1-score
- Results:
    - are there any patterns? i.e. in general, higher F1-scores with EASY than
    DIFFICULT? is LATE and EASY perhaps even better than EASY and EARLY?
- expectation: EASY outperforms DIFFICULT (i.e. higher F1-score) because easy
tweets are less ambiguous and more meaningful to the classifier

"""
import matplotlib.pyplot as plt
import operator
import os
from collections import Counter
from copy import deepcopy

import pseudo_db
from anno import Annotator
from knn import knn, hierarchical_precision_recall_f1, compute_micro_metrics

EMPTY = ""
ZERO = 0


FONTSIZE = 12
plt.rcParams.update({'font.size': FONTSIZE})

# Transition from early to late stage
THRESHOLD = 25


def run_simulation(dbs, db_idxs, inst_name, stat_dir, threshold_dir, label_dir,
                   k_max, i, anno_coll_name="user", tweet_coll_name="tweets",
                   cleaned=False, min_annos=1, train_ratio=0.66, is_early=True,
                   is_easy=True, weights="uniform", recency=False,
                   keep_na=True, use_n=50, use_n_threshold=True):
    """
    Simulates if labeling reliability in a specific stratum is affected by
    tweet difficulty. Possible strata are:
    - easy tweets in early stage
    - difficult tweets in early stage
    - easy tweets in late stage
    - difficult tweets in late stage
    Build 6 classifiers as described at the top of the script
    for each annotator.

    Parameters
    ----------
    dbs: list of strings - names of the existing DBs
    db_idxs: list of ints - name of the MongoDB from where data should be read.
    inst_name: str - name of the institution for which i is computed.
    stat_dir: str - directory in which text files are stored.
    threshold_dir: str - directory containing thresholds for early/late
    annotation stage.
    label_dir: str - directory containing tweet difficulty labels.
    k_max: int - maximum number of neighbors to consider when predicting label.
    It starts always with 1.
    i: int - number of tweets to be used for training.
    anno_coll_name: str - name of the collection holding the annotator data.
    tweet_coll_name: str - name of the collection holding the tweet data.
    cleaned: bool - True if the data should be cleaned, i.e. if tweet is
    "irrelevant", its remaining labels are ignored for computing average
    annotation times.
    min_annos: int - minimum number of annotators who must've assigned a label
    to a tweet. Otherwise it'll be discarded.
    train_ratio: float - between 0-1, specifies how many percent of the
    tweets should be used for training/testing.
    is_early: bool - True if only tweets from the early phase should be
    considered. Else only tweets from the late stage are considered.
    is_easy: bool - True if tweet difficulty is easy. Else it's difficult.
    weights: str - weight function used in prediction.  Possible values:
        - "uniform" : uniform weights.  All points in each neighborhood
          are weighted equally.
        - "distance" : weight points by the inverse of their distance.
          in this case, closer neighbors of a query point will have a
          greater influence than neighbors which are further away.
    recency: bool - True if scenario 2 should be tested. Otherwise scenario 1
    is tested. See knn() in knn.py for an explanation of the 2 scenarios.
    keep_na: bool - True if neighbors with no label should be considered when
    predicting the label of an unknown instance. If False, such instances with
    unavailable labels are discarded.
    use_n: int - considers only the first n tweets per annotator, regardless of
    annotator group.
    use_n_threshold: bool - True if early/late stage threshold over S, M, L for
    first <use_n> tweets should be used. Else the thresholds over all tweets of
    S and M is are used.

    """
    print "Institution: {}".format(inst_name)
    print "----------------"
    stage = "late"
    if is_early:
        stage = "early"
    difficulty = "difficult"
    if is_easy:
        difficulty = "easy"
    agg = "raw"
    if cleaned:
        agg = "cleaned"
        # Thresholds over all tweets of S+M annotators
    # if not use_n_threshold:
    #     fname = "{}_early_late_thresholds_without_l_{}.txt".format(inst_name,
    #                                                                agg)
    # # Thresholds over first <use_n> tweets of S+M+L annotators
    # else:
    #     fname = "{}_early_late_thresholds_use_n_{}_{}.txt".format(inst_name,
    #                                                               use_n, agg)
    # dst = os.path.join(threshold_dir, fname)
    # try:
    #     thresholds = read_thresholds(dst, inst_name)
    # except IOError:
    #     raise IOError("First run [deprecated]split_into_early_late_stage.py with proper " +
    #                   "parameters! Otherwise this script won't work!")
    thresholds = {
        "S": THRESHOLD,
        "M": THRESHOLD,
        "L": THRESHOLD,
        inst_name: THRESHOLD
    }
    print "Now process stratum '{} tweets from {} stage'".format(difficulty,
                                                                 stage)
    ################
    # 1. Load data #
    ################
    inst_annos, group_annos = read_dataset(
        dbs, db_idxs, inst_name, thresholds, anno_coll_name=anno_coll_name,
        tweet_coll_name=tweet_coll_name, cleaned=cleaned, min_annos=min_annos,
        is_early=is_early)
    if use_n_threshold:
        l_fname = "{}_min_annos_{}_difficulty_labels_early_train_ratio_{}_use" \
                "_n_{}_{}.txt"\
                .format(inst_name, min_annos, train_ratio, use_n, agg)
    else:
        l_fname = "{}_min_annos_{}_difficulty_labels_early_train_ratio_{}" \
                  "_{}.txt"\
            .format(inst_name, min_annos, train_ratio, agg)
    print "diff label dir", l_fname
    label_dst = os.path.join(label_dir, l_fname)
    print "annos", len(inst_annos)
    ###################
    # For institution #
    ###################
    # a) Get annotators who labeled at least i easy/difficult tweets in stage
    # early/late
    _, filtered_inst_annos = \
        count_annos_with_at_least_i_tweets(label_dst, inst_annos, i,
                                           is_easy=is_easy)
    print "available annotators in this stratum:", len(filtered_inst_annos)
    # b) Hierarchical learning
    # For each similarity measure
    for similarity in ["substring", "subsequence", "edit"]:
        # {
        #         k1: (precision, recall, f1_score),
        #         k2: (precision, recall, f1_score)
        #
        # }
        data = {}
        # List of Counter objects - each counter holds statistics used to
        # compute hierarchical F1-score for a single annotator
        micro_stats = []
        # Vary neighbors - only use odd k values to avoid ties in knn()
        for k in xrange(1, k_max + 1, 2):
            # Per annotator
            for anno in filtered_inst_annos:
                micro = hierarchical_learning(anno, weights, recency, k, i,
                                              keep_na, similarity)
                micro_stats.append(micro)
            # c) Aggregate hierarchical F1-scores of all annotators
            prec, rec, f1 = compute_micro_metrics(micro_stats)
            data[k] = (prec, rec, f1)

        # d) Store results in a file
        if use_n_threshold:
            oi_fname = "{}_{}_tweets_{}_stage_results_sim_metric_{}_weight_" \
                       "{}_training_size_{}_min_annos_{}_use_n_{}_{}.txt" \
                .format(inst_name, difficulty, stage, similarity, weights, i,
                        min_annos, use_n, agg)
        else:
            oi_fname = "{}_{}_tweets_{}_stage_results_sim_metric_{}_weight" \
                       "_{}_training_size_{}_min_annos_{}_{}.txt"\
                .format(inst_name, difficulty, stage, similarity, weights, i,
                        min_annos, agg)
        dst = os.path.join(stat_dir, oi_fname)
        store_csv(data, dst)


def store_csv(data, dst):
    """
    Stores data in csv file.

    Parameters
    ----------
    data: dict -
    {
        k1: (precision, recall, f1_score),
        k2: (precision, recall, f1_score)

    }
    Store tuples of precision, recall, F1-score for varying number of neighbors
    and different similarity measures.
    dst: str - path under which the data will be stored.

    """
    with open(dst, "wb") as f:
        # Header
        f.write("neighbors,precision,recall,f1\n")
        # Sort k's ascendingly
        for entries in sorted(data.items(), key=operator.itemgetter(0)):
            # (k, (prec, rec, f1))
            neighbors = entries[0]
            prec, rec, f1 = entries[1]
            f.write("{},{},{},{}\n".format(neighbors, prec, rec, f1))


def read_data_from_csv(src, metric_name):
    """
    Read in a csv file that stored results from this experiment

    Parameters
    ----------
    src: str - location of input csv file.
    metric_name: str - name of the hierarchical metric: "prec" precision,
    "rec" for recall, or "f1" for F1-score

    Returns
    -------
    List of int, list of float.
    First list are x-axis values, number of neighbors considered for
    predictions. Second list contains the y-axis values, the metric that should
    be displayed, either precision, recall, or F1-score (all are hierarchical).

    """
    with open(src, "rb") as f:
        lines = f.readlines()
    # We put k, number of neighbors considered for prediction, on x-axis
    x = []
    y = []
    # Skip header
    for line in lines[1:]:
        # Get rid of \n or \n\r or leading/trailing whitespaces before splitting
        data = line.replace('\n', '').replace('\r', '').strip().split(",")
        k = int(data[0])
        metric_idx = 2
        if metric_name == "prec":
            metric_idx = 0
        if metric_name == "rec":
            metric_idx = 1
        x.append(k)
        metric_val = float(data[metric_idx])
        y.append(metric_val)
    return x, y


def hierarchical_learning(anno, weights, recency, k, i, keep_na,
                          similarity):
    """
    Cast the learning problem as a hierarchical classification task.
    For each hierarchy level, a separate classifier is trained. It's impossible
    to build a classifier for each parent class due to a potential lack of
    training examples (e.g. for 2nd level only "relevant" tweets exist, so no
    classifier could be built for 2nd level of "irrelevant"). Therefore, in
    total 6 classifiers are created per user: 2 per level and there are 3
    levels. 2 per level because we need one classifier for learning phase and
    one for rest. Each classifier is invoked and for evaluation micro-averaged
    hierarchical precision, recall and F1-score are leveraged.


    Parameters
    ----------
    anno: anno.Annotator - annotator for whom the classifiers should be built.
    weights: str - weight function used in prediction.  Possible values:
        - "uniform" : uniform weights.  All points in each neighborhood
          are weighted equally.
        - "distance" : weight points by the inverse of their distance.
          in this case, closer neighbors of a query point will have a
          greater influence than neighbors which are further away.
    recency: bool - True if scenario 2 should be tested. Otherwise scenario 1
    is tested.
    k: int - number of neighbors to consider when predicting label.
    i: int - first i tweets of an annotator are used as the training set.
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
    dict of dicts of dicts.
    Dictionary holds the micro-averaged performance metrics per group and
    institution. Keys are "md" and "su" and in each inner dict we have for
    learning phase "learning" and rest "rest" another dict as follows.
    Keys  are the groups, "S","M" and "All" and their values are hierarchical
    precision, recall, and F1-score in that order in a list. "All" represents
    an institution's performance metrics.

    """
    # List of texts
    train_anno = deepcopy(anno)
    # Keep first i tweets as training set
    train_anno.keep_k(i)
    # print "use {} tweets for training".format(len(train_anno))

    # Keep all, but the first i tweets, as test set
    test_anno = deepcopy(anno)
    test_anno.keep_rest(i)

    build_classifier(train_anno, test_anno, keep_na=keep_na, weights=weights,
                     k=k, recency=recency, similarity=similarity)
    y_pred = test_anno.tweets.predictions
    # print "#predictions", len(y_pred)
    # print "#labels", len(test_anno.tweets.labels)
    # Convert predictions into the same format as true labels before measuring
    # performance
    # Get true labels of test instances
    y_true = test_anno.tweets.labels

    _, _, _, cnt = hierarchical_precision_recall_f1(y_pred, y_true)
    return cnt


def build_classifier(train, test, recency=False, keep_na=True,
                     similarity="edit", weights="uniform", k=3):
    """
    Builds a classifier for the given annotator trained on first k tweets.
    Predicts the labels of the remaining tweets in <test>.
    Predictions are stored in <test>.
    Format of predictions: list of list of list of tuple.
    Tuple contains (label, probability). Innermost list represents a the tuples
    predicted for a tweet on a specific hierarchy level. Middle represents
    all hierarchy levels of the tweet, i.e. it contains 3 inner lists. Outer
    represents predictions for each tweet, i.e. len(outer_list) ) = number of
    tweets for which labels were predicted.

    Parameters
    ----------
    train: anno.Annotator - contains the training data of a certain annotator.
    test: anno.Annotator - contains the test data of the same annotator.
    weights: str - weight function used in prediction.  Possible values:
        - "uniform" : uniform weights.  All points in each neighborhood
          are weighted equally.
        - "distance" : weight points by the inverse of their distance.
          in this case, closer neighbors of a query point will have a
          greater influence than neighbors which are further away.
    recency: bool - True if scenario 2 should be tested. Otherwise scenario 1
    is tested.
    k: int - number of neighbors to consider when predicting label.
    keep_na: bool - True if neighbors with no label should be considered when
    predicting the label of an unknown instance. If False, such instances with
    unavailable labels are discarded.
    similarity: str - similarity function to be used:
        - "substring": longest common SUBSTRING (= contiguously shared words)
        - "subsequence": longest common SUBSEQUENCE (= shared words in relative
        order, not necessarily contiguous)
        - "edit": edit distance

    """
    # a) Get training data
    X_train = train.get_texts()
    y_train = train.get_labels_per_level()
    # List of labels of the texts for 1st label set
    y_train_first_set = y_train[0]
    # List of labels of the texts for 2nd label set
    y_train_second_set = y_train[1]
    # List of labels of the texts for 3rd label set
    y_train_third_set = y_train[2]

    # b) Get test data
    X_test = test.get_texts()

    # c) Get predictions for test data
    # Predictions for first level
    y_pred1 = knn(X_train, y_train_first_set, X_test, recency=recency,
                  keep_na=keep_na, similarity=similarity, weights=weights,
                  k=k)
    # Predictions for second level
    y_pred2 = knn(X_train, y_train_second_set, X_test, recency=recency,
                  keep_na=keep_na, similarity=similarity, weights=weights,
                  k=k)
    # Predictions for third level
    y_pred3 = knn(X_train, y_train_third_set, X_test, recency=recency,
                  keep_na=keep_na, similarity=similarity, weights=weights,
                  k=k)

    # Concatenate the i-th entry of each hierarchy level together as a list
    # which represents the labels of a single tweet
    preds = map(list, zip(y_pred1, y_pred2, y_pred3))

    # Store predictions
    test.set_predictions(preds)


def plot_early_vs_late(src_early, src_late, dst, metric_name, similarity):
    """
    Plots the same performance metric (hierarchical precision/recall/F1-score)
    for early and late stage in a single plot.

    Parameters
    ----------
    src_early: str - file to input csv file for early stage.
    src_early: str - file to input csv file for late stage.
    dst: str - location where plot will be stored.
    metric_name: str - name of the hierarchical metric: "prec" precision,
    "rec" for recall, or "f1" for F1-score
    similarity: str - name of the metric to be displayed: "subsequence" for
    longest common subsequence, "substring" for longest common substring,
    "edit" for edit distance.

    """
    x_early, y_early = read_data_from_csv(src_early, metric_name)
    x_late, y_late = read_data_from_csv(src_late, metric_name)
    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_subplot(111)
    ax.plot(x_early, y_early, color="blue", label="$EARLY$")
    ax.plot(x_late, y_late, color="red", label="$LATE$")
    # Add a dashed line
    x = (1, x_early[-1])
    y_val = max(y_late[-1], y_early[-1]) + 0.01
    y = (y_val, y_val)
    ax.plot(x, y, "k--")
    # Hide the right and top spines (lines)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    # Set labels of axes
    ax.set_xlabel("k")
    ax.set_ylabel("F1-score")
    # Add legend outside of plot
    ax.legend(loc="lower right", shadow=True, fontsize=FONTSIZE)
    plt.xlim(1, x_late[-1]+0.5)
    plt.ylim(0, 1)
    plt.savefig(dst, bbox_inches='tight', dpi=600)
    plt.close()


def plot_easy_vs_difficult(src_early, src_late, dst, metric_name, similarity):
    """
    Plots the same performance metric (hierarchical precision/recall/F1-score)
    in the same stage for easy and difficult tweets.

    Parameters
    ----------
    src_early: str - file to input csv file for early stage.
    src_early: str - file to input csv file for late stage.
    dst: str - location where plot will be stored.
    metric_name: str - name of the hierarchical metric: "prec" precision,
    "rec" for recall, or "f1" for F1-score
    similarity: str - name of the metric to be displayed: "subsequence" for
    longest common subsequence, "substring" for longest common substring,
    "edit" for edit distance.

    """
    x_early, y_early = read_data_from_csv(src_early, metric_name)
    x_late, y_late = read_data_from_csv(src_late, metric_name)
    # print y_early
    # print y_late
    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_subplot(111)
    # Renamed from PredictorE to PredND for thesis
    # ax.plot(x_early, y_early, color="blue", label="PredictorE")
    ax.plot(x_early, y_early, color="dodgerblue", label="PredND")
    # Renamed from PredictorD to PredD for thesis
    # ax.plot(x_late, y_late, color="red", label="PredictorD")
    ax.plot(x_late, y_late, color="orangered", label="PredD")
    # Add a dashed line
    x = (1, x_early[-1])
    y_val = max(y_late[-1], y_early[-1]) + 0.01
    y = (y_val, y_val)
    ax.plot(x, y, "k--")
    # Hide the right and top spines (lines)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    # Set labels of axes
    ax.set_xlabel("k")
    ax.set_ylabel("F1-score")
    # Add legend outside of plot
    ax.legend(loc="lower right", shadow=True, fontsize=FONTSIZE)
    plt.xlim(1, x_late[-1]+0.5)
    plt.ylim(0, 1)
    plt.savefig(dst, bbox_inches='tight', dpi=600)
    plt.close()


def count_annotators_per_tweet(annos):
    """
    Counts how many annotators labeled each tweet.

    Parameters
    ----------
    annos: list of anno.Annotator - each annotator holds the tweets she labeled.

    Returns
    -------
    dict, dict, dict.
    Dictionary holding for each tweet ID how many annotators labeled it - over
    early and late stage. Same again, but this time only over early stage. Same
    again, but this time only over late stage.

    """
    # {tweet_id: count}
    counts = {}
    for anno in annos:
        for tid in anno.all_tweets():
            if tid not in counts:
                counts[tid] = 0
            counts[tid] += 1
    return counts


def read_dataset(
        tweet_path, anno_path, inst_name, thresholds, cleaned=False,
        min_annos=3, is_early=True, use_n=50, use_n_threshold=True):
    """
    Read dataset.

    Parameters
    ----------
    tweet_path: str - path to tweet csv dataset.
    anno_path: str - path to annotator csv dataset.
    inst_name: string - name of the institution.
    institution name or group and the thresholds are the corresponding values.
    thresholds: dict - thresholds for early/late annotation stage. Keys are
    institution name or group and the thresholds are the corresponding values.
    cleaned: bool - True if the cleaned data is used as input.
    min_annos: int - minimum number of annotators who must've assigned a label
    to a tweet. Otherwise it'll be discarded.
    is_early: bool - True if only tweets from the early phase should be
    considered. Else only tweets from the late stage are considered.
    use_n: int - considers only the first n tweets per annotator, regardless of
    annotator group.
    use_n_threshold: bool - True if early/late stage threshold over S, M, L for
    first <use_n> tweets should be used. Else the thresholds over all tweets of
    S and M is are used.

    Returns
    --------
    list, dict, dict, dict.
    List of Annotator objects in institution with their tweets separated into
    early and late stage.
    Dictionary (group names "S" and "M" are keys) with lists of Annotator
    objects per group as value with their tweets separated into early and late
    stage.
    Counters for institution, i.e. how often each tweet was labeled.
    Counters for groups, i.e. how often each tweet was labeled in the group.
    Counters only contain counts for tweets that were labeled sufficiently
    often. Counters return tuples (raw_count, normalized_count), where
    raw counts are divided by the max(raw_count) to obtain normalized counts.

    """
    if use_n_threshold:
        print "use n thresholds!"
        return _get_first_n_tweets_per_anno_with_l(
            tweet_path, anno_path, inst_name, thresholds, cleaned, min_annos,
            is_early, use_n)
    else:
        return _get_all_tweets_per_anno_without_l(
            tweet_path, anno_path, inst_name, thresholds, cleaned, min_annos,
            is_early)


def _get_all_tweets_per_anno_without_l(
        tweet_path, anno_path, inst_name, thresholds, cleaned=False,
        min_annos=3, is_early=True):
    """
    Read dataset (tweet texts per annotator and their labels) per group
    and institution.
    Group L is ignored. All tweets are used of all other annotators.

    Parameters
    ----------
    tweet_path: str - path to tweet csv dataset.
    anno_path: str - path to annotator csv dataset.
    inst_name: string - name of the institution.
    thresholds: dict - thresholds for early/late annotation stage. Keys are
    institution name or group and the thresholds are the corresponding values.
    cleaned: bool - True if the cleaned data is used as input.
    min_annos: int - minimum number of annotators who must've assigned a label
    to a tweet. Otherwise it'll be discarded.
    is_early: bool - True if only tweets from the early phase should be
    considered. Else only tweets from the late stage are considered.

    Returns
    --------
    list, dict, dict, dict.
    List of Annotator objects in institution with their tweets separated into
    early and late stage.
    Dictionary (group names "S" and "M" are keys) with lists of Annotator
    objects per group as value with their tweets separated into early and late
    stage.
    Counters for institution, i.e. how often each tweet was labeled.
    Counters for groups, i.e. how often each tweet was labeled in the group.
    Counters only contain counts for tweets that were labeled sufficiently
    often. Counters return tuples (raw_count, normalized_count), where
    raw counts are divided by the max(raw_count) to obtain normalized counts.

    """
    # Store a list of annotators per group/institution
    inst_annos = []
    group_annos = {
        "S": [],
        "M": []
    }
    data = pseudo_db.Data(tweet_path, anno_path)
    # For each annotator
    for anno in data.annotators.all_annos():
        group = anno.get_group()
        # Ignore annotations from group L
        if group != "L":
            name = anno.get_name()
            group_anno = Annotator(name, group)
            inst_anno = Annotator(name, group)
            # For each tweet
            for idx, t in enumerate(anno.get_labeled_tweets()):
                times = t.get_anno_times()
                labels = t.get_anno_labels()
                tweet_id = t.get_tid()
                text = t.get_text()
                # First level
                rel_label = labels[0]
                rel_time = times[0]
                fac_time = ZERO
                opi_time = ZERO
                l2 = EMPTY
                l3 = EMPTY
                # Discard remaining labels if annotator chose "Irrelevant"
                # Consider other sets of labels iff either the cleaned
                # dataset should be created and the label is "relevant" OR
                # the raw dataset should be used.
                if (cleaned and rel_label != "Irrelevant") or not cleaned:
                    l2 = labels[1]
                    # Second level
                    fac_time = times[1]
                    # Annotator labeled the 3rd set of labels as well
                    if l2 == pseudo_db.GROUND_TRUTH["Non-factual"]:
                        # Third level
                        l3 = labels[2]
                        opi_time = times[2]
                # Annotation time for i-th tweet
                anno_time = sum([rel_time, fac_time, opi_time])
                labels = [rel_label, l2, l3]

                # Use only tweets from early stage
                if is_early:
                    if anno_time <= thresholds[group]:
                        group_anno.add_tweet(tweet_id, anno_time, labels,
                                             text)
                    if anno_time <= thresholds[inst_name]:
                        inst_anno.add_tweet(tweet_id, anno_time, labels,
                                            text)
                # Use only tweets from late stage
                else:
                    if anno_time > thresholds[group]:
                        group_anno.add_tweet(tweet_id, anno_time, labels,
                                             text)
                    if anno_time > thresholds[inst_name]:
                        inst_anno.add_tweet(tweet_id, anno_time, labels,
                                            text)
            # Store annotator in group/institution
            inst_annos.append(inst_anno)
            group_annos[group].append(group_anno)

    # Count for each tweet how often it was labeled - per group and institution
    # as the results could vary, e.g. some tweets are insufficiently labeled
    # in groups but sufficiently often in the whole institution. The reason for
    # NOT counting in the previous loop is that 3 annotators of MD (M) - see
    # anno.py for a detailed explanation at the top - labeled the same tweet
    # twice, so counting would be off by 1 for 3 annotators. Therefore,
    # anno.Annotator handles these exceptions and ignores the tweets that were
    # labeled a second time.

    inst_counts = count_annotators_per_tweet(inst_annos)
    group_counts = {
        "S": count_annotators_per_tweet(group_annos["S"]),
        "M": count_annotators_per_tweet(group_annos["M"])
    }

    # Now only keep tweets that were labeled sufficiently often by annotators
    # Create a list of tweet IDs that must be removed since they weren't labeled
    # by enough annotators
    removed_inst_tweets = [tid for tid in inst_counts if
                           inst_counts[tid] < min_annos]
    print "remove from INSTITUTION all:", len(removed_inst_tweets)

    # Delete from each group/institution all tweets that weren't labeled
    # by enough annotators
    for anno in inst_annos:
        anno.delete_tweets(removed_inst_tweets)

    # Test that all tweets were removed
    for anno in inst_annos:
        for tid in anno.all_tweets():
            if tid in removed_inst_tweets:
                raise Exception("can't happen")

    # Make sure that we only count tweets that were sufficiently often
    # labeled in the institution
    for tid in removed_inst_tweets:
        del inst_counts[tid]

    # Delete tweets from groups that weren't labeled sufficiently often
    for group in group_annos:
        # Create a list of tweet IDs that must be removed since they
        # weren't labeled by enough annotators
        removed_group_tweets = [tid for tid in group_counts[group] if
                                group_counts[group][tid] < min_annos]
        print "remove from {}: {}".format(group, len(removed_group_tweets))

        for anno in group_annos[group]:
            anno.delete_tweets(removed_group_tweets)

        # Make sure that we only count tweets that were sufficiently often
        # labeled in the group
        for tid in removed_group_tweets:
            del group_counts[group][tid]

        # Test that all tweets were removed
        for anno in group_annos[group]:
            for tid in anno.all_tweets():
                if tid in removed_group_tweets:
                    raise Exception("can't happen")


def _get_first_n_tweets_per_anno_with_l(
        tweet_path, anno_path, inst_name, thresholds, cleaned=False,
        min_annos=3, is_early=True, use_n=50):
    """
    Read dataset (tweet texts per annotator and their labels) per group
    and institution.
    Group L is used. Gets only the first <use_n> tweets of each annotator.

    Parameters
    ----------
    tweet_path: str - path to tweet csv dataset.
    anno_path: str - path to annotator csv dataset.
    inst_name: string - name of the institution.
    thresholds: dict - thresholds for early/late annotation stage. Keys are
    institution name or group and the thresholds are the corresponding values.
    cleaned: bool - True if the cleaned data is used as input.
    min_annos: int - minimum number of annotators who must've assigned a label
    to a tweet. Otherwise it'll be discarded.
    is_early: bool - True if only tweets from the early phase should be
    considered. Else only tweets from the late stage are considered.
    use_n: int - considers only the first n tweets per annotator, regardless of
    annotator group.

    Returns
    --------
    list, dict, dict, dict.
    List of Annotator objects in institution with their tweets separated into
    early and late stage.
    Dictionary (group names "S", "M", "L" are keys) with lists of Annotator
    objects per group as value with their tweets separated into early and late
    stage.
    Counters for institution, i.e. how often each tweet was labeled.
    Counters for groups, i.e. how often each tweet was labeled in the group.
    Counters only contain counts for tweets that were labeled sufficiently
    often. Counters return tuples (raw_count, normalized_count), where
    raw counts are divided by the max(raw_count) to obtain normalized counts.

    """
    # Store a list of annotators per group/institution
    inst_annos = []
    group_annos = {
        "S": [],
        "M": [],
        "L": []
    }

    data = pseudo_db.Data(tweet_path, anno_path)
    # For each annotator
    for anno in data.annotators.all_annos():
        group = anno.get_group()
        # Ignore annotations from group L
        if group != "L":
            name = anno.get_name()
            group_anno = Annotator(name, group)
            inst_anno = Annotator(name, group)
            # For each tweet
            for idx, t in enumerate(anno.get_labeled_tweets()):
                if idx < use_n:
                    times = t.get_anno_times()
                    labels = t.get_anno_labels()
                    tweet_id = t.get_tid()
                    text = t.get_text()
                    # First level
                    rel_label = labels[0]
                    rel_time = times[0]
                    fac_time = ZERO
                    opi_time = ZERO
                    l2 = EMPTY
                    l3 = EMPTY
                    # Discard remaining labels if annotator chose "Irrelevant"
                    # Consider other sets of labels iff either the cleaned
                    # dataset should be created and the label is "relevant" OR
                    # the raw dataset should be used.
                    if (cleaned and rel_label != "Irrelevant") or not cleaned:
                        l2 = labels[1]
                        # Second level
                        fac_time = times[1]
                        # Annotator labeled the 3rd set of labels as well
                        if l2 == pseudo_db.GROUND_TRUTH["Non-factual"]:
                            # Third level
                            l3 = labels[2]
                            opi_time = times[2]
                    # Annotation time for i-th tweet
                    anno_time = sum([rel_time, fac_time, opi_time])
                    labels = [rel_label, l2, l3]

                    if is_early:
                        if idx < thresholds[group]:
                            group_anno.add_tweet(tweet_id, anno_time,
                                                 labels, text)
                        if idx < thresholds[inst_name]:
                            inst_anno.add_tweet(tweet_id, anno_time, labels,
                                                text)
                    # Use only tweets from late stage
                    else:
                        if idx >= thresholds[group]:
                            group_anno.add_tweet(tweet_id, anno_time,
                                                 labels, text)
                        if idx >= thresholds[inst_name]:
                            inst_anno.add_tweet(tweet_id, anno_time, labels,
                                                text)
            # Store annotator in group/institution
            inst_annos.append(inst_anno)
            group_annos[group].append(group_anno)

    # Count for each tweet how often it was labeled - per group and institution
    # as the results could vary, e.g. some tweets are insufficiently labeled
    # in groups but sufficiently often in the whole institution. The reason for
    # NOT counting in the previous loop is that 3 annotators of MD (M) - see
    # anno.py for a detailed explanation at the top - labeled the same tweet
    # twice, so counting would be off by 1 for 3 annotators. Therefore,
    # anno.Annotator handles these exceptions and ignores the tweets that were
    # labeled a second time.

    inst_counts = count_annotators_per_tweet(inst_annos)
    group_counts = {
        "S": count_annotators_per_tweet(group_annos["S"]),
        "M": count_annotators_per_tweet(group_annos["M"]),
        "L": count_annotators_per_tweet(group_annos["L"])
    }

    # Now only keep tweets that were labeled sufficiently often by annotators
    # Create a list of tweet IDs that must be removed since they weren't labeled
    # by enough annotators
    removed_inst_tweets = [tid for tid in inst_counts if
                           inst_counts[tid] < min_annos]
    print "remove from INSTITUTION all:", len(removed_inst_tweets)

    # Delete from each group/institution all tweets that weren't labeled
    # by enough annotators
    for anno in inst_annos:
        anno.delete_tweets(removed_inst_tweets)

    # Test that all tweets were removed
    for anno in inst_annos:
        for tid in anno.all_tweets():
            if tid in removed_inst_tweets:
                raise Exception("can't happen")

    # Make sure that we only count tweets that were sufficiently often
    # labeled in the institution
    for tid in removed_inst_tweets:
        del inst_counts[tid]

    # Delete tweets from groups that weren't labeled sufficiently often
    for group in group_annos:
        # Create a list of tweet IDs that must be removed since they
        # weren't labeled by enough annotators
        removed_group_tweets = [tid for tid in group_counts[group] if
                                group_counts[group][tid] < min_annos]
        print "remove from {}: {}".format(group, len(removed_group_tweets))

        for anno in group_annos[group]:
            anno.delete_tweets(removed_group_tweets)

        # Make sure that we only count tweets that were sufficiently often
        # labeled in the group
        for tid in removed_group_tweets:
            del group_counts[group][tid]

        # Test that all tweets were removed
        for anno in group_annos[group]:
            for tid in anno.all_tweets():
                if tid in removed_group_tweets:
                    raise Exception("can't happen")

    # Normalize counts, i.e. divide each count by the maximum count to get
    # values between 0-1
    norm_inst_counts = normalize_counts(inst_counts)
    norm_group_counts = {
        "S": normalize_counts(group_counts["S"]),
        "M": normalize_counts(group_counts["M"]),
        "L": normalize_counts(group_counts["L"])
    }
    return inst_annos, group_annos, norm_inst_counts, norm_group_counts


def read_dataset(
        dbs, db_idxs, inst_name, thresholds, anno_coll_name="user",
        tweet_coll_name="tweets", cleaned=False, min_annos=3, is_early=True,
        use_n=50, use_n_threshold=True):
    """
    Read dataset.

    Parameters
    ----------
    dbs: list of strings - names of the existing DBs.
    db_idxs: list of ints - name of the MongoDB from where data should be read.
    inst_name: string - name of the institution.
    thresholds: dict - thresholds for early/late annotation stage. Keys are
    institution name or group and the thresholds are the corresponding values.
    anno_coll_name: str - name of the collection holding the annotator data.
    tweet_coll_name: str - name of the collection holding the tweet data.
    cleaned: bool - True if the cleaned data is used as input.
    min_annos: int - minimum number of annotators who must've assigned a label
    to a tweet. Otherwise it'll be discarded.
    is_early: bool - True if only tweets from the early phase should be
    considered. Else only tweets from the late stage are considered.
    use_n: int - considers only the first n tweets per annotator, regardless of
    annotator group.
    use_n_threshold: bool - True if early/late stage threshold over S, M, L for
    first <use_n> tweets should be used. Else the thresholds over all tweets of
    S and M is are used.

    Returns
    --------
    list, dict, dict, dict.
    List of Annotator objects in institution with their tweets separated into
    early and late stage.
    Dictionary (group names "S" and "M" are keys) with lists of Annotator
    objects per group as value with their tweets separated into early and late
    stage.

    """
    if use_n_threshold:
        print "use n thresholds!"
        return _get_first_n_tweets_per_anno_with_l(
            dbs, db_idxs, inst_name, thresholds, anno_coll_name,
            tweet_coll_name, cleaned, min_annos, is_early, use_n)
    else:
        return _get_all_tweets_per_anno_without_l(
            dbs, db_idxs, inst_name, thresholds, anno_coll_name,
            tweet_coll_name, cleaned, min_annos, is_early)


def _get_all_tweets_per_anno_without_l(
        dbs, db_idxs, inst_name, thresholds, anno_coll_name="user",
        tweet_coll_name="tweets", cleaned=False, min_annos=3, is_early=True):
    """
    Read dataset (tweet texts per annotator and their labels) per group
    and institution.
    Group L is ignored. All tweets are used of all other annotators.

    Parameters
    ----------
    dbs: list of strings - names of the existing DBs.
    db_idxs: list of ints - name of the MongoDB from where data should be read.
    inst_name: string - name of the institution.
    thresholds: dict - thresholds for early/late annotation stage. Keys are
    institution name or group and the thresholds are the corresponding values.
    anno_coll_name: str - name of the collection holding the annotator data.
    tweet_coll_name: str - name of the collection holding the tweet data.
    cleaned: bool - True if the cleaned data is used as input.
    min_annos: int - minimum number of annotators who must've assigned a label
    to a tweet. Otherwise it'll be discarded.
    is_early: bool - True if only tweets from the early phase should be
    considered. Else only tweets from the late stage are considered.

    Returns
    --------
    list, dict, dict, dict.
    List of Annotator objects in institution with their tweets separated into
    early and late stage.
    Dictionary (group names "S" and "M" are keys) with lists of Annotator
    objects per group as value with their tweets separated into early and late
    stage.
    Counters for institution, i.e. how often each tweet was labeled.
    Counters for groups, i.e. how often each tweet was labeled in the group.
    Counters only contain counts for tweets that were labeled sufficiently
    often. Counters return tuples (raw_count, normalized_count), where
    raw counts are divided by the max(raw_count) to obtain normalized counts.

    """
    # Store a list of annotators per group/institution
    inst_annos = []
    group_annos = {
        "S": [],
        "M": []
    }

    for db_idx in db_idxs:
        # Get DB name
        db = dbs[db_idx]
        tweet_coll, anno_coll = utility.load_tweets_annotators_from_db(
            db, tweet_coll_name, anno_coll_name)
        # For each anno
        for anno in anno_coll.find():
            username = anno["username"]
            group = anno["group"]
            # Use username + "_" + group because 2 annotators of MD
            # labeled for S and M (otherwise their entries are overridden)
            dict_key = username + "_" + group
            # Ignore annotations from group L
            if group != "L":
                group_anno = Annotator(dict_key, group)
                inst_anno = Annotator(dict_key, group)
                # Tweet IDs labeled by this annotator
                labeled = anno["annotated_tweets"]
                for tid in labeled:
                    second_label = EMPTY
                    third_label = EMPTY
                    fac_time = ZERO
                    opi_time = ZERO
                    tweet = utility.get_tweet(tweet_coll, tid)
                    # Use Twitter ID because _id differs for the same tweet as
                    # it was created in multiple DBs.
                    tweet_id = tweet["id_str"]
                    text = tweet["text"]
                    first_label = tweet["relevance_label"][username]
                    rel_time = tweet["relevance_time"][username]
                    # Annotator labeled the 3rd set of labels as well
                    # Discard remaining labels if annotator chose "Irrelevant"
                    # Consider other sets of labels iff either the cleaned
                    # dataset should be created and the label is "relevant" OR
                    # the raw dataset should be used.
                    if (cleaned and first_label != "Irrelevant") or not cleaned:
                        second_label = tweet["fact_label"][username]
                        fac_time = tweet["fact_time"][username]
                        # Annotator labeled the 3rd set of labels as well
                        if username in tweet["opinion_label"]:
                            third_label = tweet["opinion_label"][username]
                            opi_time = tweet["opinion_time"][username]
                    # Add annotation times and labels to annotator
                    anno_time = sum([rel_time, fac_time, opi_time])
                    labels = [first_label, second_label, third_label]
                    # Use only tweets from early stage
                    if is_early:
                        if anno_time <= thresholds[group]:
                            group_anno.add_tweet(tweet_id, anno_time, labels,
                                                 text)
                        if anno_time <= thresholds[inst_name]:
                            inst_anno.add_tweet(tweet_id, anno_time, labels,
                                                text)
                    # Use only tweets from late stage
                    else:
                        if anno_time > thresholds[group]:
                            group_anno.add_tweet(tweet_id, anno_time, labels,
                                                 text)
                        if anno_time > thresholds[inst_name]:
                            inst_anno.add_tweet(tweet_id, anno_time, labels,
                                                text)
                # Store annotator in group/institution
                inst_annos.append(inst_anno)
                group_annos[group].append(group_anno)

    # Count for each tweet how often it was labeled - per group and institution
    # as the results could vary, e.g. some tweets are insufficiently labeled
    # in groups but sufficiently often in the whole institution. The reason for
    # NOT counting in the previous loop is that 3 annotators of MD (M) - see
    # anno.py for a detailed explanation at the top - labeled the same tweet
    # twice, so counting would be off by 1 for 3 annotators. Therefore,
    # anno.Annotator handles these exceptions and ignores the tweets that were
    # labeled a second time.

    inst_counts = count_annotators_per_tweet(inst_annos)
    group_counts = {
        "S": count_annotators_per_tweet(group_annos["S"]),
        "M": count_annotators_per_tweet(group_annos["M"])
    }

    # Now only keep tweets that were labeled sufficiently often by annotators
    # Create a list of tweet IDs that must be removed since they weren't labeled
    # by enough annotators
    removed_inst_tweets = [tid for tid in inst_counts if
                           inst_counts[tid] < min_annos]
    print "remove from INSTITUTION all:", len(removed_inst_tweets)

    # Delete from each group/institution all tweets that weren't labeled
    # by enough annotators
    for anno in inst_annos:
        anno.delete_tweets(removed_inst_tweets)

    # Test that all tweets were removed
    for anno in inst_annos:
        for tid in anno.all_tweets():
            if tid in removed_inst_tweets:
                raise Exception("can't happen")

    # Make sure that we only count tweets that were sufficiently often
    # labeled in the institution
    for tid in removed_inst_tweets:
        del inst_counts[tid]

    # Delete tweets from groups that weren't labeled sufficiently often
    for group in group_annos:
        # Create a list of tweet IDs that must be removed since they
        # weren't labeled by enough annotators
        removed_group_tweets = [tid for tid in group_counts[group] if
                                group_counts[group][tid] < min_annos]
        print "remove from {}: {}".format(group, len(removed_group_tweets))

        for anno in group_annos[group]:
            anno.delete_tweets(removed_group_tweets)

        # Make sure that we only count tweets that were sufficiently often
        # labeled in the group
        for tid in removed_group_tweets:
            del group_counts[group][tid]

        # Test that all tweets were removed
        for anno in group_annos[group]:
            for tid in anno.all_tweets():
                if tid in removed_group_tweets:
                    raise Exception("can't happen")

    return inst_annos, group_annos


def _get_first_n_tweets_per_anno_with_l(
        dbs, db_idxs, inst_name, thresholds, anno_coll_name="user",
        tweet_coll_name="tweets", cleaned=False, min_annos=3, is_early=True,
        use_n=50):
    """
    Read dataset (tweet texts per annotator and their labels) per group
    and institution.
    Group L is used. Gets only the first <use_n> tweets of each annotator.

    Parameters
    ----------
    dbs: list of strings - names of the existing DBs.
    db_idxs: list of ints - name of the MongoDB from where data should be read.
    inst_name: string - name of the institution.
    thresholds: dict - thresholds for early/late annotation stage. Keys are
    institution name or group and the thresholds are the corresponding values.
    anno_coll_name: str - name of the collection holding the annotator data.
    tweet_coll_name: str - name of the collection holding the tweet data.
    cleaned: bool - True if the cleaned data is used as input.
    min_annos: int - minimum number of annotators who must've assigned a label
    to a tweet. Otherwise it'll be discarded.
    is_early: bool - True if only tweets from the early phase should be
    considered. Else only tweets from the late stage are considered.
    use_n: int - considers only the first n tweets per annotator, regardless of
    annotator group.

    Returns
    --------
    list, dict, dict, dict.
    List of Annotator objects in institution with their tweets separated into
    early and late stage.
    Dictionary (group names "S", "M", "L" are keys) with lists of Annotator
    objects per group as value with their tweets separated into early and late
    stage.
    Counters for institution, i.e. how often each tweet was labeled.
    Counters for groups, i.e. how often each tweet was labeled in the group.
    Counters only contain counts for tweets that were labeled sufficiently
    often. Counters return tuples (raw_count, normalized_count), where
    raw counts are divided by the max(raw_count) to obtain normalized counts.

    """
    # Store a list of annotators per group/institution
    inst_annos = []
    group_annos = {
        "S": [],
        "M": [],
        "L": []
    }

    for db_idx in db_idxs:
        # Get DB name
        db = dbs[db_idx]
        tweet_coll, anno_coll = utility.load_tweets_annotators_from_db(
            db, tweet_coll_name, anno_coll_name)
        # For each anno
        for anno in anno_coll.find():
            username = anno["username"]
            group = anno["group"]
            # Use username + "_" + group because 2 annotators of MD
            # labeled for S and M (otherwise their entries are overridden)
            dict_key = username + "_" + group
            # Ignore annotations from group L
            if group != "L":
                group_anno = Annotator(dict_key, group)
                inst_anno = Annotator(dict_key, group)
                # Tweet IDs labeled by this annotator
                labeled = anno["annotated_tweets"]
                for idx, tid in enumerate(labeled):
                    if idx < use_n:
                        second_label = EMPTY
                        third_label = EMPTY
                        fac_time = ZERO
                        opi_time = ZERO
                        tweet = utility.get_tweet(tweet_coll, tid)
                        # Use Twitter ID because _id differs for the same
                        # tweet as it was created in multiple DBs.
                        tweet_id = tweet["id_str"]
                        text = tweet["text"]
                        first_label = tweet["relevance_label"][username]
                        rel_time = tweet["relevance_time"][username]
                        # Annotator labeled the 3rd set of labels as well
                        # Discard remaining labels if annotator chose
                        # "Irrelevant"
                        # Consider other sets of labels iff either the cleaned
                        # dataset should be created and the label is "relevant"
                        # OR the raw dataset should be used.
                        if (cleaned and first_label != "Irrelevant") or not \
                                cleaned:
                            second_label = tweet["fact_label"][username]
                            fac_time = tweet["fact_time"][username]
                            # Annotator labeled the 3rd set of labels as well
                            if username in tweet["opinion_label"]:
                                third_label = tweet["opinion_label"][username]
                                opi_time = tweet["opinion_time"][username]
                        # Add annotation times and labels to annotator
                        anno_time = sum([rel_time, fac_time, opi_time])
                        labels = [first_label, second_label, third_label]
                        # Use only tweets from early stage
                        if is_early:
                            if anno_time <= thresholds[group]:
                                group_anno.add_tweet(tweet_id, anno_time,
                                                     labels, text)
                            if anno_time <= thresholds[inst_name]:
                                inst_anno.add_tweet(tweet_id, anno_time, labels,
                                                    text)
                        # Use only tweets from late stage
                        else:
                            if anno_time > thresholds[group]:
                                group_anno.add_tweet(tweet_id, anno_time,
                                                     labels, text)
                            if anno_time > thresholds[inst_name]:
                                inst_anno.add_tweet(tweet_id, anno_time, labels,
                                                    text)
                # Store annotator in group/institution
                inst_annos.append(inst_anno)
                group_annos[group].append(group_anno)

    # Count for each tweet how often it was labeled - per group and institution
    # as the results could vary, e.g. some tweets are insufficiently labeled
    # in groups but sufficiently often in the whole institution. The reason for
    # NOT counting in the previous loop is that 3 annotators of MD (M) - see
    # anno.py for a detailed explanation at the top - labeled the same tweet
    # twice, so counting would be off by 1 for 3 annotators. Therefore,
    # anno.Annotator handles these exceptions and ignores the tweets that were
    # labeled a second time.

    inst_counts = count_annotators_per_tweet(inst_annos)
    group_counts = {
        "S": count_annotators_per_tweet(group_annos["S"]),
        "M": count_annotators_per_tweet(group_annos["M"]),
        "L": count_annotators_per_tweet(group_annos["L"])
    }

    # Now only keep tweets that were labeled sufficiently often by annotators
    # Create a list of tweet IDs that must be removed since they weren't labeled
    # by enough annotators
    removed_inst_tweets = [tid for tid in inst_counts if
                           inst_counts[tid] < min_annos]
    print "remove from INSTITUTION all:", len(removed_inst_tweets)

    # Delete from each group/institution all tweets that weren't labeled
    # by enough annotators
    for anno in inst_annos:
        anno.delete_tweets(removed_inst_tweets)

    # Test that all tweets were removed
    for anno in inst_annos:
        for tid in anno.all_tweets():
            if tid in removed_inst_tweets:
                raise Exception("can't happen")

    # Make sure that we only count tweets that were sufficiently often
    # labeled in the institution
    for tid in removed_inst_tweets:
        del inst_counts[tid]

    # Delete tweets from groups that weren't labeled sufficiently often
    for group in group_annos:
        # Create a list of tweet IDs that must be removed since they
        # weren't labeled by enough annotators
        removed_group_tweets = [tid for tid in group_counts[group] if
                                group_counts[group][tid] < min_annos]
        print "remove from {}: {}".format(group, len(removed_group_tweets))

        for anno in group_annos[group]:
            anno.delete_tweets(removed_group_tweets)

        # Make sure that we only count tweets that were sufficiently often
        # labeled in the group
        for tid in removed_group_tweets:
            del group_counts[group][tid]

        # Test that all tweets were removed
        for anno in group_annos[group]:
            for tid in anno.all_tweets():
                if tid in removed_group_tweets:
                    raise Exception("can't happen")

    return inst_annos, group_annos


def read_thresholds(dst, inst_name):
    """
    Reads the thresholds for early/late stage of annotation process

    Parameters
    ----------
    dst: str - location where file is stored from which to read the thresholds.
    inst_name: str - name of the institution for which the thresholds hold.

    Returns
    -------
    dict.
    {<instname>: threshold, "S": threshold, "M": threshold}

    Raises
    ------

    """
    thresholds = {}
    with open(dst, "rb") as f:
        lines = f.readlines()
    # Skip header
    for line in lines[1:]:
        try:
            inst, s, m, l = line.split(",")
            l = float(l)
            thresholds["L"] = l
        except ValueError:
            # Only S, M, and institution are available
            inst, s, m = line.split(",")
        inst = float(inst)
        s = float(s)
        m = float(m)
        thresholds["S"] = s
        thresholds["M"] = m
        thresholds[inst_name] = inst
    return thresholds


def analyze_data(tweet_path, anno_path, inst_name, analysis_dir,
                 label_dir, cleaned=False, max_annos=1, train_ratio=0.66,
                 use_n=50, use_n_threshold=True, max_i=15):
    """
    Since the early and late stages contain a different number of tweets per
    annotator, this function shows the maximum number of tweets, i, that could
    be used in the actual simulation.
    It further analyzes how many annotators labeled at least i easy/late tweets.

    Parameters
    ----------
    tweet_path: str - path to tweet csv dataset.
    anno_path: str - path to annotator csv dataset.
    inst_name: str - name of the institution for which i is computed.
    analysis_dir: str - directory in which the results will be stored.
    label_dir: str - directory containing tweet difficulty labels.
    cleaned: bool - True if the data should be cleaned, i.e. if tweet is
    "irrelevant", its remaining labels are ignored for computing average
    annotation times.
    max_annos: int - maximum number of annotators who must've assigned a label
    to a tweet. All smaller/equal values will be used for <min_annos>.
    train_ratio: float - between 0-1, specifies how many percent of the
    tweets should be used for training/testing.
    use_n: int - considers only the first n tweets per annotator, regardless of
    annotator group.
    use_n_threshold: bool - True if early/late stage threshold over S, M, L for
    first <use_n> tweets should be used. Else the thresholds over all tweets of
    S and M is are used.
    max_i: int - maximum number of tweets to be used for training.

    """
    # Start with min_annos=1
    for min_annos in xrange(1, max_annos+1):
        # Results to be stored
        early_easy = []
        late_easy = []
        early_difficult = []
        late_difficult = []
        training_sizes = []
        print "minimum annotators per tweet", min_annos
        # Minimum training set size is 5
        for i in xrange(5, max_i+1):
            training_sizes.append(i)
            print "Institution: {}".format(inst_name)
            print "----------------"
            print "training set size", i
            agg = "raw"
            if cleaned:
                agg = "cleaned"
            thresholds = {
                "S": THRESHOLD,
                "M": THRESHOLD,
                "L": THRESHOLD,
                inst_name: THRESHOLD
            }

            # Load tweets from early stage
            # ----------------------------
            inst_annos, group_annos = read_dataset(
                dbs, db_idxs, inst_name, thresholds,
                anno_coll_name=anno_coll_name, tweet_coll_name=tweet_coll_name,
                cleaned=cleaned, min_annos=min_annos,
                is_early=True, use_n=use_n, use_n_threshold=use_n_threshold)
            early = {
                inst_name: get_min_i(inst_annos)
            }
            for group in group_annos:
                early[group] = get_min_i(group_annos[group])
            # print "Maximum possible i for early"
            # print "S:", early["S"]
            # print "M:", early["M"]
            # print "Institution:", early[inst_name]

            if use_n_threshold:
                fname = "{}_min_annos_{}_difficulty_labels_early_train_ratio_" \
                        "{}_use_n_{}_{}.txt"\
                    .format(inst_name, min_annos, train_ratio, use_n, agg)
            else:
                fname = "{}_min_annos_{}_difficulty_labels_early_train_ratio_" \
                        "{}_{}.txt"\
                    .format(inst_name, min_annos, train_ratio, agg)
            label_dst = os.path.join(label_dir, fname)
            res, x = count_annos_with_at_least_i_tweets(label_dst, inst_annos,
                                                        i, is_easy=True)
            print "annos with at least {} easy tweets in early: {}"\
                .format(i, len(res))
            early_easy.append(len(res))
            res, x = count_annos_with_at_least_i_tweets(label_dst, inst_annos,
                                                        i, is_easy=False)
            print "annos with at least {} difficult tweets in early: {}"\
                .format(i, len(res))
            early_difficult.append(len(res))

            # Load tweets from late stage
            # ---------------------------
            inst_annos, group_annos = read_dataset(
                dbs, db_idxs, inst_name, thresholds, anno_coll_name=anno_coll_name,
                tweet_coll_name=tweet_coll_name, cleaned=cleaned,
                min_annos=min_annos,
                is_early=False, use_n=use_n, use_n_threshold=use_n_threshold)
            late = {
                inst_name: get_min_i(inst_annos)
            }
            # print remove_annos(inst_annos, 20)
            for group in group_annos:
                # print group
                late[group] = get_min_i(group_annos[group])
                # print remove_annos(group_annos[group], 20)
            # print "Maximum possible i for late"
            # print "S:", late["S"]
            # print "M:", late["M"]
            # print "Institution:", late[inst_name]
            res, x = count_annos_with_at_least_i_tweets(label_dst, inst_annos,
                                                     i, is_easy=True)
            print "annos with at least {} easy tweets in late: {}".format(i,
                                                                          len(res))
            late_easy.append(len(res))
            res, x = count_annos_with_at_least_i_tweets(label_dst, inst_annos,
                                                     i, is_easy=False)
            print "annos with at least {} difficult tweets in late: {}" \
                .format(i, len(res))
            late_difficult.append(len(res))

            # min_i = min(late.values() + early.values())
            # print "set i={}".format(min_i)
        print "easy, early", early_easy
        print "difficult, early", early_difficult
        print "easy, late", late_easy
        print "difficult, late", late_difficult
        print "training set sizes", training_sizes
        if use_n_threshold:
            fname = "{}_min_annos_{}_min_tweets_{}_eligible_annos_early_" \
                    "train_ratio_{}_use_n_{}_{}.txt" \
                .format(inst_name, min_annos, i, train_ratio, use_n, agg)
        else:
            fname = "{}_min_annos_{}_min_tweets_{}_eligible_annos_early_train_" \
                    "ratio_{}_{}.txt" \
                .format(inst_name, min_annos, i, train_ratio, agg)
        dst = os.path.join(analysis_dir, fname)
        store_annos(early_easy, early_difficult, late_easy, late_difficult,
                    training_sizes, dst)


def store_annos(early_easy, early_difficult, late_easy, late_difficult,
                training_sizes, dst):
    """
    Stores how many annotators are eligible for the experiment per stratum.

    Parameters
    ----------
    early_easy: list - eligible annotators in stratum.
    early_difficult: list - eligible annotators in stratum.
    late_easy: list - eligible annotators in stratum.
    late_difficult: list - eligible annotators in stratum.
    training_sizes: list - list of training sizes used for the strata.
    dst: string - path where results will be stored.

    """
    with open(dst, "wb") as f:
        # Header
        header = "stratum,"
        for train_size in training_sizes:
            header += str(train_size) + ","
        # Delete last ","
        header = header[:-1]
        f.write(header + "\n")
        line = "early_easy,"
        for annos in early_easy:
            line += str(annos) + ","
        f.write(line[:-1] + "\n")
        line = "early_difficult,"
        for annos in early_difficult:
            line += str(annos) + ","
        f.write(line[:-1] + "\n")
        line = "late_easy,"
        for annos in late_easy:
            line += str(annos) + ","
        f.write(line[:-1] + "\n")
        line = "late_difficult,"
        for annos in late_difficult:
            line += str(annos) + ","
        f.write(line[:-1] + "\n")


def get_min_i(annos):
    """
    Returns the minimum i, i.e. the lowest number of tweets an annotator
    labeled.

    Parameters
    ----------
    annos: list of anno.Annotator - annotators who labeled tweets.

    Returns
    -------
    int.
    i - minimum number of labeled tweets by any of <annos>.

    """
    count = 1000
    for anno in annos:
        tweets = len(anno.tweets)
        if tweets < count:
            count = tweets
    return count


def remove_annos(annos, n):
    """
    Remove annotators that labeled less than n tweets.

    Parameters
    ----------
    annos: list of anno.Annotator - annotators who labeled tweets.
    n: int - minimum number of tweets that each annotator must have labeled.

    Returns
    -------
    str.
    Number of annotators that are kept having labeled at least n tweets.

    """
    keep = 0
    for anno in annos:
        tweets = len(anno.tweets)
        if tweets >= n:
            keep += 1
    return "{}/{} are kept when at least {} tweets must be labeled"\
        .format(keep, len(annos), n)


def count_annos_with_at_least_i_tweets(label_dst, annos, i, is_easy=True):
    """
    Counts how many annotators exist that labeled at least <i> tweets in the
    given stage of a given difficulty.

    Parameters
    ----------
    label_dst: str - path to file in which the tweet difficulty labels are
    stored.
    annos: list of anno.Annotator - list of annotators.
    i: int - minimum number of tweets that must exist in the given stage
    (early/late) of a given difficulty(easy/difficult).
    is_easy: bool - True if tweet difficulty is easy. Else it's difficult.

    Returns
    -------
    int, list of anno.Annotator.
    Number of annotators meeting these requirements. Annotators who meet the
    requirements, i.e. who labeled at least i easy/difficult tweets in the
    given stage (early/late - this info is already encoded in <annos>).

    """
    # [Counter of difficulty labels for anno1 who exceeds i,
    # ...anno2 who exceeds i, ...]
    res = []
    # List of annotators who labeled at least i tweets of the given difficulty
    filtered_annos = []
    diff_labels = read_labels(label_dst)
    # For each annotator
    for anno in annos:
        labels = []
        # For each tweet
        for tid in anno.tweets.ids:
            # Add difficulty label
            if tid in diff_labels:
                labels.append(diff_labels[tid])
        total_labels = Counter(labels)
        # Check if annotator meets requirements - if so, add her Counter
        if is_easy:
            if total_labels["easy"] >= i:
                res.append(total_labels)
                filtered_annos.append(anno)
        else:
            if total_labels["difficult"] >= i:
                res.append(total_labels)
                filtered_annos.append(anno)
    return res, filtered_annos


def read_labels(fp):
    """
    Reads in a file that stores the tweet difficulty labels.

    Parameters
    ----------
    fp: str - path to file.

    Returns
    -------
    dict.
    {tweet_id: difficulty_label}

    """
    tweets = {}
    with open(fp, "rb") as f:
        lines = f.readlines()
    # Skip header
    for line in lines[1:]:
        tid, label, _, _, _, _ = line.split(",")
        tweets[tid] = label
    return tweets


def plot_early_vs_late_all(stat_dir, fig_dir, similarity, cleaned=False,
                           use_n=50, use_n_threshold=True):
    """
    Creates plots for all simulation results automatically. Adds in one plot
    the performances of the classifiers trained on tweets from the early and
    late stage.

    Parameters
    ----------
    stat_dir: str - directory in which the results are stored as csv files.
    fig_dir: str - directory in which the plots should be stored.
    similarity: str - name of the metric to be displayed: "subsequence" for
    longest common subsequence, "substring" for longest common substring,
    "edit" for edit distance.
    cleaned: bool - True if the data should be cleaned, i.e. if tweet is
    "irrelevant", its remaining labels are ignored for computing average
    annotation times.
    use_n: int - considers only the first n tweets per annotator, regardless of
    annotator group.
    use_n_threshold: bool - True if early/late stage threshold over S, M, L for
    first <use_n> tweets should be used. Else the thresholds over all tweets of
    S and M is are used.

    """
    agg = "raw"
    if cleaned:
        agg = "cleaned"
    training_set_sizes = [x for x in xrange(150)]
    insts = ["md", "su"]
    weights = ["uniform", "distance"]
    metrics = ["edit", "subsequence", "substring"]
    difficulties = ["easy", "difficult"]
    for inst in insts:
        for difficulty in difficulties:
            for training_size in training_set_sizes:
                for metric in metrics:
                    for weight in weights:
                        if use_n_threshold:
                            src_early_name = \
                                "{}_{}_tweets_early_stage_results_sim_metric_" \
                                "{}_weight_{}_training_size_{}_min_annos_{}_" \
                                "use_n_{}_{}.txt"\
                                .format(inst, difficulty, metric, weight,
                                        training_size, min_annos, use_n, agg)
                            src_late_name = \
                                "{}_{}_tweets_late_stage_results_sim_metric_" \
                                "{}_weight_{}_training_size_{}_min_annos_{}_" \
                                "use_n_{}_{}.txt" \
                                .format(inst, difficulty, metric, weight,
                                        training_size, min_annos, use_n, agg)
                            dst_name = \
                                "{}_{}_tweets_results_sim_metric_{}_" \
                                "weight_{}_training_size_{}_min_annos_{}_" \
                                "early_vs_late_use_n_{}_{}.pdf" \
                                    .format(inst, difficulty, metric, weight,
                                            training_size, min_annos, use_n,
                                            agg)
                        else:
                            src_early_name = \
                                "{}_{}_tweets_early_stage_results_sim_metric_" \
                                "{}_weight_{}_training_size_{}_min_annos_{}_" \
                                "{}.txt" \
                                .format(inst, difficulty, metric, weight,
                                        training_size, min_annos, agg)
                            src_late_name = \
                                "{}_{}_tweets_late_stage_results_sim_metric_" \
                                "{}_weight_{}_training_size_{}_min_annos_{}" \
                                "_{}.txt" \
                                .format(inst, difficulty, metric, weight,
                                        training_size, min_annos, agg)
                            dst_name = \
                                "{}_{}_tweets_results_sim_metric_{}_" \
                                "weight_{}_training_size_{}_min_annos_{}_" \
                                "early_vs_late_{}.pdf" \
                                .format(inst, difficulty, metric, weight,
                                        training_size, min_annos, agg)

                        # File name for early stage
                        src_early = os.path.join(stat_dir, src_early_name)
                        # File name for late stage
                        src_late = os.path.join(stat_dir, src_late_name)
                        dst = os.path.join(fig_dir, dst_name)
                        # Plot if both input files exist
                        if os.path.isfile(src_early) \
                                and os.path.isfile(src_late):
                            plot_early_vs_late(src_early, src_late, dst,
                                               metric, similarity)


def plot_easy_vs_difficult_all(stat_dir, fig_dir, similarity, cleaned=False,
                               use_n=50, use_n_threshold=True):
    """
    Creates plots for all simulation results automatically. Adds in one plot
    the performances of the classifiers trained on easy and difficult tweets
    in the same stage.

    Parameters
    ----------
    stat_dir: str - directory in which the results are stored as csv files.
    fig_dir: str - directory in which the plots should be stored.
    similarity: str - name of the metric to be displayed: "subsequence" for
    longest common subsequence, "substring" for longest common substring,
    "edit" for edit distance.
    cleaned: bool - True if the data should be cleaned, i.e. if tweet is
    "irrelevant", its remaining labels are ignored for computing average
    annotation times.
    use_n: int - considers only the first n tweets per annotator, regardless of
    annotator group.
    use_n_threshold: bool - True if early/late stage threshold over S, M, L for
    first <use_n> tweets should be used. Else the thresholds over all tweets of
    S and M is are used.

    """
    agg = "raw"
    if cleaned:
        agg = "cleaned"
    training_set_sizes = [x for x in xrange(150)]
    insts = ["md", "su"]
    weights = ["uniform", "distance"]
    metrics = ["edit", "subsequence", "substring"]
    stages = ["early", "late"]
    for inst in insts:
        for stage in stages:
            for training_size in training_set_sizes:
                for metric in metrics:
                    for weight in weights:
                        if use_n_threshold:
                            src_early_name = \
                                "{}_easy_tweets_{}_stage_results_sim_metric_" \
                                "{}_weight_{}_training_size_{}_min_annos_{}_" \
                                "use_n_{}_{}.txt" \
                                .format(inst, stage, metric, weight,
                                        training_size, min_annos, use_n, agg)
                            src_late_name = \
                                "{}_difficult_tweets_{}_stage_results_sim_" \
                                "metric_{}_weight_{}_training_size_{}_min_" \
                                "annos_{}_use_n_{}_{}.txt" \
                                .format(inst, stage, metric, weight,
                                    training_size, min_annos, use_n, agg)
                            dst_name = \
                                "{}_tweets_{}_stage_results_sim_metric_{}_" \
                                "weight_{}_training_size_{}_min_annos_{}_" \
                                "easy_vs_difficult_use_n_{}_{}.pdf" \
                                .format(inst, stage, metric, weight,
                                        training_size, min_annos, use_n, agg)
                        else:
                            src_early_name = \
                                "{}_easy_tweets_{}_stage_results_sim_metric_" \
                                "{}_weight_{}_training_size_{}_min_annos_{}" \
                                "_{}.txt" \
                                .format(inst, stage, metric, weight,
                                        training_size, min_annos, agg)
                            src_late_name = \
                                "{}_difficult_tweets_{}_stage_results_sim_" \
                                "metric_{}_weight_{}_training_size_{}_min_" \
                                "annos_{}_{}.txt" \
                                .format(inst, stage, metric, weight,
                                        training_size, min_annos, agg)
                            dst_name = \
                                "{}_tweets_{}_stage_results_sim_metric_{}_" \
                                "weight_{}_training_size_{}_min_annos_{}_" \
                                "easy_vs_difficult_{}.pdf" \
                                .format(inst, stage, metric, weight,
                                        training_size, min_annos, agg)

                        # File name for easy tweets
                        src_early = os.path.join(stat_dir, src_early_name)
                        # File name for difficult tweets
                        src_late = os.path.join(stat_dir, src_late_name)

                        dst = os.path.join(fig_dir, dst_name)
                        # Plot if both input files exist
                        if os.path.isfile(src_early) \
                                and os.path.isfile(src_late):
                            plot_easy_vs_difficult(src_early, src_late, dst,
                                                   metric, similarity)


def difficulty_label_distribution(fp):
    """
    Counts how often "easy"/"difficult" was assigned to tweets.

    Parameters
    ----------
    fp: str - path to the file that stores the difficulty labels.

    Returns
    -------
    str.
    Raw counts and fraction of labels that are easy/difficult.

    """
    label_dic = read_labels(fp)
    raw_counts = Counter(label_dic.values())
    total = 1.0*sum(raw_counts.values())
    rel_counts = {}
    for label in raw_counts:
        rel_counts[label] = raw_counts[label] / total
    return "Raw: {} Total: {} Relative:{}".format(raw_counts, total, rel_counts)


if __name__ == "__main__":
    # Directory in which stats will be stored
    # Get the absolute path to the parent directory of /scripts/
    base_dir = os.path.abspath(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), os.pardir))
    diff_score_dir = os.path.join(
        base_dir, "results", "stats",
        "tweet_difficulty_label_reliability_simulation")
    # Directory in which statistical tests will be stored
    STAT_DIR = os.path.join(base_dir, "results", "stats",
                            "tweet_difficulty_label_reliability_simulation")
    # Directory in which figures will be stored
    FIG_DIR = os.path.join(base_dir, "results", "figures",
                           "difficulty_labels_early_late")
    tweet_label_dir = os.path.join(base_dir, "results", "stats",
                                   "difficulty_labels_early_late")
    analysis_dir = os.path.join(base_dir, "results", "stats",
                                "eligible_annos_early_late")
    # Directory in which the datasets are stored as csv
    DS_DIR = os.path.join(base_dir, "results", "export")
    # Path to the tweets of MD
    md_tweets = os.path.join(base_dir, DS_DIR, "tweets_hierarchical_md.csv")
    # Path to the annotators of MD
    md_annos = os.path.join(base_dir, DS_DIR, "annotators_hierarchical_md.csv")
    # Path to the tweets of SU
    su_tweets = os.path.join(base_dir, DS_DIR, "tweets_hierarchical_su.csv")
    # Path to the annotators of SU
    su_annos = os.path.join(base_dir, DS_DIR, "annotators_hierarchical_su.csv")

    if not os.path.exists(STAT_DIR):
        os.makedirs(STAT_DIR)
    if not os.path.exists(FIG_DIR):
        os.makedirs(FIG_DIR)
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)

    train_ratio = 0.4
    use_n = 50
    # True if thresholds over S, M, L (first <use_n> tweets per annotator)
    # should be used instead of S+M (all tweets)
    use_n_threshold = True

    # Number of tweets to be used for training
    i = 6
    # Maximum number of tweets in training set - all smaller values will be
    # tested
    max_i = 15
    # Maximum number of neighbors to consider
    k_max = 9
    # Minimum annotators per tweet (it assumes that the difficulty scores
    # were calculated for this in advance - see readme.txt
    min_annos = 2
    # Maximum number of annotators per tweet - smaller values are also tested.
    max_annos = 3
    # 1. Analyze how easy/difficult tweets are distributed in early/late stage
    # to set meaningful values for experiment
    ###################################################################
    analyze_data(su_tweets, su_annos, "su", analysis_dir,
                 tweet_label_dir, cleaned=True, train_ratio=train_ratio,
                 use_n=use_n, use_n_threshold=use_n_threshold, max_i=max_i,
                 max_annos=max_annos)

    analyze_data(md_tweets, md_annos, "md", analysis_dir,
                 tweet_label_dir, cleaned=True, train_ratio=train_ratio,
                 use_n=use_n, use_n_threshold=use_n_threshold, max_i=max_i,
                 max_annos=max_annos)

    # Actual simulation starts here
    ###############################

    #########
    # 1. MD #
    #########
    # # Stratum: easy tweets, early stage
    # run_simulation(DB_NAMES, MD_ALL, "md", STAT_DIR, THRESHOLD_DIR,
    #                tweet_label_dir, k_max, i, ANNO_COLL_NAME, TWEET_COLL_NAME,
    #                cleaned=True, train_ratio=train_ratio, min_annos=min_annos,
    #                is_early=True, is_easy=True, use_n=use_n,
    #                use_n_threshold=use_n_threshold)
    # # Stratum: difficult tweets, early stage
    # run_simulation(DB_NAMES, MD_ALL, "md", STAT_DIR, THRESHOLD_DIR,
    #                tweet_label_dir, k_max, i, ANNO_COLL_NAME, TWEET_COLL_NAME,
    #                cleaned=True, train_ratio=train_ratio, min_annos=min_annos,
    #                is_early=True, is_easy=False, use_n=use_n,
    #                use_n_threshold=use_n_threshold)
    # # Stratum: easy tweets, late stage
    # run_simulation(DB_NAMES, MD_ALL, "md", STAT_DIR, THRESHOLD_DIR,
    #                tweet_label_dir, k_max, i, ANNO_COLL_NAME, TWEET_COLL_NAME,
    #                cleaned=True, train_ratio=train_ratio, min_annos=min_annos,
    #                is_early=False, is_easy=True, use_n=use_n,
    #                use_n_threshold=use_n_threshold)
    # # Stratum: difficult tweets, late stage
    # run_simulation(DB_NAMES, MD_ALL, "md", STAT_DIR, THRESHOLD_DIR,
    #                tweet_label_dir, k_max, i, ANNO_COLL_NAME, TWEET_COLL_NAME,
    #                cleaned=True, train_ratio=train_ratio, min_annos=min_annos,
    #                is_early=False, is_easy=False, use_n=use_n,
    #                use_n_threshold=use_n_threshold)
    #
    # #########
    # # 2. SU #
    # #########
    # # Stratum: easy tweets, early stage
    # run_simulation(DB_NAMES, SU_ALL, "su", STAT_DIR, THRESHOLD_DIR,
    #                tweet_label_dir, k_max, i, ANNO_COLL_NAME, TWEET_COLL_NAME,
    #                cleaned=True, train_ratio=train_ratio, min_annos=min_annos,
    #                is_early=True, is_easy=True, use_n=use_n,
    #                use_n_threshold=use_n_threshold)
    # # Stratum: difficult tweets, early stage
    # run_simulation(DB_NAMES, SU_ALL, "su", STAT_DIR, THRESHOLD_DIR,
    #                tweet_label_dir, k_max, i, ANNO_COLL_NAME, TWEET_COLL_NAME,
    #                cleaned=True, train_ratio=train_ratio, min_annos=min_annos,
    #                is_early=True, is_easy=False, use_n=use_n,
    #                use_n_threshold=use_n_threshold)
    # # Stratum: easy tweets, late stage
    # run_simulation(DB_NAMES, SU_ALL, "su", STAT_DIR, THRESHOLD_DIR,
    #                tweet_label_dir, k_max, i, ANNO_COLL_NAME, TWEET_COLL_NAME,
    #                cleaned=True, train_ratio=train_ratio, min_annos=min_annos,
    #                is_early=False, is_easy=True, use_n=use_n,
    #                use_n_threshold=use_n_threshold)
    # # Stratum: difficult tweets, late stage
    # run_simulation(DB_NAMES, SU_ALL, "su", STAT_DIR, THRESHOLD_DIR,
    #                tweet_label_dir, k_max, i, ANNO_COLL_NAME, TWEET_COLL_NAME,
    #                cleaned=True, train_ratio=train_ratio, min_annos=min_annos,
    #                is_early=False, is_easy=False, use_n=use_n,
    #                use_n_threshold=use_n_threshold)

    # # Plot all files
    # # early stage vs. late stage
    # plot_early_vs_late_all(STAT_DIR, FIG_DIR, "f1", cleaned=True, use_n=use_n,
    #                        use_n_threshold=use_n_threshold)
    # # easy tweets vs. difficult tweets
    # plot_easy_vs_difficult_all(STAT_DIR, FIG_DIR, "f1", cleaned=True,
    #                            use_n=use_n, use_n_threshold=use_n_threshold)

    # Get how often each difficulty label is assigned for the paper
    # 1. For early stage
    # MD
    # fname = "md_min_annos_{}_difficulty_labels_early_train_ratio_0.4_use_n_50_cleaned.txt".format(min_annos)
    # fp = os.path.join(tweet_label_dir, fname)
    # print "MD - Early"
    # print difficulty_label_distribution(fp)
    # # SU
    # fname = "su_min_annos_{}_difficulty_labels_early_train_ratio_0.4_use_n_50_cleaned.txt".format(min_annos)
    # fp = os.path.join(tweet_label_dir, fname)
    # print "SU - Early"
    # print difficulty_label_distribution(fp)
    # # 2. For late stage
    # fname = "md_min_annos_{}_difficulty_labels_late_train_ratio_0.4_use_n_50_cleaned.txt".format(
    #     min_annos)
    # fp = os.path.join(tweet_label_dir, fname)
    # print "MD - Late"
    # print difficulty_label_distribution(fp)
    # # SU
    # fname = "su_min_annos_{}_difficulty_labels_late_train_ratio_0.4_use_n_50_cleaned.txt".format(
    #     min_annos)
    # fp = os.path.join(tweet_label_dir, fname)
    # print "SU - Late"
    # print difficulty_label_distribution(fp)

    # This piece of code only exists to quickly get the differences in the
    # F1-scores between specific strata for the SAC2018 paper
    # In that case 2 print statements must be uncommented in
    # plot_easy_vs_difficult()
    # early = os.path.join(STAT_DIR, "md_easy_tweets_late_stage_results_sim_metric_edit_weight_uniform_training_size_8_min_annos_2_use_n_50_cleaned.txt")
    # late = os.path.join(STAT_DIR, "md_difficult_tweets_late_stage_results_sim_metric_edit_weight_uniform_training_size_8_min_annos_2_use_n_50_cleaned.txt")
    # dst = os.path.join(FIG_DIR, "test.pdf")
    # plot_easy_vs_difficult(early, late, dst, "f1", "")
