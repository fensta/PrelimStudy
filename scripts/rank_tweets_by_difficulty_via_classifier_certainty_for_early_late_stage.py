"""
Ranks tweets according to classifier certainty when labeling tweets: if
certainty is low, a tweet is more difficult to label.

How to obtain label certainties?
- We have no ground truth, so we can build 2 classifiers per annotator.
- Build one classifier trained on tweets from early stage (EARLY)
and one on tweets from late stage (LATE).
- We have calculated the threshold separating early from late stage annotations,
by choosing the median value of the learning phase of the respective group or
institution.
- Store the probabilities for each tweet over all annotators and then
compute the average values which are used as classifier certainty indicators.
But compute those values per once per early tweets, once per late tweets, once
per merged tweets. Perform the same computation not only on an institution
level, but also per group.

"""
import operator
import os
import warnings
from collections import Counter
from copy import deepcopy

import pseudo_db
from anno import Annotator
from knn_proba import knn

# Default label assigned by annotator
EMPTY = ""
# Default value for annotation time
ZERO = 0
# Number of tweets for annotator group S
S = 50
# Number of tweets for annotator group M
M = 150
# Transition from early to late stage after how many tweets?
THRESHOLD = 25


def rank_by_classifier_certainty(
        stat_dir, md_tw_p, md_a_p, su_tw_p, su_a_p, cleaned=False,
        min_annos=1, is_early=True, train_ratio=0.66, use_n=50,
        use_n_threshold=True):
    """
    Ranks tweets by agreement per group and institution.
    Group L is ignored.

    Parameters
    ----------
    stat_dir: str - directory in which the stats will be stored.
    md_tw_p: str - path to MD tweets dataset in csv format.
    md_a_p: str - path to MD annotators dataset in csv format.
    su_tw_p: str - path to SU tweets dataset in csv format.
    su_a_p: str - path to SU annotators dataset in csv format.
    cleaned: bool - True if the cleaned data is used as input.
    min_annos: int - minimum number of annotators who must've assigned a label
    to a tweet. Otherwise it'll be discarded.
    is_early: bool - True if only tweets from the early phase should be
    considered. Else only tweets from the late stage are considered.
    train_ratio: float - between 0-1, specifies how many percent of the
    tweets should be used for training/testing.
    use_n: int - considers only the first n tweets per annotator, regardless of
    annotator group.
    use_n_threshold: bool - True if early/late stage threshold over S, M, L for
    first <use_n> tweets should be used. Else the thresholds over all tweets of
    S and M is are used.

    """
    agg = "raw"
    if cleaned:
        agg = "cleaned"
    tweet_type = "late"
    if is_early:
        tweet_type = "early"

    # For MD
    inst_name = "md"
    thresholds = {
        "S": THRESHOLD,
        "M": THRESHOLD,
        "L": THRESHOLD,
        inst_name: THRESHOLD
    }
    inst_annos, group_annos, inst_counter, group_counter = \
        read_dataset(md_tw_p, md_a_p, inst_name, thresholds, cleaned=cleaned,
                     min_annos=min_annos, is_early=is_early, use_n=use_n,
                     use_n_threshold=use_n_threshold)
    compute_certainties(inst_annos, group_annos, inst_counter, group_counter,
                        stat_dir, tweet_type, agg, min_annos, inst_name,
                        train_ratio)
    # For SU
    inst_name = "su"
    thresholds = {
        "S": THRESHOLD,
        "M": THRESHOLD,
        "L": THRESHOLD,
        inst_name: THRESHOLD
    }
    inst_annos, group_annos, inst_counter, group_counter = \
        read_dataset(su_tw_p, su_a_p, inst_name, thresholds, cleaned=cleaned,
                     min_annos=min_annos, is_early=is_early, use_n=use_n,
                     use_n_threshold=use_n_threshold)
    compute_certainties(inst_annos, group_annos, inst_counter, group_counter,
                        stat_dir, tweet_type, agg, min_annos, inst_name,
                        train_ratio)


def compute_certainties(inst_annos, group_annos, inst_counter, group_counter,
                        stat_dir, tweet_type, agg, min_annos, inst_name,
                        train_ratio):
    """
    Computes the (average) certainty of the kNN predictor for each tweet.

    Parameters
    ----------
    inst_annos: ist of Annotator objects - with their tweets separated into
    early and late stage.
    Counters only contain counts for tweets that were labeled sufficiently
    often.
    group_annos: - dict - (group names "S" and "M" are keys) with lists of
    Annotator objects per group as value with their tweets separated into
    early and late stage.
    inst_counter: Counter - for institution, i.e. how often each tweet was
    labeled. Counters return tuples (raw_count, normalized_count), where
    raw counts are divided by the max(raw_count) to obtain normalized counts.
    group_counter: Counter - for groups, i.e. how often each tweet was labeled
    in the group. Counters return tuples (raw_count, normalized_count), where
    raw counts are divided by the max(raw_count) to obtain normalized counts.
    stat_dir: str - directory in which the results will be stored.
    tweet_type: str - phase, either "early" or "late".
    agg: str - type of dataset, either "cleaned" or "raw".
    min_annos: int - minimum number of annotators who must've assigned a label
    to a tweet. Otherwise it'll be discarded.
    train_ratio: float - between 0-1, specifies how many percent of the
    tweets should be used for training/testing.

    """
    print "#TWEETS in INSTITUTION:", len(inst_counter)
    print "#ANNOS", len(inst_annos)
    print "#TWEETS in S:", len(group_counter["S"])
    print "#ANNOS", len(group_annos["S"])
    print "#TWEETS in M:", len(group_counter["M"])
    print "#ANNOS", len(group_annos["M"])
    # Store predictions per tweet for  each annotator because we need to
    # average them later on to obtain a certainty for each tweet
    # {tid:
    #     [
    #         # Labels of all annotators for 1st hierarchy level.
    #         [
    #             (label1_by_anno1 for level1, proba1), (label2_by_anno1 for
    #             level1, proba2), (label1_by_anno2 for level1, proba3), ...
    #
    #         ],
    #         # Labels of all annotators for 2nd hierarchy level.
    #         [
    #             (label1_by_anno1 for level2, proba1), (label2_by_anno1 for
    #             level2, proba2), (label1_by_anno2 for level2, proba3), ...
    #         ],
    #         # Labels of all annotators for 3rd hierarchy level.
    #         [
    #             (label1_by_anno1 for level3, proba1), (label2_by_anno1 for
    #             level3, proba2), (label1_by_anno2 for level3, proba3), ...
    #         ]
    #     ],...
    # }
    # Same as inst_predictions, but for each group, i.e. we have that dictionary
    # inside a dictionary for the group
    y_groups = {}
    for group in group_annos:
        # group_predictions[group] = {}
        dataset = group_annos[group]
        y_groups[group] = get_predictions(dataset, train_ratio)
    print "S predictions:", len(y_groups["S"])
    print y_groups["S"]
    print "M predictions:", len(y_groups["M"])
    print y_groups["M"]
    dataset = inst_annos
    y_inst = get_predictions(dataset, train_ratio)
    print "inst predictions:", len(y_inst)
    print y_inst

    # Aggregate predictions
    # a) For institution
    certainties = compute_average_certainty(y_inst)
    # Sort descendingly according to certainties and store results
    # For institution
    certain_all_inst = sorted(certainties.items(),
                              key=operator.itemgetter(1), reverse=True)
    if use_n_threshold:
        f_name = "{}_min_annos_{}_ranked_by_classifier_certainty_{}_train" \
                 "_ratio_{}_use_n_{}_{}.txt" \
            .format(inst_name, min_annos, tweet_type, train_ratio, use_n, agg)
    else:
        f_name = "{}_min_annos_{}_ranked_by_classifier_certainty_{}_train" \
                 "_ratio_{}_{}.txt"\
            .format(inst_name, min_annos, tweet_type, train_ratio, agg)
    dst = os.path.join(stat_dir, f_name)
    store_data(certain_all_inst, dst, inst_counter)

    # Aggregate predictions per group
    for group in y_groups:
        certainties = compute_average_certainty(y_groups[group])
        sorted_group_certainties = sorted(certainties.items(),
                                          key=operator.itemgetter(1),
                                          reverse=True)
        if use_n_threshold:
            f_name = "{}_{}_min_annos_{}_ranked_by_classifier_certainty_{}_" \
                     "train_ratio_{}_use_n_{}_{}.txt" \
                .format(inst_name, group, min_annos, tweet_type, train_ratio,
                        use_n, agg)
        else:
            f_name = "{}_{}_min_annos_{}_ranked_by_classifier_certainty_{}_" \
                     "train_ratio_{}_{}.txt"\
                .format(inst_name, group, min_annos, tweet_type, train_ratio,
                        agg)
        dst = os.path.join(stat_dir, f_name)
        store_data(sorted_group_certainties, dst, group_counter[group])


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

                if is_early:
                    # if anno_time <= thresholds[group]:
                    #     group_anno.add_tweet(tweet_id, anno_time, labels,
                    #                          text)
                    # if anno_time <= thresholds[inst_name]:
                    #     inst_anno.add_tweet(tweet_id, anno_time, labels,
                    #                         text)
                    if idx < thresholds[group]:
                        group_anno.add_tweet(tweet_id, anno_time,
                                             labels, text)
                    if idx < thresholds[inst_name]:
                        inst_anno.add_tweet(tweet_id, anno_time, labels,
                                            text)
                # Use only tweets from late stage
                else:
                    # if anno_time > thresholds[group]:
                    #     group_anno.add_tweet(tweet_id, anno_time, labels,
                    #                          text)
                    # if anno_time > thresholds[inst_name]:
                    #     inst_anno.add_tweet(tweet_id, anno_time, labels,
                    #                         text)
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

    # Normalize counts, i.e. divide each count by the maximum count to get
    # values between 0-1
    norm_inst_counts = normalize_counts(inst_counts)
    norm_group_counts = {
        "S": normalize_counts(group_counts["S"]),
        "M": normalize_counts(group_counts["M"])
    }
    return inst_annos, group_annos, norm_inst_counts, norm_group_counts


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


def normalize_counts(counts):
    """
    Normalizes counts by dividing all counts by the maximum count.

    Parameters
    ----------
    counts: dict - dictionary to be normalized.

    Returns
    -------
    dict.
    Contains same keys as <counts>, but a tuple as value which is comprised of
    the raw count and the normalized count.

    """
    normalized = {}
    # There could be no tweets that were labeled sufficiently often
    if len(counts.values()) > 0:
        factor = 1.0 / max(counts.itervalues())
        # Special case for MD group M
        for tid, count in counts.iteritems():
            normalized[tid] = (count, count * factor)
    return normalized


def get_predictions(annos, train_ratio):
    """
    Predict the labels of the dataset.
    Build one classifier trained on early tweets and one trained on late tweets.
    Use the same number k of tweets for training, i.e. k = min(#early tweets,
    #late tweets). Predict the remaining labels and groups predicted labels by
    tweets.

    Parameters
    ----------
    annos: list of anno.Annotator - each object represents an annotator with her
    early/late tweets.
    train_ratio: float - between 0-1, specifies how many percent of the
    tweets should be used for training/testing.


    Returns
    -------
    dict, dict, dict.
    For each tweet ID: list of tuples containing predicted labels and their
    probabilities. It returns such a dictionary for merged predictions
    (EARLY+LATE), for those predicted by the EARLY classifier, and for those
    predicted by the LATE classifier.

    """
    # {tid:
    #     [
    #         # Labels of all annotators for 1st hierarchy level.
    #         [
    #             (label1_by_anno1 for level1, proba1), (label2_by_anno1 for
    #             level1, proba2), (label1_by_anno2 for level1, proba3), ...
    #
    #         ],
    #         # Labels of all annotators for 2nd hierarchy level.
    #         [
    #             (label1_by_anno1 for level2, proba1), (label2_by_anno1 for
    #             level2, proba2), (label1_by_anno2 for level2, proba3), ...
    #         ],
    #         # Labels of all annotators for 3rd hierarchy level.
    #         [
    #             (label1_by_anno1 for level3, proba1), (label2_by_anno1 for
    #             level3, proba2), (label1_by_anno2 for level3, proba3), ...
    #         ]
    #     ]
    # }
    predictions = {}
    for anno in annos:
        # Determine training set size which should be 66% of the data
        n = len(anno)
        print "labeled tweets:", n, len(anno.tweets)
        k = int(round(train_ratio * n))
        print "use for training:", k
        print "use for testing", n - k
        if n < 2:
            warnings.warn("Can't build classifier for {}: only {} training "
                          "instances are given.".format(anno, n))
        else:
            # -------------------
            # Choose first k tweets from early stage as training set and the
            # rest as test set
            # List of texts
            train_anno = deepcopy(anno)
            # Keep first k tweets as training set, use the rest as
            # test set
            train_anno.keep_k(k)
            # print "use {} tweets for training".format(len(train_anno))

            # Keep the remaining tweets
            test_anno = deepcopy(anno)
            # Keep k tweets in late stage as training set, use the rest as
            # test set
            test_anno.keep_rest(k)
            build_classifier(train_anno, test_anno)
            preds = test_anno.get_predictions()
            print "#predictions", len(preds)

            # 3. Add predictions
            update_predictions(predictions, preds)
            print "#total predictions", len(predictions)
    return predictions


def update_predictions(total_preds, preds):
    """
    Adds predictions from <preds> to the dictionary <total_preds> which holds
    all predictions.

    Parameters
    ----------
    total_preds: dict - to be updated.
    preds:: dict - holds predictions that should be added to <total_preds>.

    """
    for tid in preds:
        if tid not in total_preds:
            total_preds[tid] = [
                [],  # All labels for 1st level
                [],  # All labels for 2nd level
                []  # All labels for 3rd level
            ]
        for lvl_idx, lvl in enumerate(preds[tid]):
            for tpl in lvl:
                total_preds[tid][lvl_idx].append(tpl)


def build_classifier(train, test):
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
    y_test = test.get_labels_per_level()
    y_test_first_set = y_test[0]
    y_test_second_set = y_test[1]
    y_test_third_set = y_test[2]

    # c) Get predictions for test data
    # Predictions for first level
    y_pred1 = knn(X_train, y_train_first_set, X_test, recency=False,
                  keep_na=True, similarity="edit", weights="uniform",
                  k=3)
    # Predictions for second level
    y_pred2 = knn(X_train, y_train_second_set, X_test, recency=False,
                  keep_na=True, similarity="edit", weights="uniform",
                  k=3)
    # Predictions for third level
    y_pred3 = knn(X_train, y_train_third_set, X_test, recency=False,
                  keep_na=True, similarity="edit", weights="uniform",
                  k=3)

    # Concatenate the i-th entry of each hierarchy level together as a list
    # which represents the labels of a single tweet
    preds = map(list, zip(y_pred1, y_pred2, y_pred3))

    # Store predictions
    test.set_predictions(preds)


def compute_average_certainty(inst_predictions):
    """
    Computes average certainty of classifier for predicted tweets. The function
    assumes that for each tweet a label exist on each level of the hierarchy.
    We first average the certainties per hierarchy level and then compute the
    average over the most probable label on each level to obtain an average
    classifier certainty. Ignores tweets that were labeled by less than
    <min_annos>.

    Parameters
    ----------
    inst_predictions: dict - tuples of (labels, certainty) per hierarchy level:
    {tid:
        [
            # Labels of all annotators for 1st hierarchy level.
            [
                (label1_by_anno1 for level1, proba1), (label2_by_anno1 for
                level1, proba2), (label1_by_anno2 for level1, proba3), ...

            ],
            # Labels of all annotators for 2nd hierarchy level.
            [
                (label1_by_anno1 for level2, proba1), (label2_by_anno1 for
                level2, proba2), (label1_by_anno2 for level2, proba3), ...
            ],
            # Labels of all annotators for 3rd hierarchy level.
            [
                (label1_by_anno1 for level3, proba1), (label2_by_anno1 for
                level3, proba2), (label1_by_anno2 for level3, proba3), ...
            ]
        ],...
    }

    Returns
    -------
    dict.
    For each tweet ID a the average classifier certainty.

    """
    # Note that the tuples are sorted descendingly per level
    # Probabilities aggregated per level
    # {
    #     tid: {
    #         # Level 1
    #         [[(label1, avg. certainty), (label2, avg. certainty)...]],
    #         # Level 2
    #         [[(label1, avg. certainty), (label2, avg. certainty)...]],
    #         # Level 3
    #         [[(label1, avg. certainty), (label2, avg. certainty)...]],
    #     }
    # }
    level_wise = {}
    # Total certainties aggregated over all levels
    # {
    #     tid: average certainty,
    # }
    total = {}
    for tid in inst_predictions:
        # First, average results per hierarchy level
        level_wise[tid] = []
        for lvl_idx, level in enumerate(inst_predictions[tid]):
            # print inst_predictions[tid][level]
            # totals = Counter([tpl[0] for tpl in level])
            # print level
            # print "totals for level", lvl_idx
            # print totals
            # Count how often each label was assigned
            per_class = {}
            for label, certainty in level:
                if label not in per_class:
                    per_class[label] = 0.0
                per_class[label] += certainty
            avg_per_class = {}
            total_sum = sum(per_class.values())
            # print "total", total_sum
            # Now average probabilities
            for label in per_class:
                avg_per_class[label] = per_class[label] / total_sum
            # print "per class", per_class
            # print "avg per class", avg_per_class
            # Sort averaged probabilities descendingly
            sorted_probs = \
                sorted(avg_per_class.items(), key=operator.itemgetter(1),
                       reverse=True)
            level_wise[tid].append(sorted_probs)
            # print "sorted", level_wise[tid]
            # print "per class", per_class
            # print "avg per class", avg_per_class
            # print "sorted", level_wise
        # Now average over all levels, but use only the most probable label from
        # each level
        average_certainty = 0.0
        # Tweet might not have been labeled by a sufficient number of annos
        if tid in level_wise:
            # print "avg per level", level_wise[tid]
            for level in level_wise[tid]:
                # Only use most certain label
                # print "certainties", level_wise[tid][level]
                # print "levelwise", level
                highest_prob = level[0][1]
                average_certainty += highest_prob
            total[tid] = average_certainty / 3
    return total


def store_data(data, p, counts):
    """
    Stores the data in a .txt file, separated by commas. Stores tweet ID,
    classifier certainty and how often the tweet was labeled.
    Uses 3-digit precision after comma.

    Parameters
    ----------
    data: list of tuples - each tuple represents a tweet ID, and a tuple with
    its agreement, and total number of annotators who labeled it.
    p: str - path where output file should be stored.
    counts: dict - for each tweet ID it's counted how often the tweet was
    labeled. Values are tuples of raw and relative counts.

    """
    with open(p, "wb") as f:
        f.write("tweet_id,weighted_agreement,raw_counts,relative_counts\n")
        for tupl in data:
            # tupl=(Tweet ID, weighted agreement)
            tweet_id = tupl[0]
            certainty = tupl[1]
            count = counts[tweet_id]
            f.write("{},{:.3f},{},{:.3f}\n".format(tweet_id, certainty,
                                                   count[0], count[1]))


def test_compute_avg_agreement():
    """Tests if compute_avg_agreement() computes agreement correctly"""
    # 0.666
    t1 = [Counter(["a", "a", "b"]), Counter(["c", "c", "e"])]
    # 0.5
    t2 = [Counter(["a", "a", "b", "b"])]
    # 0.111
    t3 = [Counter(["a", "c", "b"]), Counter(["d", "e", "f"])]
    # 0.249
    t4 = [Counter(["a", "c", "b"]), Counter(["d", "d", "e"]),
          Counter(["f", "g", "h"])]
    # 1
    t5 = [Counter(["a", "a", "a"]), Counter(["d", "d", "d"]),
          Counter(["f", "f", "f"])]
    # 0.649
    t6 = [Counter(["a", "a", "a", "b"]), Counter(["d", "d", "e", "e"])]
    # 0.5
    t7 = [Counter(["a", "a", "a", "b", "b", "b"]),
          Counter(["d", "d", "e", "e"])]
    # 0.583
    t8 = [Counter(["a", "a", "b"]), Counter(["c"])]
    tweets = {"1": t1, "2": t2, "3": t3, "4": t4, "5": t5, "6": t6, "7": t7,
              "8": t8}


if __name__ == "__main__":
    # Get the absolute path to the parent directory of /scripts/
    base_dir = os.path.abspath(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), os.pardir))

    # Use the first 50 annotated tweets of each annotator
    n = 50

    # Directory in which the datasets are stored as csv
    DS_DIR = os.path.join(base_dir, "results", "export")
    # Directory in which statistical tests will be stored
    STAT_DIR = os.path.join(base_dir, "results", "stats",
                            "rank_tweets_by_classifier_certainty_early_late")
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
    names = ["md", "su"]
    # Minimum number of annotators who labeled a tweet for it to be considered
    # in the analysis
    min_votes = 1
    # Percentage of tweets of an annotator to be used as the training set
    train_ratio = 0.4
    # Consider only <use_n> first tweets of each annotator
    use_n = 50
    # True if thresholds over S, M, L (first <use_n> tweets per annotator)
    # should be used instead of S+M (all tweets)
    use_n_threshold = True
    DEBUG = False
    if DEBUG:
        test_compute_avg_agreement()
    else:
        # 1. Build classifier EARLY which uses only tweets from EARLY stage
        # For each file
        rank_by_classifier_certainty(
            STAT_DIR, md_tweets, md_annos, su_tweets, su_annos, cleaned=True,
            min_annos=min_votes, is_early=True, train_ratio=train_ratio,
            use_n=use_n, use_n_threshold=use_n_threshold)

        # 2. Build classifier LATE which uses only tweets from LATE stage
        # For each file
        rank_by_classifier_certainty(
            STAT_DIR, md_tweets, md_annos, su_tweets, su_annos, cleaned=True,
            min_annos=min_votes, is_early=False, train_ratio=train_ratio,
            use_n=use_n, use_n_threshold=use_n_threshold)
