"""
Ranks tweets according to agreement of annotators: if label agreement on a tweet
is low, it's more difficult to label.

How to get 1 agreement value per tweet?
see f()

"""
import os
import warnings
from collections import Counter
import operator

import pseudo_db
from anno import Annotator

# Default label assigned by annotator
EMPTY = ""
# Default value for annotation time
ZERO = 0
# Number of tweets for annotator group S
S = 50
# Number of tweets for annotator group M
M = 150
# Transition from early to late stage
THRESHOLD = 25


def rank_by_agreement(
        md_tw_p, md_a_p, su_tw_p, su_a_p, stat_dir, cleaned=False,
        min_annos=1, is_early=True, use_n=50, use_n_threshold=True):
    """
    Ranks tweets by agreement per group and institution.
    Group L is ignored.

    Parameters
    ----------
    md_tw_p: str - path to MD tweets dataset in csv format.
    md_a_p: str - path to MD annotators dataset in csv format.
    su_tw_p: str - path to SU tweets dataset in csv format.
    su_a_p: str - path to SU annotators dataset in csv format.
    stat_dir: str - directory in which the stats will be stored.
    of the annotation process are stored per group/institution.
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
    print "S", len(group_annos["S"])
    compute_agreements(inst_annos, group_annos, inst_counter, group_counter,
                       stat_dir, tweet_type, agg, min_annos, inst_name,
                       use_n_threshold)

    # For SU
    inst_name = "su"
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
    print "S", len(group_annos["S"])
    compute_agreements(inst_annos, group_annos, inst_counter, group_counter,
                       stat_dir, tweet_type, agg, min_annos, inst_name,
                       use_n_threshold)


def compute_agreements(inst_annos, group_annos, inst_counter, group_counter,
                       stat_dir, tweet_type, agg, min_annos, inst_name,
                       use_n_threshold):
    """
    Computes the (average) agreement of the annotators for each tweet.

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
    use_n_threshold: bool - True if early/late stage threshold over S, M, L for
    first <use_n> tweets should be used. Else the thresholds over all tweets of
    S and M is are used.

    """
    if use_n_threshold:
        groups = {
            "S": convert(group_annos["S"]),
            "M": convert(group_annos["M"]),
            "L": convert(group_annos["L"])
        }
    else:
        groups = {
            "S": convert(group_annos["S"]),
            "M": convert(group_annos["M"])
        }
    inst = convert(inst_annos)
    inst_agreement = compute_weighted_agreement(inst)
    # Sort descendingly according to agreement and store results
    # For institution
    sorted_inst_agreement = sorted(inst_agreement.items(),
                                   key=operator.itemgetter(1), reverse=True)
    if use_n_threshold:
        f_name = "{}_min_annos_{}_ranked_by_agreement_{}_use_n_{}_{}.txt" \
            .format(inst_name, min_annos, tweet_type, use_n, agg)
    else:
        f_name = "{}_min_annos_{}_ranked_by_agreement_{}_{}.txt" \
            .format(inst_name, min_annos, tweet_type, agg)
    dst = os.path.join(stat_dir, f_name)
    store_data(sorted_inst_agreement, dst, inst_counter)
    # For groups
    for g in groups:
        if use_n_threshold:
            f_name = "{}_{}_min_annos_{}_ranked_by_agreement_{}_use_n_{}_" \
                     "{}.txt" \
                .format(inst_name, g, min_annos, tweet_type, use_n, agg)
        else:
            f_name = "{}_{}_min_annos_{}_ranked_by_agreement_{}_{}.txt" \
                .format(inst_name, g, min_annos, tweet_type, agg)
        group_agreement = compute_weighted_agreement(groups[g])
        sorted_group_agreement = sorted(group_agreement.items(),
                                        key=operator.itemgetter(1),
                                        reverse=True)
        dst = os.path.join(stat_dir, f_name)
        store_data(sorted_group_agreement, dst, group_counter[g])


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


def convert(annos):
    """
    Converts Annotator objects into a datastructure (remainder of a previous
    attempt) for which the weighted agreement can be computed.
    IMPORTANT: only consider tweets with > 1 annotator because otherwise
    there'll be always 100% agreement.

    Parameters
    ----------
    annos: list of anno.Annotator - list of annotators.

    Returns
    -------
    Datastructure that is needed for calculating the average weighted label
    agreement per tweet.
    {tweet_id: Counter per label1, Counter per label2, Counter per label3}


    """
    # {tid: [[all labels for level 1], [all labels for level 2],
    # [all labels for level 3]]}
    # First create for each tweet ID a list of labels
    converted = {}
    for anno in annos:
        labels = anno.get_labels()
        for tid in labels:
            if tid not in converted:
                converted[tid] = [[], [], []]
            converted[tid][0].append(labels[tid][0])
            converted[tid][1].append(labels[tid][1])
            converted[tid][2].append(labels[tid][2])
    # Second, convert those lists into counters
    res = {}
    for tid in converted:
        number_of_annos = sum(Counter(converted[tid][0]).values())
        print "labels", Counter(converted[tid][0])
        print "annos", number_of_annos
        if number_of_annos > 1:
            res[tid] = [[], [], []]
            res[tid][0] = Counter(converted[tid][0])
            res[tid][1] = Counter(converted[tid][1])
            res[tid][2] = Counter(converted[tid][2])
    return res


def compute_weighted_agreement(tweets):
    """
    Computes the average agreement over all hierarchy levels for each tweet.

    How to get 1 agreement value per tweet?
    see f()

    Parameters
    ----------
    tweets: dict - each key is a tweet ID and it has a list of up to 3 Counters,
    one per hierarchy level (some levels were never assigned, so they might
    not exist).

    Returns
    -------
    dict.
    Tweet IDs as keys and the average agreement over all hierarchy
    levels.

    """
    results = {}
    # max_annos = 0
    for tid in tweets:
        # print "tweet:", t
        agreement, _ = f(tweets[tid])
        # if annos > max_annos:
        #     max_annos = annos
        # print "normalized agreement per level:", agreement
        results[tid] = agreement
    return results


def f(tweet):
    """
    Computes the agreement for a given tweet. The agreement of each hierarchy
    level contributes to the final agreement according to the number of people
    who agree on the majority label.

    That implies that the higher levels in the hierarchy contribute more to
    agreement (it's also easier to assign those).

    Weighted agreement formula:
    Agreement = agree1/total_agree * agree1/level_labels1 +
    agree2/total_agree * agree2/level_labels2 + agree3/total_agree *
    agree3/level_labels3

    - agreeX: number of annotators on level X that agree on the label
    - total_agree: sum of all annotators across all levels that agree on the
    labels
    - level_labelsX: number of annotators who assigned a label on level X
    NOTE THAT in a tweet the levels with mixed labels (= no majority) decrease
    the influence of levels having a majority (for each such level, total_agree
    is increased by 1)


    Parameters
    ----------
    tweet: list of Counter - each Counter holds the counts for the labels of
    that level in the hierarchy.

    Returns
    -------
    float, int.
    Level of average agreement for the given tweet, number of annotators who
    labeled the tweet

    """
    # Count of all annotators that agree on majority labels over all levels
    agree_total = 0
    # Total votes of annotators on each hierarchy level
    level_labels = []
    # Number of annotators agreeing on the label per hierarchy level
    agrees = []
    annos = 0
    for idx, level in enumerate(tweet):
        # print "labels:", level
        votes_per_level = sum(level.values())
        # Count annotators who labeled a tweet
        if idx == 0:
            annos = votes_per_level
        level_labels.append(votes_per_level)
        # Number of annotators agreeing on the label
        votes = max(level.values())
        agree = 1.0 * votes
        agree_total += agree
        # If a level has mixed labels (= no majority label), penalize it by
        # increasing the contribution of such levels
        if abs(agree / votes_per_level - 0.5) < 0.001:
            agree = 1.0
            agree_total += 1
            # print "found a mixed level - penalize tweet"
        agrees.append(agree)

    # Weight by which the agreement of each level is weighed
    agree_weights = []
    for votes in agrees:
        weight = 0
        if agree_total > 0:
            weight = 1.0 * votes / agree_total
        agree_weights.append(weight)
        # print "weight", weight
    agreement = 0.0
    for idx, (weight, agree) in enumerate(zip(agree_weights, agrees)):
        # print weight, "*", agree, "/", level_labels[idx]
        agreement += weight * 1.0 * agree / level_labels[idx]
    return agreement, annos


def store_data(data, p, counter):
    """
    Stores the data in a .txt file, separated by commas. Stores tweet ID,
    label agreement and how often the tweet was labeled.
    Uses 3-digit precision after comma.

    Parameters
    ----------
    data: list of tuples - each tuple represents a tweet ID, and a tuple with
    its agreement, total number of annotators who labeled it, and relative
    number of annotators who labeled it.
    p: str - path where output file should be stored.
    counter: - dict - contains per tweet ID a tuple of raw and relative counts,
    i.e. how often each tweet was labeled by annotators.

    """
    with open(p, "wb") as f:
        f.write("tweet_id,weighted_agreement,raw_count,relative_count\n")
        for tupl in data:
            tid = tupl[0]
            # (Tweet ID, (weighted agreement, annos, relative annos))
            f.write("{},{:.3f},{},{:.3f}\n".format(tid, tupl[1],
                    counter[tid][0], counter[tid][1]))


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
    res = compute_weighted_agreement(tweets)
    print res
    assert(res["5"] > res["1"] > res["6"] > res["8"] and res["2"] ==
           res["7"] and res["4"] > res["3"])
    # assert(0.66 < res["1"] < 0.67 and res["2"] == 0.5 and
    #        0.11 < res["3"] < 0.12 and 0.24 < res["4"] < 0.25 and
    #        res["t5"] == 1 and 0.64 < res["t6"] < 0.65 and res["t7"] == 0.5 and
    #        0.58 < res["t6"] < 0.59)


if __name__ == "__main__":
    # Get the absolute path to the parent directory of /scripts/
    base_dir = os.path.abspath(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), os.pardir))
    # Directory in which the datasets are stored as csv
    DS_DIR = os.path.join(base_dir, "results", "export")
    # Directory in which statistical tests will be stored
    STAT_DIR = os.path.join(base_dir, "results", "stats",
                            "rank_tweets_by_label_agreement_early_late")
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

    # Minimum number of annotators who labeled a tweet for it to be considered
    # in the analysis
    min_votes = 1
    if min_votes == 1:
        warnings.warn(
            "Internally, the script will require a minimum of 2 annotators " +
            "per tweet because otherwise label agreement can't be measured. " +
            "So in the output file only tweets with at least 2 annotators " +
            "will be listed.")
    DEBUG = False
    # Consider only <use_n> first tweets of each annotator
    use_n = 50
    # True if thresholds over S, M, L (first <use_n> tweets per annotator)
    # should be used instead of S+M (all tweets)
    use_n_threshold = True
    if DEBUG:
        test_compute_avg_agreement()
    else:
        # 1. Count label agreement only for tweets from EARLY stage
        rank_by_agreement(
            md_tweets, md_annos, su_tweets, su_annos, STAT_DIR, cleaned=True,
            min_annos=min_votes, is_early=True, use_n=use_n,
            use_n_threshold=use_n_threshold)

        # 2. Count label agreement only for tweets from LATE stage
        rank_by_agreement(
            md_tweets, md_annos, su_tweets, su_annos, STAT_DIR, cleaned=True,
            min_annos=min_votes, is_early=False, use_n=use_n,
            use_n_threshold=use_n_threshold)
