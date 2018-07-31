"""
Ranks tweets according to labeling costs (= overall annotation
time for a tweet) of annotators: if labeling costs for a tweet
are high, it's more difficult to label.

Choose for each tweet the median labeling costs (because some annotators
had a spike at a random position indicating they were distracted and to avoid
any bias of these spikes on the times, we prefer median over average).

"""
import operator
import os

import numpy as np

import pseudo_db
from anno import Annotator

# Default annotation time
ZERO = 0
EMPTY = ""
# Transition from early to late stage
THRESHOLD = 25


def rank_by_labeling_costs(
        stat_dir, md_tw_p, md_a_p, su_tw_p, su_a_p, cleaned=False, min_annos=1,
        is_early=True, use_n=50, use_n_threshold=True):
    """
    Ranks tweets by labeling costs (w.r.t. a single tweet) per group and
    institution.
    Group L is ignored.

    Parameters
    ----------
    stat_dir: str - directory in which the stats will be stored.
    md_tw_p: str - path to MD tweets dataset in csv format.
    md_a_p: str - path to MD annotators dataset in csv format.
    su_tw_p: str - path to SU tweets dataset in csv format.
    su_a_p: str - path to SU annotators dataset in csv format.
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
    inst, groups, inst_counter, group_counter = \
        read_dataset(md_tw_p, md_a_p, inst_name, thresholds, cleaned=cleaned,
                     min_annos=min_annos, is_early=is_early, use_n=use_n,
                     use_n_threshold=use_n_threshold)
    compute_labeling_costs(inst, groups, inst_counter, group_counter,
                           stat_dir, tweet_type, agg, min_annos, inst_name)

    # For SU
    inst_name = "su"
    thresholds = {
        "S": THRESHOLD,
        "M": THRESHOLD,
        "L": THRESHOLD,
        inst_name: THRESHOLD
    }
    inst, groups, inst_counter, group_counter = \
        read_dataset(su_tw_p, su_a_p, inst_name, thresholds, cleaned=cleaned,
                     min_annos=min_annos, is_early=is_early, use_n=use_n,
                     use_n_threshold=use_n_threshold)
    compute_labeling_costs(inst, groups, inst_counter, group_counter,
                           stat_dir, tweet_type, agg, min_annos, inst_name)


def compute_labeling_costs(inst, groups, inst_counter, group_counter,
                           stat_dir, tweet_type, agg, min_annos, inst_name):
    """
    Computes the (average) labeling costs for the sentiment labels of each
    tweet.

    Parameters
    ----------
    inst: ist of Annotator objects - with their tweets separated into
    early and late stage.
    Counters only contain counts for tweets that were labeled sufficiently
    often.
    groups: - dict - (group names "S" and "M" are keys) with lists of
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

    """
    inst_costs = compute_medians(inst)
    # Sort descendingly according to labeling costs and store results
    # For institution
    sorted_inst_costs = sorted(inst_costs.items(), key=operator.itemgetter(1))

    # Normalize costs
    # https://stats.stackexchange.com/questions/70801/how-to-normalize-data-to-0-1-range
    norm_costs = []
    # If there are tweets available (because they might not)
    if len(sorted_inst_costs) > 0:
        max_cost = sorted_inst_costs[-1][1]
        min_cost = sorted_inst_costs[0][1]
        for tid, cost in sorted_inst_costs:
            normalized = 1 - (cost - min_cost) / (max_cost - min_cost)
            norm_costs.append((tid, normalized))
    if use_n_threshold:
        f_name = "{}_min_annos_{}_ranked_by_labeling_costs_{}_use_" \
                 "n_{}_{}.txt" \
            .format(inst_name, min_annos, tweet_type, use_n, agg)
    else:
        f_name = "{}_min_annos_{}_ranked_by_labeling_costs_{}.txt" \
            .format(inst_name, min_annos, agg)

    store_data(norm_costs, os.path.join(stat_dir, f_name), inst_counter)

    # For groups
    for g in groups:
        if use_n_threshold:
            f_name = "{}_{}_min_annos_{}_ranked_by_labeling_costs_{}_use_" \
                     "n_{}_{}.txt" \
                .format(inst_name, g, min_annos, tweet_type, use_n, agg)
        else:
            f_name = "{}_{}_min_annos_{}_ranked_by_labeling_costs_{}.txt" \
                .format(inst_name, g, min_annos, agg)
        group_costs = compute_medians(groups[g])
        sorted_group_costs = sorted(group_costs.items(),
                                    key=operator.itemgetter(1))

        # If there are tweets available (because they might not)
        norm_costs = []
        if len(sorted_group_costs) > 0:
            # Normalize costs
            # https://stats.stackexchange.com/questions/70801/how-to-normalize-data-to-0-1-range
            max_cost = sorted_group_costs[-1][1]
            min_cost = sorted_inst_costs[0][1]

            for tid, cost in sorted_group_costs:
                normalized = 1 - (cost - min_cost) / (max_cost - min_cost)
                norm_costs.append((tid, normalized))

        store_data(norm_costs, os.path.join(stat_dir, f_name), group_counter[g])


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

    # Get annotation times per tweet
    inst = {
        # {tweet_id: [costs for anno1, costs for anno2...]}
    }
    # {tweet_id: [costs for anno1, costs for anno2...]}
    groups = {
        "S": {},
        "M": {},
        "L": {}
    }
    for anno in inst_annos:
        for idx, tid in enumerate(anno.tweets.ids):
            if tid not in inst:
                inst[tid] = []
            # Add costs to institution
            inst[tid].append(anno.tweets.anno_times[idx])
    print inst
    print len(inst)
    for group in group_annos:
        for anno in group_annos[group]:
            for idx, tid in enumerate(anno.tweets.ids):
                if tid not in groups:
                    groups[group][tid] = []
                # Add costs to group
                groups[group][tid].append(anno.tweets.anno_times[idx])

    return inst, groups, norm_inst_counts, norm_group_counts


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


def compute_medians(tweets):
    """
    Computes the median labeling costs per tweet.
    Parameters
    ----------
    tweets: dict - each key is a tweet ID and it has a list of labeling costs
    associated with it.

    Returns
    -------
    dict.
    Tweet IDs as keys and median labeling costs of a tweet as a value.
    """
    median_costs = {}
    for tweet in tweets:
        median_costs[tweet] = np.median(np.array(tweets[tweet]))
    return median_costs


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
        f.write("tweet_id,labeling_cost,raw_count,relative_count\n")
        for tupl in data:
            tid = tupl[0]
            # (Tweet ID, (weighted agreement, annos, relative annos))
            f.write("{},{:.3f},{},{:.3f}\n".format(tid, tupl[1],
                    counter[tid][0], counter[tid][1]))


def test_compute_medians():
    """Tests if compute_medians() computes median labeling costs correctly"""
    t1 = [3, 4, 1, 5]
    t2 = [1]
    t3 = [6, 3, 1]
    tweets = {"1": t1, "2": t2, "3": t3}
    res = compute_medians(tweets)
    print res
    assert(res["1"] == 3.5 and res["2"] == 1 and res["3"] == 3)


if __name__ == "__main__":
    # Get the absolute path to the parent directory of /scripts/
    base_dir = os.path.abspath(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), os.pardir))
    # Directory in which the datasets are stored as csv
    DS_DIR = os.path.join(base_dir, "results", "export")
    # Directory in which statistical tests will be stored
    STAT_DIR = os.path.join(base_dir, "results", "stats",
                            "rank_tweets_by_labeling_costs_early_late")
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
    DEBUG = False
    # Consider only <use_n> first tweets of each annotator
    use_n = 50
    # True if thresholds over S, M, L (first <use_n> tweets per annotator)
    # should be used instead of S+M (all tweets)
    use_n_threshold = True
    if DEBUG:
        test_compute_medians()
    else:
        # 1. Compute labeling costs only for tweets from EARLY stage
        # For each file
        rank_by_labeling_costs(
            STAT_DIR, md_tweets, md_annos, su_tweets, su_annos, cleaned=True,
            min_annos=min_votes, is_early=True, use_n=use_n,
            use_n_threshold=use_n_threshold)

        # 2. Compute labeling costs only for tweets from LATE stage
        # For each file
        rank_by_labeling_costs(
            STAT_DIR, md_tweets, md_annos, su_tweets, su_annos, cleaned=True,
            min_annos=min_votes, is_early=False, use_n=use_n,
            use_n_threshold=use_n_threshold)
