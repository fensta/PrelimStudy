"""
Analyze how the labels are distributed in easy/difficult tweets (per stratum
and in total), i.e. we examine the following strata:
1. easy tweets labeled in early stage
2. difficult tweets labeled in early stage
3. easy tweets labeled in late stage
4. difficult tweets labeled in late stage

In addition, we analyze easy tweets vs. difficult tweets, i.e. strata 1+3 vs.
2+4.
In our analysis, we focus on which labels were assigned. If certain labels are
more frequent in 1+3 vs. 2+4, it suggests that this decision was easier to make
for humans. We could use that as a recommendation for future labeling tasks.
"""
import matplotlib.pyplot as plt
import os
from collections import Counter

import numpy as np

import utility
from anno import Annotator

EMPTY = ""
ZERO = 0

# For Information sciences
FONTSIZE = 12
plt.rcParams.update({'font.size': FONTSIZE})


def plot_label_distribution(
        dbs, db_idxs, inst_name, fig_dir, threshold_dir, label_dir, i,
        anno_coll_name="user", tweet_coll_name="tweets",
                   cleaned=False, min_annos=1, is_early=True,
                   is_easy=True, both_stages=False, train_ratio=0.66):
    """

    Parameters
    ----------
    dbs: list of strings - names of the existing DBs
    db_idxs: list of ints - name of the MongoDB from where data should be read.
    inst_name: str - name of the institution for which i is computed.
    fig_dir: str - directory in which plots are stored.
    threshold_dir: str - directory containing thresholds for early/late
    annotation stage.
    label_dir: str - directory containing tweet difficulty labels.
    i: int - number of tweets to be used for training.
    anno_coll_name: str - name of the collection holding the annotator data.
    tweet_coll_name: str - name of the collection holding the tweet data.
    cleaned: bool - True if the data should be cleaned, i.e. if tweet is
    "irrelevant", its remaining labels are ignored for computing average
    annotation times.
    min_annos: int - minimum number of annotators who must've assigned a label
    to a tweet. Otherwise it'll be discarded.
    is_early: bool - True if only tweets from the early phase should be
    considered. Else only tweets from the late stage are considered.
    is_easy: bool - True if tweet difficulty is easy. Else it's difficult.
    both_stages: bool - True if both stages should be used instead of just early
    or late stage. If True, <is_early> is ignored. If False, only one stage
    according to <is_early> is considered.
    train_ratio: float - between 0-1, specifies how many percent of the
    tweets should be used for training/testing.

    """
    print "Institution: {}".format(inst_name)
    print "----------------"
    stage = "late"
    if is_early:
        stage = "early"
    if both_stages:
        stage = "both"
    difficulty = "difficult"
    if is_easy:
        difficulty = "easy"
    agg = "raw"
    if cleaned:
        agg = "cleaned"
    fname = "{}_early_late_thresholds_without_l_{}.txt".format(inst_name, agg)
    dst = os.path.join(threshold_dir, fname)
    try:
        thresholds = read_thresholds(dst, inst_name)
    except IOError:
        raise IOError("First run [deprecated]split_into_early_late_stage.py with proper " +
                      "parameters! Otherwise this script won't work!")

    print "Now process stratum '{} tweets from {} stage'".format(difficulty,
                                                                 stage)
    ################
    # 1. Load data #
    ################
    inst_annos, _ = read_dataset(
        dbs, db_idxs, inst_name, thresholds, anno_coll_name=anno_coll_name,
        tweet_coll_name=tweet_coll_name, cleaned=cleaned, min_annos=min_annos,
        is_early=is_early, both_stages=both_stages)

    ###################
    # For institution #
    ###################
    # a) Get annotators who labeled at least i easy/difficult tweets in stage
    # early/late
    fname = "{}_min_annos_{}_difficulty_labels_early_train_ratio_{}_{}.txt" \
        .format(inst_name, min_annos, train_ratio, agg)
    label_dst = os.path.join(label_dir, fname)
    _, filtered_inst_annos = \
        count_annos_with_at_least_i_tweets(label_dst, inst_annos, i,
                                           is_easy=is_easy)
    print "available annotators in this stratum:", len(filtered_inst_annos)

    # b) Count the labels assigned by annotators
    labels = []
    for anno in filtered_inst_annos:
        for tweet_labels in anno.tweets.labels:
            labels.extend(tweet_labels)
    x_labels = ["Relevant", "Irrelevant", "Factual", "Non-factual", "Positive",
                "Negative"]
    # Ignore empty labels
    counted = Counter([label for label in labels if label in x_labels])

    fname = "{}_{}_tweets_{}_stage_label_distribution_train_ratio_{}_min_" \
            "annos_{}_{}.pdf"\
        .format(inst_name, difficulty, stage, train_ratio, min_annos, agg)
    dst = os.path.join(fig_dir, fname)
    _plot_distribution(counted, x_labels, "", dst)


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
        tid, label, _, _, _ = line.split(",")
        tweets[tid] = label
    return tweets


def _plot_distribution(counter, x_labels, title, fpath):
    """
    Plots a bar chart given some counted labels.

    Parameters
    ----------
    counter: collection.Counter - a counter holding class label.
    xlabels: List of str - list of labels for x-axis.
    title: str - title of the plot.
    fpath: str - path where the plot is stored.

    """
    # Number of labels
    num_items = len(x_labels)
    # Color for each annotator group S, M, L
    COLORS = ["dodgerblue", "darkorange", "green"]
    # Label names in legend
    legend_labels = []
    y = []
    # Bar graphs expect a total width of "1.0" per group
    # Thus, you should make the sum of the two margins
    # plus the sum of the width for each entry equal 1.0.
    # One way of doing that is shown below. You can make
    # The margins smaller if they're still too big.
    # See
    # http://stackoverflow.com/questions/11597785/setting-spacing-between-grouped-bar-plots-in-matplotlib
    # to calculate gap in between bars; if gap isn't large enough, increase
    # <margin
    margin = 0.1
    # width = (1.-0.5*margin) / num_items
    width = (1.-0.5*margin) / (num_items-5)
    ind = np.arange(num_items)
    y_tmp = []
    # Compute total number of labels because we want to display %
    # on y-axis, but consider only labels that are displayed in this
    # plot
    total = sum(counter[k] for k in counter if k in
                x_labels)
    # Get counts of the institution for the desired labels and normalize
    # counts (i.e. convert them to percentage)

    for label in x_labels:
        # if label in ["Relevant", "Irrelevant"]:
        #     total = counter["Relevant"] + counter["Irrelevant"]
        # if label in ["Factual", "Non-factual"]:
        #     total = counter["Factual"] + counter["Non-factual"]
        # if label in ["Positive", "Negative"]:
        #     total = counter["Positive"] + counter["Negative"]
        print "TOTAL:", total
        prcnt = 1.0 * counter[label] / total * 100
        y_tmp.append(prcnt)
    y.append(y_tmp)

    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_subplot(111)

    # Plot bars by displaying each annotator group after the other
    # Initially, each class adds 0 to its y-value as we start from the bottom
    # We have in total <Institution> graphs per label and each row represents
    # the percentages of the institutions for a different label
    bottom = np.zeros(shape=num_items,)
    # See
    # http://stackoverflow.com/questions/11597785/setting-spacing-between-grouped-bar-plots-in-matplotlib
    # to calculate gap in between bars; if gap isn't large enough, increase
    # <margin>
    for idx, y_ in enumerate(y):
        x = ind + margin + (idx * width)
        # If it's a stack, i.e. not the one at the bottom, always add the bottom
        ax.bar(x, y_, width, color=COLORS[idx])
        # Add current y-values to the bottom, so that next bar starts directly
        # above
        bottom += y_
    # plt.title(title)
    # Set title position
    ttl = ax.title
    ttl.set_position([.5, 1.05])
    # Set labels for ticks
    ax.set_xticks(ind)
    # Rotate x-axis labels
    ax.set_xticklabels(x_labels, rotation=45)
    # y-axis from 0-100
    # plt.yticks(np.arange(0, 110, 10))
    # Hide the right and top spines (lines)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    # Set labels of axes
    # ax.set_xlabel("Labels")
    ax.set_ylabel("Percentage")
    # Add a legend
    # ax.legend(loc="center left", shadow=True, bbox_to_anchor=(1, 0.5))
    # ax.legend(loc="best", shadow=True, fontsize=FONTSIZE)
    plt.savefig(fpath, bbox_inches="tight", dpi=600)
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
        dbs, db_idxs, inst_name, thresholds, anno_coll_name="user",
        tweet_coll_name="tweets", cleaned=False, min_annos=1,
        is_early=True, both_stages=False):
    """
    Read dataset (tweet texts per annotator and their labels) per group
    and institution.
    Group L is ignored.

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
    both_stages: bool - True if both stages should be used instead of just early
    or late stage. If True, <is_early> is ignored. If False, only one stage
    according to <is_early> is considered.

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
                for idx, tid in enumerate(labeled):
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
                    # Use tweets from both stages
                    if both_stages:
                        group_anno.add_tweet(tweet_id, anno_time,
                                             labels, text)
                        inst_anno.add_tweet(tweet_id, anno_time, labels, text)
                    else:
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
        "M": count_annotators_per_tweet(group_annos["M"])
    }

    # Now only keep tweets that were labeled sufficiently often by annotators
    # Create a list of tweet IDs that must be removed since they weren't labeled
    # by enough annotators
    removed_inst_tweets = [tid for tid in inst_counts if
                           inst_counts[tid] < min_annos]

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

    """
    thresholds = {}
    with open(dst, "rb") as f:
        lines = f.readlines()
    # Skip header
    for line in lines[1:]:
        inst, s, m = line.split(",")
        inst = float(inst)
        s = float(s)
        m = float(m)
        thresholds["S"] = s
        thresholds["M"] = m
        thresholds[inst_name] = inst
    return thresholds


if __name__ == "__main__":
    # Name of the collection in each DB holding annotator data
    ANNO_COLL_NAME = "user"
    # Name of the collection in each DB holding tweet data
    TWEET_COLL_NAME = "tweets"
    # Directory in which stats will be stored
    STAT_DIR = "/media/data/Workspaces/PythonWorkspace/phd/" \
               "Analyze-Labeled-Dataset/sac2018_results/stats/label_distribution/"
    FIG_DIR = "/media/data/Workspaces/PythonWorkspace/phd/" \
              "Analyze-Labeled-Dataset/sac2018_results/figures/label_distribution/"
    THRESHOLD_DIR = "/media/data/Workspaces/PythonWorkspace/phd/Analyze-Labeled-Dataset/sac2018_results/stats/early_late_stage_threshold/"
    tweet_label_dir = "/media/data/Workspaces/PythonWorkspace/phd/" \
                      "Analyze-Labeled-Dataset/sac2018_results/stats/difficulty_labels_early_late/"
    if not os.path.exists(STAT_DIR):
        os.makedirs(STAT_DIR)
    if not os.path.exists(FIG_DIR):
        os.makedirs(FIG_DIR)

    DB_NAMES = [
        "lturannotationtool",
        "mdlannotationtool",
        "mdsmannotationtool",
        "turannotationtool",
        "harishannotationtool",
        "kantharajuannotationtool",
        "mdslaterannotationtool",
    ]

    # Different DBs for MD, LATER, and SU
    SU_ALL = [0, 3]
    MD_ALL = [1, 2, 4, 5, 6]
    ALL = [0, 1, 2, 3, 4, 5, 6]
    # Previous analysis suggested 15 tweets to be used for the training set size
    i = 10
    train_ratio = 0.4

    # Minimum annotators per tweet (it assumes that the difficulty scores
    # were calculated for this in advance - see readme.txt
    min_annos = 1
    #########
    # 1. MD #
    #########
    # Stratum: easy tweets, early stage
    plot_label_distribution(DB_NAMES, MD_ALL, "md", FIG_DIR, THRESHOLD_DIR,
                   tweet_label_dir, i, ANNO_COLL_NAME, TWEET_COLL_NAME,
                   cleaned=True, min_annos=min_annos,
                   is_early=True, is_easy=True, train_ratio=train_ratio)
    # Stratum: difficult tweets, early stage
    plot_label_distribution(DB_NAMES, MD_ALL, "md", FIG_DIR, THRESHOLD_DIR,
                   tweet_label_dir, i, ANNO_COLL_NAME, TWEET_COLL_NAME,
                   cleaned=True, min_annos=min_annos,
                   is_early=True, is_easy=False, train_ratio=train_ratio)
    # Stratum: easy tweets, late stage
    plot_label_distribution(DB_NAMES, MD_ALL, "md", FIG_DIR, THRESHOLD_DIR,
                   tweet_label_dir, i, ANNO_COLL_NAME, TWEET_COLL_NAME,
                   cleaned=True, min_annos=min_annos,
                   is_early=False, is_easy=True, train_ratio=train_ratio)
    # Stratum: difficult tweets, late stage
    plot_label_distribution(DB_NAMES, MD_ALL, "md", FIG_DIR, THRESHOLD_DIR,
                   tweet_label_dir, i, ANNO_COLL_NAME, TWEET_COLL_NAME,
                   cleaned=True, min_annos=min_annos,
                   is_early=False, is_easy=False, train_ratio=train_ratio)

    #########
    # 2. SU #
    #########
    # Stratum: easy tweets, early stage
    plot_label_distribution(DB_NAMES, SU_ALL, "su", FIG_DIR, THRESHOLD_DIR,
                   tweet_label_dir, i, ANNO_COLL_NAME, TWEET_COLL_NAME,
                   cleaned=True, min_annos=min_annos,
                   is_early=True, is_easy=True, train_ratio=train_ratio)
    # Stratum: difficult tweets, early stage
    plot_label_distribution(DB_NAMES, SU_ALL, "su", FIG_DIR, THRESHOLD_DIR,
                   tweet_label_dir, i, ANNO_COLL_NAME, TWEET_COLL_NAME,
                   cleaned=True, min_annos=min_annos,
                   is_early=True, is_easy=False, train_ratio=train_ratio)
    # Stratum: easy tweets, late stage
    plot_label_distribution(DB_NAMES, SU_ALL, "su", FIG_DIR, THRESHOLD_DIR,
                   tweet_label_dir, i, ANNO_COLL_NAME, TWEET_COLL_NAME,
                   cleaned=True, min_annos=min_annos,
                   is_early=False, is_easy=True, train_ratio=train_ratio)
    # Stratum: difficult tweets, late stage
    plot_label_distribution(DB_NAMES, SU_ALL, "su", FIG_DIR, THRESHOLD_DIR,
                   tweet_label_dir, i, ANNO_COLL_NAME, TWEET_COLL_NAME,
                   cleaned=True, min_annos=min_annos,
                   is_early=False, is_easy=False, train_ratio=train_ratio)

    ########################################################
    # 3. MD, stages together, i.e. only easy vs. difficult #
    ########################################################
    # Stratum: easy tweets
    plot_label_distribution(DB_NAMES, MD_ALL, "md", FIG_DIR, THRESHOLD_DIR,
                            tweet_label_dir, i, ANNO_COLL_NAME, TWEET_COLL_NAME,
                            cleaned=True, min_annos=min_annos,
                            is_easy=True, both_stages=True,
                            train_ratio=train_ratio)
    # Stratum: difficult tweets
    plot_label_distribution(DB_NAMES, MD_ALL, "md", FIG_DIR, THRESHOLD_DIR,
                            tweet_label_dir, i, ANNO_COLL_NAME, TWEET_COLL_NAME,
                            cleaned=True, min_annos=min_annos,
                            is_easy=False, both_stages=True,
                            train_ratio=train_ratio)

    ########################################################
    # 4. SU, stages together, i.e. only easy vs. difficult #
    ########################################################
    # Stratum: easy tweets
    plot_label_distribution(DB_NAMES, SU_ALL, "su", FIG_DIR, THRESHOLD_DIR,
                            tweet_label_dir, i, ANNO_COLL_NAME, TWEET_COLL_NAME,
                            cleaned=True, min_annos=min_annos,
                            is_easy=True, both_stages=True,
                            train_ratio=train_ratio)
    # Stratum: difficult tweets
    plot_label_distribution(DB_NAMES, SU_ALL, "su", FIG_DIR, THRESHOLD_DIR,
                            tweet_label_dir, i, ANNO_COLL_NAME, TWEET_COLL_NAME,
                            cleaned=True, min_annos=min_annos,
                            is_easy=False, both_stages=True,
                            train_ratio=train_ratio)
