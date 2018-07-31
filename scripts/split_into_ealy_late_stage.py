"""
Computes the threshold for early/late stage in the annotation process.
It's computed as the median value over all annotation times from the learning
phase. That means for annos from group S the first 20 tweets are considered,
for annos from group M the first 40 tweets are considered. From those
20/40 tweets a median is chosen for each annotator, say k. From those k
annotation times the median is selected as the threshold. To distinguish
early and late stage in the annotation process.

Distinguishing between both phases allows to control for labeling costs because
we know from our InfSci paper that it affects labeling costs and labeling
reliability. Thus, by splitting our data into two intervals, we can assume that
the variable remains constant in each block and its influence may be neglected.

The script creates an output file that stores for groups S, M, L and the whole
institution those thresholds.

"""
import matplotlib.pyplot as plt
import os

import numpy as np

import pseudo_db


FONTSIZE = 12
plt.rcParams.update({'font.size': FONTSIZE})


def compute_early_late_stage_threshold(
        stat_dir, md_tw_p, md_a_p, su_tw_p, su_a_p, cleaned=False,
        with_l=True, use_n=50):
    """
    Computes the threshold between early and late stage of the annotation
    process per group (S, M, L) and for the entire institution (MD, SU). The
    resulting thresholds are stored.

    Parameters
    ----------
    stat_dir: str - directory in which the stats will be stored.
    md_tw_p: str - path to MD tweets dataset in csv format.
    md_a_p: str - path to MD annotators dataset in csv format.
    su_tw_p: str - path to SU tweets dataset in csv format.
    su_a_p: str - path to SU annotators dataset in csv format.
    cleaned: bool - True if the data should be cleaned, i.e. if tweet is
    "irrelevant", its remaining labels are ignored for computing average
    annotation times.
    with_l: True if annotators of L should also be used.
    use_n: int - considers only the first n tweets per annotator, regardless of
    annotator group.

    """
    # Find out for which institution this plot is - su or md
    dataset_type = "raw"
    if cleaned:
        dataset_type = "cleaned"
    size = "without_l"
    if with_l:
        size = "with_l"

    # For MD
    md_groups, md_n_groups = get_anno_times(md_tw_p, md_a_p, cleaned=cleaned,
                                            with_l=with_l, with_m=True,
                                            use_n=use_n)
    institution = "md"
    fname = "{}_early_late_thresholds_{}_{}.txt".format(institution, size,
                                                        dataset_type)
    dst = os.path.join(stat_dir, fname)
    compute_median_times(md_groups, md_n_groups, dst, institution)

    # For SU
    institution = "su"
    fname = "{}_early_late_thresholds_{}_{}.txt".format(institution, size,
                                                        dataset_type)
    dst = os.path.join(stat_dir, fname)
    su_groups, su_n_groups = get_anno_times(su_tw_p, su_a_p, cleaned=cleaned,
                                            with_l=with_l, with_m=True,
                                            use_n=use_n)
    compute_median_times(su_groups, su_n_groups, dst, institution)


def get_anno_times(tweet_path, anno_path, cleaned=False, with_l=False,
                   with_m=True, use_n=50):
    """
    Obtain the annotation times of the sentiment labels of the first <use_n>
    tweets per annotator. Also returns all annotation times per annotator.

    Parameters
    ----------
    tweet_path: str - path to tweet csv dataset.
    anno_path: str - path to annotator csv dataset.
    cleaned: bool - True if the data should be cleaned, i.e. if tweet is
    "irrelevant", its remaining labels are ignored for computing median
    annotation times.
    with_l: bool - True if group L should be considered in the results.
    with_m: bool - True if group M should be considered in the results.
    use_n: int - considers only the first n tweets per annotator, regardless of
    annotator group.

    Returns
    -------
    dict, dict.
    First dictionary holds all annotation times per annotator. They are stored
    per annotator group ("S", "M", "L"). The value is a list of lists where
    the outer list represents the annotators and each inner list the annotation
    times of a specific annotator.
    Second dictionary has the same structure as the first one, but only stores
    the first <use_n> annotation times per annotator. If an annotator labeled
    less than <use_n> tweets, the missing entries are filled with numpy.nan.

    """
    # Contains only the annotation times of the first n tweets
    n_groups = {
        "S": [],
        "M": [],
        "L": []
    }
    # Store annotation times of each annotator per group
    groups = {
        "S": [],
        "M": [],
        "L": []
    }
    data = pseudo_db.Data(tweet_path, anno_path)
    # List of sentiment labels assigned by annotators
    # For each annotator
    for anno in data.annotators.all_annos():
        group = anno.get_group()
        # Overall annotation times for an annotator per tweet in s
        anno_times = []
        # Annotation times of the first <use_n> tweets in s
        n_list = []
        # If we specified to exclude annotators from M or L, we skip this
        # loop. Otherwise, we extract all tweets of an annotator.
        if ((not with_l and group != "L") or with_l) \
                and ((not with_m and group != "M") or with_m):
            # For each tweet
            for i, t in enumerate(anno.get_labeled_tweets()):
                times = t.get_anno_times()
                labels = t.get_anno_labels()
                # First level
                rel_label = labels[0]
                rel_time = times[0]
                fac_time = 0
                op_time = 0
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
                        # c3 = labels[2]
                        op_time = times[2]
                # Annotation time for i-th tweet
                total_time = sum([rel_time, fac_time, op_time])
                anno_times.append(total_time)
                if i < use_n:
                    n_list.append(total_time)
            # Fill up the list with "None" values if less than <use_n> values
            # are available for an annotator
            if len(anno_times) < n:
                anno_times.extend([None] * (n - len(anno_times)))
            if len(n_list) < n:
                n_list.extend([None] * (n - len(n_list)))
            assert (len(n_list) == use_n)

        # Add L or not depending on the preferences
        if with_l or group != "L":
            # Make sure that each list contains exactly <n> entries
            # assert(len(anno_times) == n)
            groups[group].append(anno_times)
            n_groups[group].append(n_list)
    return groups, n_groups


def compute_median_times(groups, n_groups, dst, institution):
    """


    Parameters
    ----------
    groups: dict - holds all annotation times per annotator. They are stored
    per annotator group ("S", "M", "L"). The value is a list of lists where
    the outer list represents the annotators and each inner list the annotation
    times of a specific annotator. If an annotator labeled
    less than <use_n> tweets, the missing entries are filled with numpy.nan.
    n_groups: dict - same as groups, but for each annotator only annotation
    times of her first n labeled tweets are stored.
    dst: str - path where output should be stored.
    institution: str - "md" or "su" = name of the location where dataset was
    labeled.
    """
    # Store median annotation times for learning phase (first 20 tweets for
    # group S and first 40 tweets for group M) of each annotator
    group_medians = {
        "S": [],
        "M": [],
        "L": []
    }
    inst_medians = []
    thresholds = {}
    # 1) Compute medians over all tweets of S and M, i.e. use 50 tweets
    # for S and 150 for M and ignore L
    ##################################
    # Compute median per annotator in learning phase for each group and for the
    # entire institution
    for group in groups:
        # First k tweets to consider for calculating the medians of the learning
        # phase. Values for k were identified in our InfSci paper.
        k = 40
        if group == "S":
            k = 20
        print "#annotators {}: {}".format(group, len(groups[group]))
        for anno in groups[group]:
            print "labeled tweets (possibly np.nan)", len(anno)
            times = np.array(anno[:k], dtype=np.float64)
            median = np.nanmedian(times, axis=0).T
            group_medians[group].append(median)
            inst_medians.append(median)
    print inst_medians
    print len(inst_medians)
    inst_median = np.nanmedian(np.array(inst_medians, np.float64), axis=0).T
    print "median for {}: {}".format(institution, inst_median)
    thresholds[institution] = inst_median
    for group in group_medians:
        print group_medians[group]
        print len(group_medians[group])
        group_median = np.nanmedian(np.array(group_medians[group], np.float64),
                                    axis=0).T
        print "median for {}: {}".format(group, group_median)
        thresholds[group] = group_median
    store_thresholds(thresholds, dst)

    # 2) Compute medians over the first <use_n> tweets of S, M, and L, i.e.
    # use the same number of tweets for all annotators, so we assume the
    # learning phase is over after first 20 tweets (like in S)
    #################################################
    # Store median annotation times for learning phase (first 20 tweets for
    # group S and first 40 tweets for group M) of each annotator
    group_medians = {
        "S": [],
        "M": [],
        "L": []
    }
    inst_medians = []
    thresholds = {}
    # Compute median per annotator in learning phase for each group and for the
    # entire institution
    for group in n_groups:
        k = 20
        print "#annotators {}: {}".format(group, len(n_groups[group]))
        for anno in n_groups[group]:
            print "labeled tweets (possibly np.nan)", len(anno)
            times = np.array(anno[:k], dtype=np.float64)
            median = np.nanmedian(times, axis=0).T
            group_medians[group].append(median)
            inst_medians.append(median)
    # We now have lists of medians and we pick the median from a list as
    # threshold
    print inst_medians
    print len(inst_medians)
    # For institution
    inst_median = np.nanmedian(np.array(inst_medians, np.float64), axis=0).T
    print "median for {}: {}".format(institution, inst_median)
    thresholds[institution] = inst_median
    # For groups
    for group in group_medians:
        print group_medians[group]
        print len(group_medians[group])
        group_median = np.nanmedian(np.array(group_medians[group], np.float64),
                                    axis=0).T
        print "median for {}: {}".format(group, group_median)
        thresholds[group] = group_median
    store_thresholds(thresholds, dst)


def store_thresholds(thresholds, dst):
    """
    Stores the thresholds for early/late stage of the annotation process
    for groups, and institution.

    Parameters
    ----------
    thresholds: dict - contains thresholds and group/institution names as keys.
    computed.
    dst: str - destination where results will be stored.

    """
    with open(dst, "wb") as f:
        # Thresholds for S, M, L, institution
        if "L" in thresholds:
            f.write("institution,s,m,l\n")
            if "md" in thresholds:
                inst_thresh = thresholds["md"]
            else:
                inst_thresh = thresholds["su"]
            f.write("{:.3f},{:.3f},{:.3f},{:.3f}\n"
                    .format(inst_thresh, thresholds["S"], thresholds["M"],
                            thresholds["L"]))
        # Thresholds for S, M, institution
        else:
            f.write("institution,s,m\n")
            if "md" in thresholds:
                inst_thresh = thresholds["md"]
            else:
                inst_thresh = thresholds["su"]
            f.write("{:.3f},{:.3f},{:.3f}\n"
                    .format(inst_thresh, thresholds["S"], thresholds["M"]))


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
                            "early_late_stage_threshold")
    # Path to the tweets of MD
    md_tweets = os.path.join(base_dir, DS_DIR, "tweets_hierarchical_md.csv")
    # Path to the annotators of MD
    md_annos = os.path.join(base_dir, DS_DIR, "annotators_hierarchical_md.csv")
    # Path to the tweets of SU
    su_tweets = os.path.join(base_dir, DS_DIR, "tweets_hierarchical_su.csv")
    # Path to the annotators of SU
    su_annos = os.path.join(base_dir, DS_DIR, "annotators_hierarchical_su.csv")

    # a) with L
    ##############
    if not os.path.exists(STAT_DIR):
        os.makedirs(STAT_DIR)
    # Raw
    compute_early_late_stage_threshold(
        STAT_DIR, md_tweets, md_annos, su_tweets, su_annos, with_l=True,
        use_n=n, cleaned=False)
    # Cleaned
    compute_early_late_stage_threshold(
        STAT_DIR, md_tweets, md_annos, su_tweets, su_annos, with_l=True,
        use_n=n, cleaned=True)

    # b) without L
    ##############
    if not os.path.exists(STAT_DIR):
        os.makedirs(STAT_DIR)
    # Raw
    compute_early_late_stage_threshold(
        STAT_DIR, md_tweets, md_annos, su_tweets, su_annos, with_l=False,
        use_n=n, cleaned=False)
    # Cleaned
    compute_early_late_stage_threshold(
        STAT_DIR, md_tweets, md_annos, su_tweets, su_annos, with_l=False,
        use_n=n, cleaned=True)
