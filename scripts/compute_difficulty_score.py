"""
Computes for the difficulty score of a single tweet given the following
information:
- label agreement [0-1], 1 = perfect agreement
- classifier certainty [0-1], 1 = 100% certainty
- number of labels [0-1], 1= most labeled tweet

A higher score indicates easier tweets.

This script also plots the resulting difficulty scores for visual inspection as
well as it prints out how these scores are distributed (normally?) which is
important for later analyses.

"""
import os
import warnings
import matplotlib.pyplot as plt
import operator
import math

import numpy as np

FONTSIZE = 12
plt.rcParams.update({'font.size': FONTSIZE})


def main(certainty_dir, agree_dir, cost_dir, inst_name, fig_dir, stat_dir,
         s=True, m=True,
         min_annos=1, cleaned=True, is_early=True, train_ratio=0.66, use_n=50,
         use_n_threshold=True):
    """
    Analyze the data.

    Parameters
    ----------
    certainty_dir: str - directory in which results for classifier certainty
    are stored.
    agree_dir: str - directory in which results for label agreement
    are stored.
    cost_dir: str - directory in which results for labeling costs are stored.
    inst_name: str - name of the institution for which data should be loaded.
    fig_dir: str - directory in which the plot will be stored.
    stat_dir: str - directory in which the stats will be stored.
    s: bool - True if group S of <inst> should be considered separately.
    m: bool - True if group M of <inst> should be considered separately.
    min_annos: int - minimum number of annotators who must've assigned a label
    to a tweet. Otherwise it'll be discarded.
    cleaned: bool - True if cleaned dataset should be used. Else the raw one
    is used.
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
    agreement_name = "agreement"
    certainty_name = "classifier_certainty"
    cost_name = "labeling_costs"
    ds_name = "cleaned"
    if not cleaned:
        ds_name = "raw"
    tweet_type = "late"
    if is_early:
        tweet_type = "early"
    # Figure out which files to open
    data = ["S", "M"]
    if s and not m:
        data = ["S"]
    if not s and m:
        data = ["M"]
    if use_n_threshold:
        data = ["S", "M", "L"]
    # Determine file names of input and output files
    if use_n_threshold:
        certain_fname = "{}_min_annos_{}_ranked_by_{}_{}_train_ratio_{}_" \
                        "use_n_{}_{}.txt" \
            .format(inst_name, min_annos, certainty_name, tweet_type,
                    train_ratio, use_n, ds_name)
        agree_fname = "{}_min_annos_{}_ranked_by_{}_{}_use_n_{}_{}.txt" \
            .format(inst_name, min_annos, agreement_name, tweet_type, use_n,
                    ds_name)
        cost_fname = "{}_min_annos_{}_ranked_by_{}_{}_use_n_{}_{}.txt" \
            .format(inst_name, min_annos, cost_name, tweet_type, use_n,
                    ds_name)
        fp_fname = "{}_min_annos_{}_difficulty_score_{}_train_ratio_{}_" \
                   "use_n_{}_{}.txt" \
            .format(inst_name, min_annos, tweet_type, train_ratio, use_n,
                    ds_name)
        v_fname = "{}_min_annos_{}_difficulty_score_{}_train_ratio_{}_use_n_{}" \
                  "_{}.pdf" \
            .format(inst_name, min_annos, tweet_type, train_ratio, use_n,
                    ds_name)
        l_fname = "{}_min_annos_{}_difficulty_score_distribution_normal_" \
                  "scores_{}_train_ratio_{}_use_n_{}_{}.pdf" \
            .format(inst_name, min_annos, tweet_type, train_ratio, use_n,
                    ds_name)
        ll_fname = "{}_min_annos_{}_difficulty_score_distribution_log_normal" \
                   "_scores_{}_train_ratio_{}_use_n_{}_{}.pdf" \
            .format(inst_name, min_annos, tweet_type, train_ratio, use_n,
                    ds_name)
    else:
        certain_fname = "{}_min_annos_{}_ranked_by_{}_{}_train_ratio_{}_{}" \
                        ".txt" \
            .format(inst_name, min_annos, certainty_name, tweet_type,
                    train_ratio, ds_name)
        agree_fname = "{}_min_annos_{}_ranked_by_{}_{}_{}.txt" \
            .format(inst_name, min_annos, agreement_name, tweet_type, ds_name)
        cost_fname = "{}_min_annos_{}_ranked_by_{}_{}_{}.txt" \
            .format(inst_name, min_annos, cost_name, tweet_type, ds_name)
        fp_fname = "{}_min_annos_{}_difficulty_score_{}_train_ratio_{}_{}.txt" \
            .format(inst_name, min_annos, tweet_type, train_ratio, ds_name)
        v_fname = "{}_min_annos_{}_difficulty_score_{}_train_ratio_{}_{}.pdf" \
            .format(inst_name, min_annos, tweet_type, train_ratio, ds_name)
        l_fname = "{}_min_annos_{}_difficulty_score_distribution_normal_" \
                  "scores_{}_train_ratio_{}_{}.pdf" \
            .format(inst_name, min_annos, tweet_type, train_ratio, ds_name)
        ll_fname = "{}_min_annos_{}_difficulty_score_distribution_log_normal" \
                   "_scores_{}_train_ratio_{}_{}.pdf" \
            .format(inst_name, min_annos, tweet_type, train_ratio, ds_name)
    certainty_path_inst = os.path.join(certainty_dir, certain_fname)
    agreement_path_inst = os.path.join(agree_dir, agree_fname)
    cost_path_inst = os.path.join(cost_dir, cost_fname)
    # Use counts from agreement because as soon as 1 annotator labeled the tweet
    # we can compute label agreement, but we need at least 2 annotators
    certainty_tids, inst_certainties, _ = \
        read_file(certainty_path_inst)
    fp_inst = os.path.join(stat_dir, fp_fname)
    # agreement_tids, inst_agreements = read_agreement(agreement_path_inst)

    # Create a subdirectory in figures for distributions
    f_dir = os.path.join(fig_dir, "distributions")
    if not os.path.exists(f_dir):
        os.makedirs(f_dir)

    # Be able to quickly access the values of a tweet at a certain position
    # {tweet_id: position in list}
    inst_certain_lookup = {tid: idx for idx, tid in enumerate(certainty_tids)}
    # Compute difficulty scores also for the entire institution
    scores = compute_difficulty_scores(
        certainty_path_inst, agreement_path_inst, cost_path_inst, fp_inst,
        min_annos, inst_certain_lookup, None)
    # Visualize tweet difficulty scores to see how well they're distributed
    dst = os.path.join(fig_dir, v_fname)
    plot_scores(scores, dst)
    # Visualize the tweet difficulty score distribution using original scores
    # and log-normal transformed scores to see which one approximates a
    # normal distribution better (since we want to use t-test later on)
    dst = os.path.join(f_dir, l_fname)
    plot_difficulty_score_distribution(scores.values(), dst)
    dst = os.path.join(f_dir, ll_fname)
    # Log-normal transform difficulty scores
    vals = [(math.log(scr), agr, crt, cst, cnt) for scr, agr, crt, cst, cnt in
            scores.values()]
    plot_difficulty_score_distribution(vals, dst)

    # For each group
    for group in data:
        print "open the following files:", group
        # Determine input file and output file names
        if use_n_threshold:
            c_fname = "{}_{}_min_annos_{}_ranked_by_{}_{}_train_ratio_{}_" \
                      "use_n_{}_{}.txt" \
                .format(inst_name, group, min_annos, certainty_name, tweet_type,
                        train_ratio, use_n, ds_name)
            a_fname = "{}_{}_min_annos_{}_ranked_by_{}_{}_use_n_{}_{}.txt" \
                .format(inst_name, group, min_annos, agreement_name, tweet_type,
                        use_n, ds_name)
            cc_fname = "{}_{}_min_annos_{}_ranked_by_{}_{}_use_n_{}_{}.txt" \
                .format(inst_name, group, min_annos, cost_name, tweet_type,
                        use_n, ds_name)
            fp_fname = "{}_{}_min_annos_{}_difficulty_score_{}_train_ratio_{}" \
                       "_use_n_{}_{}.txt" \
                .format(inst_name, group, min_annos, tweet_type, train_ratio,
                        use_n, ds_name)
            v_fname = "{}_{}_min_annos_{}_difficulty_score_{}_train_ratio_{}_" \
                      "use_n_{}_{}.pdf" \
                .format(inst_name, group, min_annos, tweet_type, train_ratio,
                        use_n, ds_name)
            l_fname = "{}_{}_min_annos_{}_difficulty_score_distribution_log_" \
                      "normal_scores_{}_train_ratio_{}_use_n_{}_{}.pdf" \
                .format(inst_name, group, min_annos, tweet_type, train_ratio,
                        use_n, ds_name)
            n_fname = "{}_{}_min_annos_{}_difficulty_score_distribution_" \
                      "normal_scores_{}_train_ratio_{}_use_n_{}_{}.pdf" \
                .format(inst_name, group, min_annos, tweet_type, train_ratio,
                        use_n, ds_name)
        else:
            c_fname = "{}_{}_min_annos_{}_ranked_by_{}_{}_train_ratio_{}_" \
                      "{}.txt" \
                .format(inst_name, group, min_annos, certainty_name, tweet_type,
                        train_ratio, ds_name)
            a_fname = "{}_{}_min_annos_{}_ranked_by_{}_{}_{}.txt" \
                .format(inst_name, group, min_annos, agreement_name, tweet_type,
                        ds_name)
            cc_fname = "{}_{}_min_annos_{}_ranked_by_{}_{}_{}.txt" \
                .format(inst_name, group, min_annos, cost_name, tweet_type,
                        ds_name)
            fp_fname = "{}_{}_min_annos_{}_difficulty_score_{}_train_ratio_{}" \
                       "_{}.txt" \
                .format(inst_name, group, min_annos, tweet_type, train_ratio,
                        ds_name)
            v_fname = "{}_{}_min_annos_{}_difficulty_score_{}_train_ratio_{}_" \
                      "{}.pdf" \
                .format(inst_name, group, min_annos, tweet_type, train_ratio,
                        ds_name)
            l_fname = "{}_{}_min_annos_{}_difficulty_score_distribution_log_" \
                      "normal_scores_{}_train_ratio_{}_{}.pdf" \
                .format(inst_name, group, min_annos, tweet_type, train_ratio,
                        ds_name)
            n_fname = "{}_{}_min_annos_{}_difficulty_score_distribution_" \
                      "normal_scores_{}_train_ratio_{}_{}.pdf" \
                .format(inst_name, group, min_annos, tweet_type, train_ratio,
                        ds_name)
        certainty_path = os.path.join(certainty_dir, c_fname)
        agreement_path = os.path.join(agree_dir, a_fname)
        cost_path = os.path.join(cost_dir, cc_fname)
        fp = os.path.join(stat_dir, fp_fname)
        scores = compute_difficulty_scores(
            certainty_path, agreement_path, cost_path, fp, min_annos,
            inst_certain_lookup, inst_certainties)
        dst = os.path.join(fig_dir, v_fname)
        plot_scores(scores, dst)
        # Visualize the tweet difficulty score distribution using original
        # scores and log-normal transformed scores to see which one
        # approximates a normal distribution better (since we want to use
        # t-test later on)
        dst = os.path.join(f_dir, n_fname)
        plot_difficulty_score_distribution(scores.values(), dst)
        dst = os.path.join(f_dir, l_fname)
        # Log-normal transform difficulty scores
        vals = [(math.log(scr), agr, crt, cst, cnt) for scr, agr, crt, cst, cnt
                in scores.values()]
        plot_difficulty_score_distribution(vals, dst)


def compute_difficulty_scores(certain_path, agree_path, cost_path, dst,
                              min_annos, inst_certain_lookup, inst_certainties):
    """
    Computes the difficulty scores of an institution or group of an institution.
    Stores the resulting scores and visualizes them.

    Parameters
    ----------
    certain_path: str - path to file that holds classifier certainties.
    agree_path: str - path to file that holds label agreement.
    cost_path: str - path to file that holds the labeling costs.
    dst: str - path where output file is stored.
    min_annos: int - minimum number of annotators that labeled each tweet.
    inst_certain_lookup: dict - stores for each tweet ID the index at which the
    classifier certainty for the institution is stored in <inst_certainties>.
    inst_certainties: list of float - tweet IDs as keys and classifier
    certainties as values. For institutions this isn't used.

    Returns
    -------
    dict.
    {tweet ID: (difficulty, label agreement, classifier certainty,
    labeling cost, relative counts), ...}

    """
    # 1. Read in files
    # -----------------
    certain_tids, certainties, _ = read_file(certain_path)
    agree_tids, agreements, counts = read_file(agree_path)
    cost_tids, costs, _ = read_file(cost_path)

    print "AGREE", len(agree_tids), len(agreements), len(counts)
    print "Certain", len(certain_tids), len(certainties), len(counts)
    print "cost", len(cost_tids), len(costs), len(counts)

    # group_agree_lookup = {}
    # Create quick lookup so that the value for a given tweet ID can be
    # retrieved easily
    certain_lookup = {tid: idx for idx, tid in enumerate(certain_tids)}
    cost_lookup = {tid: idx for idx, tid in enumerate(cost_tids)}
    # group_agree_lookup = {tid: idx for idx, tid in enumerate(agree_tids)}

    # Compute difficulty scores - iterate over label agreement as it's
    # computed for every labeled tweet (while classifier certainty might
    # not exist for tweets of group S if too few tweets were available for
    # building a classifier). In that case choose the median certainty.
    median_certainty = np.median(certainties)
    print certainties
    print "median", median_certainty
    median_cost = np.median(costs)
    print "cost median", median_cost
    # Difficulty scores: {tweet ID: difficulty score,...}
    scores = {}
    # 2. Extract for each tweet classifier certainty, label agreement, votes
    # -----------------------------------------------------------------------
    for agree_idx, tid in enumerate(agree_tids):
        if tid in cost_lookup:
            cost_idx = cost_lookup[tid]
            cost = costs[cost_idx]
        else:
            cost = median_cost
        # Tweet was labeled sufficiently often
        if tid in certain_lookup:
            certain_idx = certain_lookup[tid]
            certainty = certainties[certain_idx]
        # Tweet wasn't labeled sufficiently often, so use the average of value
        # computed over the whole institution and the median annotation time
        # in this particular group instead
        elif tid in inst_certain_lookup:
            certain_idx = inst_certain_lookup[tid]
            certainty = (1.0*inst_certainties[certain_idx]+median_certainty) / 2
        # Tweet wasn't labeled sufficiently often at all
        else:
            certainty = median_certainty
            warnings.warn("No classifier was trained for {} due to "
                          "insufficient labels (={})!".format(tid, min_annos))
        agreement = agreements[agree_idx]
        count = counts[agree_idx]
        # 3. Compute tweet difficulty
        # ---------------------------
        difficulty = difficulty_score(certainty, agreement, cost, count)
        # print "inputs", certainty, agreement, count
        # print "Difficulty {}: {}".format(tid, difficulty)
        scores[tid] = (difficulty, agreement, certainty, cost, count)
    # 4. Store tweet difficulties descendingly
    # ----------------------------------------
    sorted_scores = \
        sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
    write_difficulty_scores(dst, sorted_scores)
    return scores


def plot_scores(scores, dst):
    """
    Creates a scatter plot for the tweet difficulty scores where the x-axis
    represents tweet IDs and the y-axis the resulting tweet difficulties.
    The tweet difficulties are sorted ascendingly.

    Parameters
    ----------
    scores: dict - the tweet difficulty scores where keys are tweet IDs, and
    values are the respective scores.
    dst: str - path where plot will be stored.

    """
    # In some groups there might be no annos and tweets at all, hence no scores
    if len(scores) > 0:
        fig = plt.figure(figsize=(5, 3))
        ax = fig.add_subplot(111)
        x = range(len(scores))
        y = sorted(scores.values(), key=operator.itemgetter(0), reverse=True)
        # First unzip the lists into tuples, then convert the tuples into lists
        difficulty, agreement, certainty, cost, count = map(list, zip(*y))
        ax.scatter(x, difficulty, color="black", label="Tweet difficulty")
        ax.scatter(x, agreement, color="blue", label="Label agreement")
        ax.scatter(x, certainty, color="red", label="Classifier certainty")
        ax.scatter(x, cost, color="orange", label="Classifier certainty")
        ax.scatter(x, count, color="green", label="Counts")
        # Title
        # Hide the right and top spines (lines)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        # Limits of axes
        plt.xlim(0, x[-1] + 1)
        plt.ylim(0, y[0][0] + 0.5)
        # Set labels of axes
        ax.set_xlabel("Tweet ID")
        ax.set_ylabel("Tweet difficulty")
        # Add legend outside of plot
        ax.legend(loc="upper right", shadow=True, bbox_to_anchor=(0.8,  1.4),
                  fontsize=FONTSIZE)
        # plt.tick_params(
        #     axis='x',          # changes apply to the x-axis
        #     which='both',      # both major and minor ticks are affected
        #     bottom='off',      # ticks along the bottom edge are off
        #     top='off',         # ticks along the top edge are off
        #     labelbottom='off') # labels along the bottom edge are off
        plt.savefig(dst, bbox_inches='tight', dpi=600)
        plt.close()


def read_agreement(fp):
    """
    Reads in a file that stores ranked agreement of tweets.

    Parameters
    ----------
    fp: str - path to file.

    Returns
    -------
    list, list.
    Tweet IDs, label agreement value.

    """
    ids = []
    metrics = []
    with open(fp, "rb") as f:
        lines = f.readlines()
    # Skip header
    for line in lines[1:]:
        tid, metr, _, _ = line.split(",")
        metr = float(metr)
        ids.append(tid)
        metrics.append(metr)
    return ids, metrics


def read_file(fp):
    """
    Reads in a file that stores ranked tweets w.r.t. a certain metric
    (label agreement or classifier certainty) and how
    often each tweet was labeled.

    Parameters
    ----------
    fp: str - path to file.

    Returns
    -------
    list, list, list.
    Tweet IDs, label agreement value, relative counts (= #annos who labeled
    current tweet/#max number of annos who labeled any tweet in the group).

    """
    ids = []
    certainties = []
    counts = []
    with open(fp, "rb") as f:
        lines = f.readlines()
    # Skip header
    for line in lines[1:]:
        tid, metr, _, count = line.split(",")
        metr = float(metr)
        count = float(count)
        ids.append(tid)
        certainties.append(metr)
        counts.append(count)
    return ids, certainties, counts


def difficulty_score(agreement, certainty, cost, votes):
    """
    Computes the difficulty score for a given tweet.

    Parameters
    ----------
    agreement: float - average agreement over all hierarchy levels.
    certainty: float - average classifier certainty over all hierarchy levels.
    cost: float - average labeling cost over all hierarchy levels.
    votes: float - relative number of annotators who labeled the tweet
    (compared to the tweet with most labels).

    Returns
    -------
    float.
    Difficulty score of a  tweet.

    """
    return agreement + certainty + +cost  # + votes


def write_difficulty_scores(fp, data):
    """
    Stores the data in a .txt file, separated by commas. Stores tweet ID,
    and difficulty score.
    Uses 3-digit precision after comma.

    Parameters
    ----------
    data: list of tuples - [(tweet ID, (difficulty score, agreement, classifier
    certainty, relative counts)),...].
    fp: str - path where output file should be stored.

    """
    with open(fp, "wb") as f:
        f.write("tweet id,difficulty score,label agreement,classifier "
                "certainty,labeling_cost,relative counts\n")
        for tid, (score, agree, certain, cost, count) in data:
            f.write("{},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}\n"
                    .format(tid, score, agree, certain, cost, count))


def plot_difficulty_score_distribution(score_tpls, dst):
    """
    Plots the probability density function (PDF) of difficulty scores.

    Parameters
    ----------
    score_tpls: list of tuples - each entry corresponds to a single tweet and is
    represented by a tuple of (difficulty score, label agreement, classifier
    certainty, relative counts). These scores must be binned.
    dst: str - path in which plot should be stored.

    """
    # In some groups there might be no annos and tweets, hence no scores
    if len(score_tpls) > 0:
        # We only need the difficulty scores and discard the rest
        scores, _, _, _,_ = zip(*score_tpls)
        # Number of bins used in the histogram = square root of elements in
        # <scores>
        bins = int(round(math.sqrt(len(scores))))
        print "BINS:", bins
        fig = plt.figure(figsize=(5, 3))
        ax = fig.add_subplot(111)
        # x = range(max(anno_times))

        # Make sure that the sum of the bars equals 1
        # https://stackoverflow.com/questions/3866520/plotting-histograms-whose-bar-heights-sum-to-1-in-matplotlib/16399202#16399202
        weights = np.ones_like(scores) / float(len(scores))
        n, bins, _ = ax.hist(scores, bins=bins, weights=weights,
                             histtype='stepfilled', alpha=0.2)
        # n, bins, _ = ax.hist(scores, bins=bins, normed=True,
        #                      histtype='stepfilled', alpha=0.2)
        # Hide the right and top spines (lines)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        # Limits of axes
        x = ax.get_xlim()
        plt.xlim(0, x[1])
        # plt.ylim(0, max(max(n), max(y_fit)))
        # Set labels of axes
        ax.set_xlabel("Difficulty score")
        ax.set_ylabel("Probability")
        # ax.set_ylabel("Probability")
        # Add legend outside of plot
        legend = ax.legend(loc="best", shadow=True, bbox_to_anchor=(0.5, 1.5))
        plt.savefig(dst, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    # Get the absolute path to the parent directory of /scripts/
    base_dir = os.path.abspath(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), os.pardir))

    # Directory in which rankings of tweets w.r.t. labeling costs is stored
    COST_DIR = os.path.join(base_dir, "results", "stats",
                            "rank_tweets_by_labeling_costs_early_late")
    # Directory in which rankings of tweets w.r.t. annotator agreement is stored
    AGREE_DIR = os.path.join(base_dir, "results", "stats",
                             "rank_tweets_by_label_agreement_early_late")
    # Directory in which rankings of tweets w.r.t. predictor certainty is stored
    CERTAIN_DIR = os.path.join(base_dir, "results", "stats",
                               "rank_tweets_by_classifier_certainty_early_late")
    # Directory in which statistical tests will be stored
    STAT_DIR = os.path.join(base_dir, "results", "stats",
                            "difficulty_scores_early_late")
    # Directory in which figures will be stored
    FIG_DIR = os.path.join(base_dir, "results", "figures",
                           "difficulty_scores_early_late")

    if not os.path.exists(FIG_DIR):
        os.makedirs(FIG_DIR)
    if not os.path.exists(STAT_DIR):
        os.makedirs(STAT_DIR)
    MIN_ANNOS = 1
    train_ratio = 0.4
    # Consider only <use_n> first tweets of each annotator
    use_n = 50
    # True if thresholds over S, M, L (first <use_n> tweets per annotator)
    # should be used instead of S+M (all tweets)
    use_n_threshold = True

    # 1. Compute tweet difficulty scores for early stage
    main(CERTAIN_DIR, AGREE_DIR, COST_DIR, "md", FIG_DIR, STAT_DIR, s=True,
         m=True, cleaned=True, min_annos=MIN_ANNOS, is_early=True,
         train_ratio=train_ratio, use_n=use_n, use_n_threshold=use_n_threshold)

    main(CERTAIN_DIR, AGREE_DIR, COST_DIR, "su", FIG_DIR, STAT_DIR, s=True,
         m=True, cleaned=True, min_annos=MIN_ANNOS, is_early=True,
         train_ratio=train_ratio, use_n=use_n, use_n_threshold=use_n_threshold)

    # 2. Compute tweet difficulty scores for late stage
    main(CERTAIN_DIR, AGREE_DIR, COST_DIR, "md", FIG_DIR, STAT_DIR, s=True,
         m=True, cleaned=True, min_annos=MIN_ANNOS, is_early=False,
         train_ratio=train_ratio, use_n=use_n, use_n_threshold=use_n_threshold)

    main(CERTAIN_DIR, AGREE_DIR, COST_DIR, "su", FIG_DIR, STAT_DIR, s=True,
         m=True, cleaned=True, min_annos=MIN_ANNOS, is_early=False,
         train_ratio=train_ratio, use_n=use_n, use_n_threshold=use_n_threshold)
