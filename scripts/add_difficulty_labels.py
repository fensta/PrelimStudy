"""
Assigns to each tweet in an institution/group a difficulty label easy/difficult.

First, compute a cut-off value to distinguish among ranked difficulty scores
between easy and difficult. Perform this within early and late stage separately.
To determine cut-off value, run k-means with k=2 which clusters the scores.

To assign easy/difficult labels to a tweet Y, follow this procedure:
Case 1: Y appears in only one stage. Then, we know its label.
Case 2: Y appears in both stages and has the same label in both stages.
        Then, we know its label.
Case 3: Y appears in both stages and has different labels in each stage. Then,
        we can resolve the tie by checking how far the difficulty-score inside
        each stage is from the stage's cut-off.
        E.g. cut-off for easy tweets is 3.2 and cut-off for difficult tweets
        is 3.0 and Y's difficulty score is 3.6, i.e. it's closer to easy
        tweets (higher scores indicate easier tweets), so easy is assigned to Y.
For case 3 use a two-tailed t-test to check if (difficulty score - cut-off
value) is significantly different from 0 (same test as if checking if a slope
is significantly different from 0). For the t-test we know that the difficulty
scores are normally distributed (see compute_difficulty_score.py) where we
checked if the PDFs of the difficulty scores need to be log-normal transformed
or not - it turns out it's unnecessary.

"""
import os

from sklearn.cluster import KMeans
import numpy as np


class Tweet(object):
    """
    Wrapper class for data. Stores

    self.id = tweet ID
    self.score = tweet difficulty score
    self.agree = label agreement
    self.certain = classifier certainty
    self.cost = labeling cost
    self.count = relative count
    self.cluster = cluster to which the tweet belongs ("easy"/"difficult")

    """
    def __init__(self, id_, score_, agree_, certain_, cost_, count_):
        self.id = id_
        self.score = score_
        self.agree = agree_
        self.certain = certain_
        self.cost = cost_
        self.count = count_
        self.difficulty = ""

    def get_score(self):
        return self.score

    def set_difficulty(self, d):
        """Sets difficulty to the str <d>."""
        self.difficulty = d

    def get_difficulty(self):
        """Gets the tweet difficulty, either "easy" or "difficult"."""
        return self.difficulty

    def __str__(self):
        return self.id

    def __len__(self):
        return len(self.id)


def compute_cut_off(inst_name, fig_dir, stat_dir, diff_dir, s=True,
                             m=True, min_annos=1, cleaned=True, is_early=True,
                             train_ratio=0.66, use_n=50, use_n_threshold=True):
    """
    Computes the cut-off value for an institution/group using k-means with k=2.
    The cut-off value is the threshold that indicates if a tweet difficulty
    score refers to "difficult" or "easy" tweets.

    Parameters
    ----------
    inst_name: str - name of the institution for which data should be loaded.
    fig_dir: str - directory in which the plot will be stored.
    stat_dir: str - directory in which the stats will be stored.
    diff_dir: str - directory from which the tweet difficulty scores are read.
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

    Returns
    -------
    dict.
    Cut-off values for the given institution and groups with the institution/
    groups being keys and the cut-off values being the values. Cut-off values
    are the lowest tweet difficulty scores that refer to "easy" tweets. That
    means that any tweet with a lower difficulty score is considered
    "difficult".

    """
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

    # For an institution
    # 1. Read in the ranked tweet difficulty scores
    if use_n_threshold:
        fname = "{}_min_annos_{}_difficulty_score_{}_train_ratio_{}_use_n_" \
                "{}_{}.txt" \
            .format(inst_name, min_annos, tweet_type, train_ratio, use_n,
                    ds_name)
    else:
        fname = "{}_min_annos_{}_difficulty_score_{}_train_ratio_{}_{}.txt" \
            .format(inst_name, min_annos, tweet_type, train_ratio,
                    ds_name)
    fp = os.path.join(diff_dir, fname)
    tweets = read_scores(fp)
    # 2. Cluster them into easy and difficult using k-means with k=2
    # Tweets with highest difficulty score are easiest.
    x = []
    for t in tweets:
        x.append(t.get_score())
    x = np.array(x)
    # Make k-means work with 1D data
    # https://stackoverflow.com/questions/28416408/scikit-learn-how-to-run-kmeans-on-a-one-dimensional-array
    km = KMeans(n_clusters=2, random_state=42, n_jobs=-1)
    res = km.fit(x.reshape(-1, 1))
    cut_off = {
        inst_name: get_cut_off_value(tweets, res.labels_)
    }
    # fname = "{}_min_annos_{}_difficulty_label_{}_train_ratio_{}_{}.txt" \
    #     .format(inst_name, min_annos, tweet_type, train_ratio,
    #             ds_name)
    # dst = os.path.join(stat_dir, fname)
    # store_difficulty_labels(tweets, dst)

    # For each group
    for group in data:
        if use_n_threshold:
            fname = "{}_{}_min_annos_{}_difficulty_score_{}_train_ratio_{}_" \
                    "use_n_{}_{}.txt" \
                .format(inst_name, group, min_annos, tweet_type, train_ratio,
                        use_n, ds_name)
        else:
            fname = "{}_{}_min_annos_{}_difficulty_score_{}_train_ratio_{}_" \
                    "{}.txt" \
                .format(inst_name, group, min_annos, tweet_type, train_ratio,
                        ds_name)
        fp = os.path.join(diff_dir, fname)
        # 1. Read in the ranked tweet difficulty scores
        tweets = read_scores(fp)
        # Could be empty for some groups
        if len(tweets) > 0:
            # 2. Cluster them into easy and difficult using k-means with k=2
            # Tweets with highest difficulty score are easiest
            x = []
            for t in tweets:
                x.append(t.get_score())
            x = np.array(x)
            # Make k-means work with 1D data
            km = KMeans(n_clusters=2, random_state=42, n_jobs=-1)
            res = km.fit(x.reshape(-1, 1))
            cut_off[group] = get_cut_off_value(tweets, res.labels_)

            # assign_labels_to_tweets(tweets, res.labels_)
            # fname = "{}_{}_min_annos_{}_difficulty_label_{}_train_ratio_{}_{}.txt" \
            #     .format(inst_name, group, min_annos, tweet_type, train_ratio,
            #             ds_name)
            # dst = os.path.join(stat_dir, fname)
            # store_difficulty_labels(tweets, dst)
    return cut_off


def read_scores(fp):
    """
    Reads in a file that stores the tweet difficulty scores.

    Parameters
    ----------
    fp: str - path to file.

    Returns
    -------
    list of Tweet.
    Each tweet represents a row in the input file and stores the relevant data
    (tweet id, difficulty score, label agreement, classifier certainty,
    labeling cost, count).

    """
    tweets = []
    with open(fp, "rb") as f:
        lines = f.readlines()
    # Skip header
    for line in lines[1:]:
        tid, score, agree, certain, cost, count = line.split(",")
        score = float(score)
        agree = float(agree)
        certain = float(certain)
        cost = float(cost)
        count = float(count)
        new_tweet = Tweet(tid, score, agree, certain, cost, count)
        tweets.append(new_tweet)
    return tweets


def get_cut_off_value(tweets, cluster_ids):
    """
    Gets the lowest tweet difficulty score that refers to an "easy" tweet.

    Parameters
    ----------
    tweets: list of Tweet - tweets that should be assigned labels.
    cluster_ids: numpy.ndarray - IDs of the clusters. There are only 2 possible
    values, namely 0 and 1.

    Returns
    -------
    float.
    Cut-off value, i.e. the lowest tweet difficulty score that refers to an
    "easy" tweet. In other words, any score lower than the threshold results in
    a "difficult" tweet.

    """
    # Since tweets are sorted w.r.t. difficulty scores, the first tweet is the
    # easiest one, hence, whatever its cluster ID will be, all other tweets
    # with that ID are also considered "easy" and at least the last tweet's ID
    # must have have the other ID (because we set k=2)
    labels = {cluster_ids[0]: "easy",
              cluster_ids[-1]: "difficult"}
    cut_off = 3
    for t, cid in zip(tweets, cluster_ids):
        if labels[cid] == "easy":
            # Store tweet difficulty score
            cut_off = t.get_score()
    return cut_off


def assign_labels_to_tweets(tweets, cluster_ids):
    """
    Assigns each tweet either the label "easy" or "difficult".

    Parameters
    ----------
    tweets: list of Tweet - tweets that should be assigned labels. Updates the
    clusters accordingly.
    cluster_ids: numpy.ndarray - IDs of the clusters. There are only 2 possible
    values, namely 0 and 1.

    """
    # Since tweets are sorted w.r.t. difficulty scores, the first tweet is the
    # easiest one, hence, whatever its cluster ID will be, all other tweets
    # with that ID are also considered "easy" and at least the last tweet's ID
    # must have have the other ID (because we set k=2)
    labels = {cluster_ids[0]: "easy",
              cluster_ids[-1]: "difficult"}
    print cluster_ids
    for t, cid in zip(tweets, cluster_ids):
        t.set_cluster(labels[cid])


def store_difficulty_labels(tweets, dst):
    """
    Stores the tweets with their assigned labels (easy/difficult) together
    with their difficulty scores and other data in a .txt file, separated by
    commas.

    Parameters
    ----------
    tweets: list of Tweet - each tweet represents a row in the file.
    dst: str - path where output file should be stored.

    """
    with open(dst, "wb") as f:
        f.write("tweet_id,difficulty_label,difficulty_score,weighted_agreement,"
                "classifier_certainty,labeling_cost,relative_count\n")
        for tweet in tweets:
            f.write("{},{},{:.3f},{:.3f},{:.3f},{:.3f}\n"
                    .format(tweet.id, tweet.difficulty, tweet.score,
                            tweet.agree, tweet.certain, tweet.cost,
                            tweet.count))


def assign_difficulty_labels_to_tweets(
        inst_name, fig_dir, stat_dir, diff_dir, cut_off_early, cut_off_late,
        s=True, m=True, min_annos=1, cleaned=True, is_early=True,
        train_ratio=0.66, use_n=50, use_n_threshold=True):
    """
    Assigns each tweet either "easy" or "difficult" as a label. To do so, it
    follows the methodology outlined at the beginning of the script (cases 1-3).
    The results are stored in a txt file.

    Parameters
    ----------
    inst_name: str - name of the institution for which data should be loaded.
    fig_dir: str - directory in which the plot will be stored.
    stat_dir: str - directory in which the stats will be stored.
    diff_dir: str - directory from which the tweet difficulty scores are read.
    cut_off_early: dict - contains cut-off values for tweets of early stage.
    cut_off_late: dict - contains cut-off values for tweets of late stage.
    s: bool - True if group S of <inst> should be considered separately.
    m: bool - True if group M of <inst> should be considered separately.
    min_annos: int - minimum number of annotators who must've assigned a label
    to a tweet. Otherwise it'll be discarded.
    cleaned: bool - True if cleaned dataset should be used. Else the raw one
    is used.
    train_ratio: float - between 0-1, specifies how many percent of the
    tweets should be used for training/testing.
    use_n: int - considers only the first n tweets per annotator, regardless of
    annotator group.
    use_n_threshold: bool - True if early/late stage threshold over S, M, L for
    first <use_n> tweets should be used. Else the thresholds over all tweets of
    S and M is are used.

    """
    ds_name = "cleaned"
    if not cleaned:
        ds_name = "raw"
    # Figure out which files to open
    data = ["S", "M"]
    if s and not m:
        data = ["S"]
    if not s and m:
        data = ["M"]
    if use_n_threshold:
        data = ["S", "M", "L"]

    # ----------------------
    # a) For an institution
    # ----------------------
    # a.1. Load tweets from early stage
    if use_n_threshold:
        e_fname = "{}_min_annos_{}_difficulty_score_early_train_ratio_{}_" \
                  "use_n_{}_{}.txt" \
            .format(inst_name, min_annos, train_ratio, use_n, ds_name)
        l_fname = "{}_min_annos_{}_difficulty_score_late_train_ratio_{}_" \
                  "use_n_{}_{}" \
                  ".txt" \
            .format(inst_name, min_annos, train_ratio, use_n, ds_name)
        eo_fname = "{}_min_annos_{}_difficulty_labels_early_train_ratio_{}_" \
                   "use_n_{}_{}.txt" \
            .format(inst_name, min_annos, train_ratio, use_n, ds_name)
        lo_fname = "{}_min_annos_{}_difficulty_labels_late_train_ratio_{}_" \
                   "use_n_{}_{}.txt" \
            .format(inst_name, min_annos, train_ratio, use_n, ds_name)
    else:
        e_fname = "{}_min_annos_{}_difficulty_score_early_train_ratio_{}_{}" \
                  ".txt" \
            .format(inst_name, min_annos, train_ratio, ds_name)
        l_fname = "{}_min_annos_{}_difficulty_score_late_train_ratio_{}_{}" \
                  ".txt" \
            .format(inst_name, min_annos, train_ratio, ds_name)
        eo_fname = "{}_min_annos_{}_difficulty_labels_early_train_ratio_{}_" \
                   "{}.txt" \
            .format(inst_name, min_annos, train_ratio, ds_name)
        lo_fname = "{}_min_annos_{}_difficulty_labels_late_train_ratio_{}_" \
                   "{}.txt" \
            .format(inst_name, min_annos, train_ratio, ds_name)
    fp = os.path.join(diff_dir, e_fname)
    early_tweets = read_scores(fp)

    # a.2. Load tweets from late stage
    fp = os.path.join(diff_dir, l_fname)
    late_tweets = read_scores(fp)

    # a.3. Assign labels for tweets from early stage
    compute_difficulty_label(early_tweets, late_tweets,
                             cut_off_early[inst_name], cut_off_late[inst_name])

    # a.4. Assign labels for tweets from late stage
    compute_difficulty_label(late_tweets, early_tweets,
                             cut_off_late[inst_name], cut_off_early[inst_name])

    # a.5. Store labels for early stage
    dst = os.path.join(stat_dir, eo_fname)
    store_difficulty_labels(early_tweets, dst)

    # a.6. Store labels for late stage
    dst = os.path.join(stat_dir, lo_fname)
    store_difficulty_labels(late_tweets, dst)

    # ------------------
    # b) For each group
    # ------------------
    for group in data:
        if use_n_threshold:
            ei_fname = "{}_{}_min_annos_{}_difficulty_score_early_train_" \
                       "ratio_{}_use_n_{}_{}.txt" \
                .format(inst_name, group, min_annos, train_ratio, use_n,
                        ds_name)
            li_fname = "{}_{}_min_annos_{}_difficulty_score_late_train_" \
                       "ratio_{}_use_n_{}_{}.txt" \
                .format(inst_name, group, min_annos, train_ratio, use_n,
                        ds_name)
            eo_fname = "{}_{}_min_annos_{}_difficulty_labels_early_train_" \
                       "use_n_{}_ratio_{}_{}.txt" \
                .format(inst_name, group, min_annos, train_ratio, use_n,
                        ds_name)
            lo_fname = "{}_{}_min_annos_{}_difficulty_labels_late_train_" \
                       "ratio_{}_use_n_{}_{}.txt" \
                .format(inst_name, group, min_annos, train_ratio, use_n,
                        ds_name)
        else:
            ei_fname = "{}_{}_min_annos_{}_difficulty_score_early_train_" \
                       "ratio_{}_{}.txt" \
                .format(inst_name, group, min_annos, train_ratio, ds_name)
            li_fname = "{}_{}_min_annos_{}_difficulty_score_late_train_" \
                       "ratio_{}_{}.txt" \
                .format(inst_name, group, min_annos, train_ratio, ds_name)
            eo_fname = "{}_{}_min_annos_{}_difficulty_labels_early_train_" \
                       "ratio_{}_{}.txt" \
                .format(inst_name, group, min_annos, train_ratio, ds_name)
            lo_fname = "{}_{}_min_annos_{}_difficulty_labels_late_train_" \
                       "ratio_{}_{}.txt" \
            .format(inst_name, group, min_annos, train_ratio, ds_name)
        # b.1. Load tweets from early stage
        fp = os.path.join(diff_dir, ei_fname)
        early_tweets = read_scores(fp)

        # b.2. Load tweets from late stage
        fp = os.path.join(diff_dir, li_fname)
        late_tweets = read_scores(fp)

        # b.3. Assign labels for tweets from early stage
        compute_difficulty_label(early_tweets, late_tweets,
                                 cut_off_early[inst_name],
                                 cut_off_late[inst_name])

        # b.4. Assign labels for tweets from late stage
        compute_difficulty_label(late_tweets, early_tweets,
                                 cut_off_late[inst_name],
                                 cut_off_early[inst_name])

        # b.5. Store labels for early stage
        dst = os.path.join(stat_dir, eo_fname)
        store_difficulty_labels(early_tweets, dst)

        # b.6. Store labels for late stage
        dst = os.path.join(stat_dir, lo_fname)
        store_difficulty_labels(late_tweets, dst)


def compute_difficulty_label(tweets, other_tweets, cut_off, other_cut_off):
    """
    Derives tweet difficulty labels for <tweets>, which are tweets labeled
    in a certain stage.

    Parameters
    ----------
    tweets: list of Tweet - tweets for which labels are derived. Updates
    self.difficulty entry of each tweet, i.e. assigned labels are stored there.
    other_tweets: list of Tweet - tweets from the other stage.
    cut_off: float - contains cut-off value for <tweets> for difficulty scores.
    Cut-off value is the lowest value that's still considered "easy", so any
    score smaller than that is "difficult".
    other_cut_off: float - contains cut-off value for <others> for difficulty
    scores. Cut-off value is the lowest value that's still considered "easy",
    so any score smaller than that is "difficult".

    """
    # Store IDs of other tweets for faster look-up together with their index
    # {tid1: index1}
    others_lookup = {t.id: idx for idx, t in enumerate(other_tweets)}
    for tweet in tweets:
        difficulty = derive_difficulty(tweet.score, cut_off)
        # Case 1: does tweet only exist in one of the stages?
        if tweet.id not in others_lookup:
            # print "CASE 1: {} not in others".format(tweet.id)
            # print "set difficulty to {} because of score {} and cut-off {}"\
            #     .format(difficulty, tweet.score, cut_off)
            tweet.set_difficulty(difficulty)
        # Case 2: does the tweet have the same label in both stages?
        else:
            # Get index of same tweet in other stage
            idx = others_lookup[tweet.id]
            other_tweet = other_tweets[idx]
            other_difficulty = derive_difficulty(other_tweet.score,
                                                 other_cut_off)
            # print "other tweet has score {}".format(other_tweet.score)
            if difficulty == other_difficulty:
                # print "CASE 2: {} has {} and {} with cut offs {} and {}" \
                #     .format(tweet.id, difficulty, other_difficulty, cut_off,
                #             other_cut_off)
                # print "Since {} == {}, {} gets this label"\
                #     .format(difficulty, other_difficulty, tweet.id)
                tweet.set_difficulty(difficulty)
            # Case 3: to which cut-off value is the tweet closer in both stages?
            else:
                # print "other tweet has score {}".format(other_tweet.score)
                # print "{} has {} and {} with cut offs {} and {}" \
                #     .format(tweet.id, difficulty, other_difficulty, cut_off,
                #             other_cut_off)
                early_difference = abs(tweet.score - cut_off)
                late_difference = abs(other_tweet.score - other_cut_off)
                # print "CASE 3: {} has differences {} and {} "\
                #     .format(tweet.id, early_difference, late_difference)
                # Tweet in early stage could easier be on the other side of
                # the cut-off value, so assign label of other stage
                if early_difference < late_difference:
                    # print "assign label of other stage", other_difficulty
                    tweet.set_difficulty(other_difficulty)
                else:
                    # print "assign label of this stage", difficulty
                    tweet.set_difficulty(difficulty)


def derive_difficulty(score, cut_off):
    """
    Determines the difficulty of a tweet based on the cut-off value.

    Parameters
    ----------
    score: float - tweet difficulty score.
    cut_off: float - cut-off value is the lowest value that's still considered
    "easy", so any score smaller than that is "difficult".

    Returns
    -------
    str.
    Difficulty, either "easy" or "difficult".

    """
    difficulty = "easy"
    if score < cut_off:
        difficulty = "difficult"
    return difficulty


if __name__ == "__main__":
    # Get the absolute path to the parent directory of /scripts/
    base_dir = os.path.abspath(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), os.pardir))
    diff_score_dir = os.path.join(base_dir, "results", "stats",
                                  "difficulty_scores_early_late")
    # Directory in which statistical tests will be stored
    STAT_DIR = os.path.join(base_dir, "results", "stats",
                            "difficulty_labels_early_late")
    # Directory in which figures will be stored
    FIG_DIR = os.path.join(base_dir, "results", "figures",
                           "difficulty_labels_early_late")

    if not os.path.exists(FIG_DIR):
        os.makedirs(FIG_DIR)
    if not os.path.exists(STAT_DIR):
        os.makedirs(STAT_DIR)
    MIN_ANNOS = 1
    train_ratio = 0.4
    use_n = 50
    # True if thresholds over S, M, L (first <use_n> tweets per annotator)
    # should be used instead of S+M (all tweets)
    use_n_threshold = True
    # a) Determine cut-off difficulty score value that distinguishes "easy" from
    # "difficult" tweets
    # --------------------------------------------------------------------------
    # 1. Compute tweet difficulty labels (easy/difficult) for early stage
    cut1 = compute_cut_off(
        "md", FIG_DIR, STAT_DIR, diff_score_dir, s=True, m=True,
        cleaned=True, min_annos=MIN_ANNOS, is_early=True,
        train_ratio=train_ratio, use_n=use_n, use_n_threshold=use_n_threshold)
    print cut1
    cut2 = compute_cut_off(
        "su", FIG_DIR, STAT_DIR, diff_score_dir, s=True, m=True,
        cleaned=True, min_annos=MIN_ANNOS, is_early=True,
        train_ratio=train_ratio, use_n=use_n, use_n_threshold=use_n_threshold)
    print cut2
    # 2. Compute tweet difficulty labels (easy/difficult) for late stage
    cut3 = compute_cut_off(
        "md", FIG_DIR, STAT_DIR, diff_score_dir, s=True, m=True,
        cleaned=True, min_annos=MIN_ANNOS, is_early=False,
        train_ratio=train_ratio, use_n=use_n, use_n_threshold=use_n_threshold)
    print cut3
    cut4 = compute_cut_off(
        "su", FIG_DIR, STAT_DIR, diff_score_dir, s=True, m=True,
        cleaned=True, min_annos=MIN_ANNOS, is_early=False,
        train_ratio=train_ratio, use_n=use_n, use_n_threshold=use_n_threshold)
    print cut4

    # b) Determine the labels of each tweet following the methodology described
    # on top. That means we compare the labels of the tweets between early and
    # late stage in an institution/groups.
    # --------------------------------------------------------------------------
    assign_difficulty_labels_to_tweets(
        "md", FIG_DIR, STAT_DIR, diff_score_dir, cut1, cut3, s=True, m=True,
        cleaned=True, min_annos=MIN_ANNOS, train_ratio=train_ratio, use_n=use_n,
        use_n_threshold=use_n_threshold)

    assign_difficulty_labels_to_tweets(
        "su", FIG_DIR, STAT_DIR, diff_score_dir, cut2, cut4, s=True, m=True,
        cleaned=True, min_annos=MIN_ANNOS, train_ratio=train_ratio,
        use_n_threshold=use_n_threshold)
