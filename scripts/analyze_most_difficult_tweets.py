"""
Identify at which positions the tweets, which were most difficult to label,
were annotated. Calculate the variance to show that they are equally distributed
over the whole annotation process. Difficulty of tweets is measured by the
avg. annotation time of a tweet.
Why 30 tweets? Because it's 20% (Pareto distribution, randomly assumed) of M
which we can analyze best since we have many annotators and tweets (the latter
isn't true for S while the former is wrong for L).
"""
import matplotlib.pyplot as plt
import os

import numpy as np
from bson.objectid import ObjectId
from statsmodels import robust

import utility


def find_k_most_difficult_tweets(
        dbs, fig_dst, stat_dst, anno_coll_name="user", tweet_coll_name="tweets",
        cleaned=False, k=30, use_median=False):
    """
    Plots the average/median annotation times over all annotators (over all
    institutions) for each
    tweet. Does the same thing, but for the different institutions.
    Sorts them in all cases in ascending order before plotting. Also stores
    the k most difficult tweets in a separate .txt file.

    Parameters
    ----------
    dbs: list of strings - names of the existing DBs
    fig_dst: str - directory in which the plot will be stored.
    stat_dst: str - directory in which the stats will be stored.
    anno_coll_name: str - name of the collection holding the annotator data.
    tweet_coll_name: str - name of the collection holding the tweet data.
    cleaned: bool - True if the data should be cleaned, i.e. if tweet is
    "irrelevant", its remaining labels are ignored for computing average
    annotation times.
    k: int - number of tweets that should be retrieved.
    use_median: bool - True if median annotation times should be used instead
    of averages. Otherwise average annotation times will be used to determine
    difficult tweets.

    """
    dataset_type = "raw"
    if cleaned:
        dataset_type = "cleaned"

    avg_type = "average"
    if use_median:
        avg_type = "median"

    # Different visualization for MD and SU
    SU_ALL = [0, 3]
    MD_ALL = [1, 2, 4, 5, 6]
    # Experiment run in MD after election in Jan 2017
    LATER_ALL = [4, 5, 6]
    # institute_dbs = [SU_ALL, MD_ALL, LATER_ALL]
    # institutions = ["su", "md", "later"]

    # {tweet_id1: {anno time1, anno time2, ...}}
    # Stores annotation times over all institutions
    anno_times = {}
    # Same as <anno_times>, but for each institution now
    anno_times_md = {}
    anno_times_su = {}
    anno_times_later = {}

    # This DB contains all 500 tweets used in the experiment
    db_all = "lturannotationtool"
    all_tweets_coll, anno_coll = utility.load_tweets_annotators_from_db(
            db_all, tweet_coll_name, anno_coll_name)
    # a = 0
    # For every tweet
    for tweet_no, tweet in enumerate(all_tweets_coll.find()):
        # Use Twitter ID because _id differs for the same tweet as it was
        # created in multiple DBs.
        twitter_id = tweet["id_str"]
        anno_times[twitter_id] = []
        # Search in every DB for annotations of this tweet
        for db_idx, db in enumerate(dbs):
            tweet_coll, anno_coll = utility.load_tweets_annotators_from_db(
                db, tweet_coll_name, anno_coll_name)
            DUMMY = 0
            # Search in every annotator
            for idx, anno in enumerate(anno_coll.find()):
                username = anno["username"]
                # print "Collect tweets for annotator '{}'".format(username)
                # Tweet IDs labeled by this annotator
                labeled = anno["annotated_tweets"]
                for tid in labeled:
                    t = utility.get_tweet(tweet_coll, tid)
                    twitter_id_other = t["id_str"]
                    # We found a matching entry, so add it's annotation time
                    if twitter_id == twitter_id_other:
                        rel_label = t["relevance_label"][username]
                        l1 = t["relevance_time"][username]
                        # c1 = t["confidence_relevance_time"][username]

                        # Discard remaining labels if annotator chose
                        # "Irrelevant". Consider other sets of labels only
                        # iff either the cleaned dataset should be created
                        # and the label is "relevant" OR
                        # the raw dataset should be used.
                        if (cleaned and rel_label != "Irrelevant") \
                                or not cleaned:
                            l2 = t["fact_time"][username]
                            # c2 = t["confidence_fact_time"][username]
                            # Annotator labeled the 3rd set of labels as well
                            if username in tweet["opinion_label"]:
                                l3 = t["opinion_time"][username]
                                # c3 = t["confidence_opinion_time"][username]
                            else:
                                # 3rd set of labels might not have been
                                # assigned by annotator, so choose some low
                                # constants that max()
                                # calculations won't get affected
                                l3 = DUMMY
                                # c3 = DUMMY
                        else:
                            # Ignore remaining labels
                            l2 = DUMMY
                            # c2 = DUMMY
                            l3 = DUMMY
                            # c3 = DUMMY
                        ls = [l1, l2, l3]
                        # cs = [c1, c2, c3]
                        # Add up relative timers
                        total = sum(ls)
                        # Append time to all suitable data structures
                        # a) over all institutions
                        anno_times[twitter_id].append(total)
                        # b) SU
                        if db_idx in SU_ALL:
                            if twitter_id not in anno_times_su:
                                anno_times_su[twitter_id] = []
                            anno_times_su[twitter_id].append(total)
                        # c) MD (+LATER)
                        if db_idx in MD_ALL or db_idx in LATER_ALL:
                            if twitter_id not in anno_times_md:
                                anno_times_md[twitter_id] = []
                            anno_times_md[twitter_id].append(total)
                        # d) LATER
                        if db_idx in LATER_ALL:
                            if twitter_id not in anno_times_later:
                                anno_times_later[twitter_id] = []
                            anno_times_later[twitter_id].append(total)
        # a += 1
        # if a == 4:
        #     break
        print "Processed annotators:", (tweet_no + 1)
    names = ["total", "su", "md", "later"]
    datasets = [anno_times, anno_times_su, anno_times_md, anno_times_later]
    for dataset, dataset_name in zip(datasets, names):
        # Compute average annotation time per tweet
        avg_times = {}
        for tid, times in dataset.iteritems():
            # print anno_times[tid]
            # Use median or average of annotation times
            if use_median:
                avg = median(dataset[tid])
            else:
                avg = 1.0*sum(dataset[tid]) / len(dataset[tid])
            votes = len(dataset[tid])
            avg_times[tid] = (avg, votes)
        # Sort average annotation times ascendingly
        # THIS ONLY SORTS THE KEYS, so values must be retrieved from <avg_times>
        # http://stackoverflow.com/questions/6349296/sorting-a-dict-with-tuples-as-values
        sorted_avg_times_keys = sorted(avg_times.keys(),
                                       key=lambda x: avg_times[x][0])

        # Store k most difficult (= highest avg. annotation time) tweets in file
        fname = "{}_most_difficult_tweets_{}_{}.txt"\
            .format(dataset_name, avg_type, dataset_type)
        t = "Avg. time"
        if use_median:
            t = "Med. time"
        with open(stat_dst + fname, "w") as f:
            title = "Most difficult tweets"
            f.write(title + "\n")
            f.write("-" * len(title) + "\n\n")
            f.write("{:<19} | {:<9} | {:<6}\n"
                    .format("Twitter ID", t, "#Annotators who labeled tweet"))
            written = 0
            for tid in reversed(sorted_avg_times_keys):
                if written < k:
                    avg_time = avg_times[tid][0]
                    votes = avg_times[tid][1]
                    f.write("{:<21} {:9.2f}   {:<6}\n".format(tid, avg_time,
                                                              votes))
                written += 1

        # Plot infsci2017_results
        fname = "{}_most_difficult_tweets_{}_{}.png"\
            .format(dataset_name, avg_type, dataset_type)
        fig = plt.figure(figsize=(20, 3))
        ax = fig.add_subplot(111)
        width = 0.02
        x = range(len(sorted_avg_times_keys))
        y = [avg_times[t][0] for t in sorted_avg_times_keys]
        ax.bar(x, y, width, color="black")
        # Title
        title = "{} annotation time per tweet".format(avg_type.title())
        if dataset_name == "su" or dataset_name == "md" \
                or dataset_name == "later":
            title = "{} annotation time per tweet in {}"\
                .format(avg_type.title(), dataset_name)
        plt.title(title)
        # Hide the right and top spines (lines)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        # Limits of axes
        plt.xlim(-0.1, x[-1] + 1)
        plt.ylim(0, y[-1] + 0.5)
        # Set labels of axes
        ax.set_xlabel("Tweet")
        ax.set_ylabel("{} annotation time in s".format(avg_type))
        # Add legend outside of plot
        legend = ax.legend(loc="best", shadow=True, bbox_to_anchor=(0.5, 1.5))
        # plt.tick_params(
        #     axis='x',          # changes apply to the x-axis
        #     which='both',      # both major and minor ticks are affected
        #     bottom='off',      # ticks along the bottom edge are off
        #     top='off',         # ticks along the top edge are off
        #     labelbottom='off') # labels along the bottom edge are off
        plt.savefig(fig_dst + fname, bbox_inches='tight')
        plt.close()


def median(lst):
    return np.median(np.array(lst))


def _get_id_from_other_db(tweet_coll, other_tweet_coll, oid):
    """
    Gets the ObjectID from a different DB for a tweet with a certain text.

    Parameters
    ----------
    tweet_coll: pymongo.collection.Collection - current collection whose ID
    isn't correct.
    other_tweet_coll: pymongo.collection.Collection - other collection that
    contains the correct tweet ID.
    oid: bson.objectid.ObjectId - tweet ID in <tweet_coll>.

    Returns
    -------
    bson.objectid.ObjectId.
    ID of the tweet with the given text in the other DB.

    """
    # Get tweet text from current collection and search in other collection for
    # a tweet with the same text. Use this other tweet's object ID.
    tweet = tweet_coll.find_one({"_id": ObjectId(oid)})
    print "ID of tweet with ID {}: {}".format(oid, tweet["id_str"])
    t = other_tweet_coll.find_one({"id_str": tweet["id_str"]})
    print "actual ID:", t["_id"]
    return t["_id"]


def analyze_variance_of_k_most_difficult_tweets_per_group(
        dbs, db_idxs, name, src, fig_dst, stat_dst, anno_coll_name="user",
        tweet_coll_name="tweets", cleaned=False, use_median=False):
    """
    Lists for each annotator group at which positions a given tweet was labeled.
    This information is used to compute average and variance of the
    positions per group. The same computation can be done using median and
    median absolute deviation (MAD). A high variance would indicate an equally
    likely occurrence of difficult strings during the annotations which would be
    desirable.
    Note that the tweets are sorted wrt. avg. annotation times (according to
    find_k_most_difficult_tweets()), i.e. in plots the tweets with the highest
    annotation times start with low x-values, while in the text file the
    tweets with high annotation times come first in the list.

    Parameters
    ----------
    dbs: list of strings - names of the existing DBs.
    db_idxs: list of ints - name of the MongoDB from where data should be read.
    name: string - name of the institution
    src: string - path to the input file storing the avg. annotation times
    per tweet and the number of annotators who labeled them.
    fig_dst: str - directory in which the plot will be stored.
    stat_dst: str - directory in which the stats will be stored.
    anno_coll_name: str - name of the collection holding the annotator data.
    tweet_coll_name: str - name of the collection holding the tweet data.
    cleaned: bool - True if the cleaned data is used as input.
    use_median: bool - True if median instead of average should be used for
    calculating the annotation time for the i-th tweet. Otherwise the average
    is used for the computation.

    """
    tweet_ids = []
    # Read in data from input file
    with open(src, "r") as f:
        content = f.readlines()
        # Skip first 4 lines as they contain header info and so on
        for line in content[4:]:
            tid = line.split(" ")[0]
            tweet_ids.append(tid)

    # Store annotation times per tweet: {tweet_id: [time1, time2, time3..]}
    # For group S
    s = {}
    # For group M
    m = {}
    # For group L
    l = {}
    # For each tweet:
    for tid in tweet_ids:
        # For each DB that represents the investigated institution
        for db_idx in db_idxs:
            db = dbs[db_idx]
            tweet_coll, anno_coll = utility.load_tweets_annotators_from_db(
                db, tweet_coll_name, anno_coll_name)
            # Find tweet in DB
            tweet = tweet_coll.find_one({"id_str": tid})
            # If tweet exists in DB (300 tweets only exist in L and not S/M)
            if tweet:
                # Get the MongoDB ID of the tweet in that particular DB
                db_id = tweet["_id"]
                # Look at all annotators who labeled this tweet
                annotator_names = tweet["relevance_label"].keys()
                # Retrieve the position at which the given tweet was labeled
                for username in annotator_names:
                    anno = anno_coll.find_one({"username": username})
                    group = anno["group"]
                    labeled = anno["annotated_tweets"]
                    # Search at which position the given tweet was labeled by
                    # that annotator
                    for pos, db_id_other in enumerate(labeled):
                        if db_id == db_id_other:
                            # It's 0-based, so add 1
                            pos += 1
                            # Add label position to the correct group
                            if group == "S":
                                if tid not in s:
                                    s[tid] = []
                                s[tid].append(pos)
                            if group == "M":
                                if tid not in m:
                                    m[tid] = []
                                m[tid].append(pos)
                            if group == "L":
                                if tid not in l:
                                    l[tid] = []
                                l[tid].append(pos)
                            # No need to search any further
                            break
            # break
    data_type = "raw"
    if cleaned:
        data_type = "cleaned"
    agg_type = "average"
    if use_median:
        agg_type = "median"
    fname = "{}_{}_label_position_per_group_{}.txt".format(name, agg_type,
                                                           data_type)
    # Store infsci2017_results in txt file
    with open(stat_dst+fname, "w") as f:
        # Compute mean and standard deviation of the position at which a tweet
        # is labeled per tweet and per group
        group_names = ["S", "M", "L"]
        group_values = [s, m, l]
        title = "{} position per annotator group".format(agg_type.title())
        f.write(title + "\n")
        f.write("-" * len(title) + "\n\n")
        pos_type = "Average pos."
        dev_type = "Standard sample deviation"
        if use_median:
            pos_type = "Median pos."
            dev_type = "Median absolute deviation"
        f.write("{:<4} | {:<13} | {:<25} | {:<5}\n"
                .format("Group", pos_type, dev_type, "Votes"))
        for idx, group in enumerate(group_names):
            # Average position of a tweet in the group
            group_avg = []
            # Go through each tweet ID and add the list of stored annotation
            #  times
            for poss in group_values[idx].itervalues():
                group_avg.extend(poss)
            # Use either mean or median for calculations
            if not use_median:
                mean_group, std_group = _calculate_mean_and_std(group_avg)
            else:
                mean_group, std_group = _calculate_median_and_mad(group_avg)
            f.write("{:<5}   {:13.2f}   {:25.2f}   {:<5}\n"
                    .format(group_names[idx], mean_group, std_group,
                            len(group_avg)))

        # Calculate mean position + standard sample deviation per tweet
        # It should be relatively messy
        title = "\n{} position per tweet (sorted from most difficult " \
                "to easiest) over all groups\n".format(agg_type.title())
        f.write(title + "\n")
        f.write("-" * len(title) + "\n\n")
        f.write("{:<19} | {:<13} | {:<25} | {:<5}\n"
                .format("Twitter ID", pos_type, dev_type, "Votes"))
        for tweet_id in tweet_ids:
            times = []
            if tweet_id in s:
                times.extend(s[tweet_id])
            if tweet_id in m:
                times.extend(m[tweet_id])
            if tweet_id in l:
                times.extend(l[tweet_id])
            # Use either mean or median for calculations
            if not use_median:
                mean_avg, std_avg = _calculate_mean_and_std(times)
            else:
                mean_avg, std_avg = _calculate_median_and_mad(times)
            f.write("{:<19}   {:13.2f}   {:25.2f}   {:<5}\n"
                    .format(tweet_id, mean_avg, std_avg, len(times)))

        # Calculate mean position + standard sample deviation per tweet per
        # group.
        for idx, group in enumerate(group_names):
            y = []
            errors = []
            title = "\n{} position per tweet (sorted from most " \
                    "difficult to easiest) for {}".format(agg_type.title(),
                                                          group)
            f.write(title + "\n")
            f.write("-" * len(title) + "\n\n")
            f.write("{:<19} | {:<13} | {:<25} | {:<5}\n"
                    .format("Twitter ID", pos_type, dev_type, "Votes"))
            # Store all raw times to calculate overall deviation per group
            inst_times = []
            for tweet_id in tweet_ids:
                times = []
                if tweet_id in group_values[idx]:
                    times.extend(group_values[idx][tweet_id])
                    inst_times.extend(group_values[idx][tweet_id])
                # Use either mean or median for calculations
                if not use_median:
                    mean_avg, std_avg = _calculate_mean_and_std(times)
                else:
                    mean_avg, std_avg = _calculate_median_and_mad(times)
                f.write("{:<19}   {:13.2f}   {:25.2f}   {:<5}\n"
                        .format(tweet_id, mean_avg, std_avg, len(times)))
                # Add data for plotting
                y.append(mean_avg)
                # Deviation * 2 because it should show deviation in both
                # directions
                errors.append(std_avg)

            # Use either mean or median for calculations
            if not use_median:
                mean_avg, std_avg = _calculate_mean_and_std(inst_times)
                avg_label = "Mean, y={:.2f}".format(mean_avg)
                std_label = "STD: +-{:.2f}".format(std_avg)
            else:
                mean_avg, std_avg = _calculate_median_and_mad(inst_times)
                avg_label = "Median, y={:.2f}".format(mean_avg)
                std_label = "MAD: +-{:.2f}".format(std_avg)

            # Plot positions and deviations
            fig = plt.figure(figsize=(20, 3))
            ax = fig.add_subplot(111)
            x = range(len(y))
            ax.errorbar(x, y, yerr=errors, fmt='o')
            # Display median/average as horizontal line
            ax.plot((x[0], x[-1]), (mean_avg, mean_avg), "-", color="red",
                    linewidth=2, label=avg_label)
            # Display deviation
            ax.plot((x[0], x[-1]), (mean_avg+std_avg, mean_avg+std_avg), "-",
                    linewidth=2, color="limegreen", label=std_label)
            ax.plot((x[0], x[-1]), (mean_avg-std_avg, mean_avg-std_avg), "-",
                    linewidth=2, color="limegreen", label=std_label)
            plt.title("{} labeling position in group {} "
                      "(most difficult tweets have low x-values)"
                      .format(agg_type.title(), group))
            # plt.ylim(0, max(y) + 3)
            # Hide the right and top spines (lines)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            # Only show ticks on the left and bottom spines
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            # Set labels of axes
            ax.set_xlabel("Tweet ID")
            ax.set_ylabel("{} labeling position".format(agg_type.title()))
            # Add legend outside of plot
            legend = ax.legend(loc='best', shadow=True,
                               bbox_to_anchor=(0.5, 1.7))
            fname = "{}_{}_{}_position_{}.png".format(name, group.lower(),
                                                      agg_type, data_type)
            # Save plot
            plt.savefig(fig_dst + fname, bbox_inches='tight')
            plt.close()


def _calculate_mean_and_std(l):
    """
    Calculates mean position of a given group and its standard sample
    deviation.

    Parameters
    ----------
    l: list of int - each int represents a position at which the tweet was
    labeled by an annotator.

    Returns
    -------
    float, float.
    Mean and sample deviation of the list.

    """
    mean = np.nan
    std = np.nan
    # Calculate mean iff list isn't empty
    if len(l) > 0:
        vals = np.array(l)
        # Discard np.nan values
        vals = vals[~np.isnan(vals)]
        mean = np.mean(vals)
        # Use sample deviation (divide by n-1) instead of population
        # deviation (divide by n) since we use only a sample of the whole
        # population
        std = np.std(vals, ddof=1)
    return mean, std


def _calculate_median_and_mad(l):
    """
    Calculates median position of a given group and its median absolute
    deviation (robust version of standard deviation).

    Parameters
    ----------
    l: list of int - each int represents a position at which the tweet was
    labeled by an annotator. Potentially contains np.nan values.

    Returns
    -------
    float, float.
    Median and median absolute deviation of the list.

    """
    med = np.nan
    mad = np.nan
    # Calculate mean iff list isn't empty
    if len(l) > 0:
        vals = np.array(l)
        # Discard np.nan values
        vals = vals[~np.isnan(vals)]
        med = np.median(vals)
        # Use sample deviation (divide by n-1) instead of population
        # deviation (divide by n) since we use only a sample of the whole
        # population
        mad = robust.mad(vals)
    return med, mad


if __name__ == "__main__":
    # Names of all the MongoDBs which contain data from experiments
    DB_NAMES = [
        "lturannotationtool",
        "mdlannotationtool",
        "mdsmannotationtool",
        "turannotationtool",
        "harishannotationtool",
        "kantharajuannotationtool",
        "mdslaterannotationtool",
        ]
    # Name of the collection in each DB holding annotator data
    ANNO_COLL_NAME = "user"
    # Name of the collection in each DB holding tweet data
    TWEET_COLL_NAME = "tweets"
    # Directory in which figures will be stored
    FIG_DIR = "/media/data/Workspaces/PythonWorkspace/phd/Analyze-Labeled-Dataset/sac2018_results/figures/most_difficult_tweets/"
    STAT_DIR = "/media/data/Workspaces/PythonWorkspace/phd/Analyze-Labeled-Dataset/sac2018_results/stats/most_difficult_tweets/"
    if not os.path.exists(FIG_DIR):
        os.makedirs(FIG_DIR)
    if not os.path.exists(STAT_DIR):
        os.makedirs(STAT_DIR)

    # Find all difficult tweets over all institutions and per institution
    # and store them - time-consuming, hence commented out
    # Compute it with mean and standard deviation
    # find_k_most_difficult_tweets(DB_NAMES, FIG_DIR, STAT_DIR, ANNO_COLL_NAME,
    #                              TWEET_COLL_NAME, k=600)
    # find_k_most_difficult_tweets(DB_NAMES, FIG_DIR, STAT_DIR, ANNO_COLL_NAME,
    #                              TWEET_COLL_NAME, cleaned=True, k=600)
    # # Compute it with median and median absolute deviation
    find_k_most_difficult_tweets(DB_NAMES, FIG_DIR, STAT_DIR, ANNO_COLL_NAME,
                                 TWEET_COLL_NAME, k=150, use_median=True)
    find_k_most_difficult_tweets(DB_NAMES, FIG_DIR, STAT_DIR, ANNO_COLL_NAME,
                                 TWEET_COLL_NAME, cleaned=True, k=150,
                                 use_median=True)
    avg_types = ["median"]

    # Indices of DBs to consider for analyses
    ALL = [0, 1, 2, 3, 4, 5, 6]
    SU_ALL = [0, 3]
    MD_ALL = [1, 2, 4, 5, 6]
    LATER_ALL = [4, 5, 6]
    dbs = [MD_ALL, ALL, SU_ALL]
    # names = ["total", "md", "su", "later"]
    names = ["md", "total", "su"]

    # For each file
    for idcs, f in zip(dbs, names):
        # For each avg mode
        for avg_type in avg_types:
            raw_name = "{}_most_difficult_tweets_{}_raw.txt"\
                .format(f, avg_type)
            cleaned_name = "{}_most_difficult_tweets_{}_cleaned.txt"\
                .format(f, avg_type)
            # Use average and standard deviation for annotation times that
            # were averaged. Use median and median absolute deviation (MAD) for
            # files where the median was used to compute average annotation
            # times. Mixing those two different measurements up with computing
            # standard deviation and MAD wouldn't make much sense.
            if avg_type == "average":
                # Identify the standard deviation/median absolute deviation
                analyze_variance_of_k_most_difficult_tweets_per_group(
                    DB_NAMES, idcs, f, STAT_DIR + raw_name, FIG_DIR,
                    STAT_DIR, ANNO_COLL_NAME, TWEET_COLL_NAME)

                analyze_variance_of_k_most_difficult_tweets_per_group(
                    DB_NAMES, idcs, f, STAT_DIR + cleaned_name, FIG_DIR,
                    STAT_DIR, ANNO_COLL_NAME, TWEET_COLL_NAME, cleaned=True)
            else:
                # Repeat analysis with median instead of avg., i.e. compute median and
                # median absolute deviation (MAD)
                # This analysis could be more robust because a few outliers (as in our data)
                # won't influence the infsci2017_results much. Standard deviation is affected by the
                # sample size, so we use MAD instead which is more robust. See plots:
                # http://stackoverflow.com/questions/22354094/pythonic-way-of-detecting-outliers-in-one-dimensional-observation-data
                # TODO: does use of MAD make sense?
                # http://stats.stackexchange.com/questions/29116/mean-pmsd-or-median-pmmad-to-summarise-a-highly-skewed-variable
                # I think the distribution is kinda symmetric with few elements being
                # larger/smaller than median (see histogram "most_difficult_tweets_cleaned")
                analyze_variance_of_k_most_difficult_tweets_per_group(
                    DB_NAMES, idcs, f, STAT_DIR + raw_name, FIG_DIR,
                    STAT_DIR, ANNO_COLL_NAME, TWEET_COLL_NAME, use_median=True)

                analyze_variance_of_k_most_difficult_tweets_per_group(
                    DB_NAMES, idcs, f, STAT_DIR + cleaned_name, FIG_DIR,
                    STAT_DIR, ANNO_COLL_NAME, TWEET_COLL_NAME, cleaned=True,
                    use_median=True)
