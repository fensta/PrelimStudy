"""
Provides the implementations of 2 objects storing information from the dataset:
Tweet and Annotator.

IMPORTANT:
----------
In MD (M) there was a small bug, s.t. 3 annotators labeled the same tweet
twice (these annotators have IDs 1, 2, and 10), so their 2nd annotations for
that tweet are ignored. Therefore, these annotators labeled only 149 instead
of 150 tweets.

Sentiment labels are referred to as "annotation labels" throughout the script.
There are also labels available for annotators rating their confidence in
assigning these said sentiment labels.

"""
import warnings
import unicodecsv as csv
import os


# Indicates that a tweet has no labels
NA = " "
# Default label assigned by annotator - it indicates no label was assigned
# for a certain hierarchy level, e.g. ["Relevant", "Factual", "?"] means
# on 1st level "Relevant" was assigned, on 2nd level "Factual", hence the
# 3rd level wasn't displayed anymore, so "?" is used
EMPTY = "?"
# Default value for annotation time
ZERO = 0

# Keys for dictionary
TEXT = "text"
LABEL = "label"
TIME = "time"
TIMES = "times"
TID = "tid"
UID = "uid"
WID = "wid"
VOTES = "votes"
ORDERED = "ordered"
LABELED = "labeled"
DS_NAME = "ds_type"
GROUP = "group"
CONF_TIMES = "conf_times"
CONF_LABEL = "conf_label"

# Labels to use for majority labels
# I think your suggestion for deriving sentiment is fine. That means for
# our existing labels from the previous labeling experiments in Magdeburg and
# Sabanci we only consider sentiment of relevant tweets and otherwise declare
# them irrelevant.
GROUND_TRUTH = {
    "Irrelevant": "Irrelevant",
    "Relevant": "Relevant",
    "Factual": "Factual",
    "Non-factual": "Non-factual",
    "Positive": "Positive",
    "Negative": "Negative",
    EMPTY: EMPTY    # Only added to make the code work - used for placeholders
}


def read_workers_from_csv(src):
    """
    Reads in workers from training set in csv format separated by commas.

    Parameters
    ----------
    src: str - path where csv file is stored.

    Returns
    -------
    dict.
    Workers.
    {
        <GROUP>: # Annotator group
        {
            wid: # Worker ID
                {
                <LABELED>: ...# Number of labeled tweets by this annotator
                <TYPE>: ... # Dataset name
                <ORDERED>: [tid1, tid2,...] # Order in which tweets were labeled
                }
        }
    }

    """
    with open(src, "rb") as f:
        reader = csv.reader(f, delimiter=",")
        workers = {}
        for row in reader:
            wid, ds_name, group, labeled, ordered = row[0], row[1], row[2], \
                                                    row[3], row[4:]
            if group not in workers:
                workers[group] = {}
            workers[group][wid] = {}
            workers[group][wid][DS_NAME] = ds_name
            workers[group][wid][LABELED] = labeled
            workers[group][wid][ORDERED] = ordered
    return workers


def read_tweets_from_csv(src):
    """
    Reads in tweets from dataset in csv format separated by commas.

    Parameters
    ----------
    src: str - path where csv file is stored.
    assumed that the hierarchical one is used.

    Returns
    -------
    dict.
    Tweets.
    {tid:
        {
            <VOTES>: ...., # Number of workers who labeled the tweet
            # Annotation time of i-th annotator for j-th hierarchy level
            <TIMES>: [[time1 by anno1, time2 by anno1, time3 by anno1], [..]]
            <LABEL>: [[label1 by anno1, label2 by anno1, label3 by anno1], [..]]
            <UID>: .... # ID of tweet author (to retrieve tweets from Twitter)
            <TEXT>: ... # Text of the tweet - to be downloaded via Twitter API
            <WID>: [ID of anno1 who labeled this tweet, ID of anno2...]
            <CONF_TIMES>: same as <TIMES>, but for confidence times
            <CONF_LABEL>: same as <LABEL>, but for confidence labels
            <TYPE>: ...
        }
    }

    """
    with open(src, "rb") as f:
        reader = csv.reader(f, delimiter=",")
        tweets = {}
        for row in reader:
            tid, uid, ds_name, text, votes, rest = row[0], row[1], row[2], \
                                             row[3], int(row[4]), row[5:]
            labels = []
            times = []
            conf_labels = []
            conf_times = []
            wids, labelss, timess, conf_labelss, conf_timess = \
                rest[:votes], rest[votes: 3*votes+votes], \
                rest[3*votes+votes: 6*votes+votes], \
                rest[6*votes+votes: 9*votes+votes], \
                rest[9*votes+votes: 12*votes+votes]

            # 3 sentiment labels per worker
            for i in xrange(0, len(labelss), 3):
                one, two, three = labelss[i], labelss[i+1], labelss[i+2]
                labels.append([one, two, three])
            # 3 annotation times per worker
            for i in xrange(0, len(timess), 3):
                one, two, three = float(timess[i]), float(timess[i+1]), \
                                  float(timess[i+2])
                times.append([one, two, three])
            # 3 confidence labels per worker
            for i in xrange(0, len(conf_labelss), 3):
                one, two, three = conf_labelss[i], conf_labelss[i + 1], \
                                  conf_labelss[i + 2]
                conf_labels.append([one, two, three])
            # 3 confidence times per worker
            for i in xrange(0, len(conf_timess), 3):
                one, two, three = float(conf_timess[i]), \
                                  float(conf_timess[i + 1]), \
                                  float(conf_timess[i + 2])
                conf_times.append([one, two, three])
            # Store info for tweet
            tweets[tid] = {}
            tweets[tid][VOTES] = votes
            tweets[tid][LABEL] = labels
            tweets[tid][WID] = wids
            tweets[tid][UID] = uid
            tweets[tid][TIMES] = times
            tweets[tid][DS_NAME] = ds_name
            tweets[tid][TEXT] = text
            tweets[tid][CONF_LABEL] = conf_labels
            tweets[tid][CONF_TIMES] = conf_times
    return tweets


class Data(object):
    """
    Acts as a wrapper for the tweets and annotators read from a csv file.
    """
    def __init__(self, tweet_path, anno_path):
        """
        Reads in the tweet and annotator datasets in csv format and creates
        objects that can be used for more convenient access to the data.

        Parameters
        ----------
        tweet_path: str - path to csv file containing tweets.
        anno_path: str - path to csv file containing the annotators who labeled
        these tweets in <tweet_path>.

        """
        tweets = read_tweets_from_csv(tweet_path)
        # print "#tweets", len(tweets)
        # all_votes = 0
        # for t in tweets:
        #     print t, tweets[t]
        #     all_votes += tweets[t][VOTES]
        # print "total votes", all_votes
        annos = read_workers_from_csv(anno_path)
        self.annotators = Annotators()
        self.tweets = Tweets()
        # For each labeled
        for tid in tweets:
            uid = tweets[tid][UID]
            # Store collection of this tweet
            tc = TweetColl()
            tc.update(tid, uid, tweets[tid][WID],
                      tweets[tid][TIMES], tweets[tid][CONF_TIMES],
                      tweets[tid][LABEL], tweets[tid][CONF_LABEL],
                      tweets[tid][TEXT], tweets[tid][DS_NAME])
            self.tweets.add_tweet_coll(tc)
            # Create Tweet object for each annotator who labeled it
            for idx, aid in enumerate(tweets[tid][WID]):
                t = Tweet()
                t.update(tid, tweets[tid][TIMES][idx],
                         tweets[tid][CONF_TIMES][idx], tweets[tid][LABEL][idx],
                         tweets[tid][CONF_LABEL][idx],
                         tweets[tid][TEXT], tweets[tid][DS_NAME])

                # Create annotator
                if not self.annotators.anno_exists(aid):
                    a = Annotator()
                    # Find info about annotator - it must be available in one
                    # of the groups
                    for group in annos:
                        if aid in annos[group]:
                            a.update(aid, uid, group,
                                     annos[group][aid][DS_NAME],
                                     annos[group][aid][ORDERED])
                    self.annotators.add_annotator(a)

                # Add tweet
                self.annotators.add_tweet(aid, t)
        # Make sure that the Tweet objects are sorted in each annotator
        # according to self.labeled, i.e. in the order the annotator labeled
        # all tweets
        for anno in self.annotators.all_annos():
            anno.sort_by_label_order()


class TweetColl(object):
    """Represents a collection of all annotations of a single tweet"""
    def __init__(self):
        # Tweet ID
        self.tid = -1
        # Tweet user ID, i.e. ID of tweet author - important to download
        # tweets with Twitter API
        self.uid = -1
        # List of annotator IDs who labeled this tweet
        self.labeled_by = []
        # List of lists of annotation times. The i-th inner list represents the
        # annotation times of the i-th annotator. Each inner list contains
        # 3 entries, one per hierarchy level. The order of the inner lists
        # is the same as in self.labeled_by
        self.anno_times = []
        # Same as self.anno_times, but for annotation labels
        self.anno_labels = []
        # Same as self.anno_times, but for confidence times
        self.conf_times = []
        # Same as self.anno_times, but for confidence labels
        self.conf_labels = []
        # Name of the dataset
        self.ds_name = "None"
        # Text of tweet
        self.text = "None"

    def __str__(self):
        return "{}:  {} {} {} {} {} {}"\
            .format(self.tid, self.labeled_by, self.text, self.anno_labels,
                    self.anno_times, self.conf_labels, self.conf_times,
                    self.ds_name)

    def update(self, tid, uid, labeled_by, anno_times, conf_times, anno_labels,
               conf_labels, text, ds_name):
        """Updates the data stored for the current TweetColl object."""
        self.tid = tid
        self.uid = uid
        self.labeled_by = labeled_by
        self.anno_times = anno_times
        self.conf_times = conf_times
        self.anno_labels = anno_labels
        self.conf_labels = conf_labels
        self.text = text
        self.ds_name = ds_name

    def get_labelers(self):
        """Returns the list of all annotator IDs who labeled that tweet"""
        return self.labeled_by

    def get_anno_labels(self):
        """Returns the sentiment labels of the different hierarchy levels
        for that tweet of the different annotators. The i-th inner list
        represents the labels of the i-th annotator according to
        self.labeled_by"""
        return self.anno_labels

    def get_anno_times(self):
        """Returns the time needed to assign the sentiment labels for the
        different hierarchy levelsThe i-th inner list represents the
        annotation times of the i-th annotator according to self.labeled_by"""
        return self.anno_times

    def get_conf_labels(self):
        """Returns the confidence labels of the different hierarchy levels
        for that tweet. The i-th inner list represents the labels of the i-th
        annotator according to self.labeled_by."""
        return self.conf_labels

    def get_conf_times(self):
        """Returns the time needed to assign the sentiment labels for the
        different hierarchy levels. The i-th inner list represents the
        confidence times of the i-th annotator according to self.labeled_by."""
        return self.conf_times


class Tweet(object):
    """
    Wrapper class to store data about tweets.
    """
    def __init__(self):
        # Tweet ID
        self.tid = -1
        # List of separate annotation times per hierarchy level
        self.anno_times = []
        # List of separate confidence times per hierarchy level
        self.conf_times = []
        # List of assigned annotation labels per hierarchy level
        self.anno_labels = []
        # List of assigned confidence labels per hierarchy level
        self.conf_labels = []
        # Actual tweet message
        self.text = ""
        # Dataset name from which the tweet is taken
        self.ds_name = "None"

    def __str__(self):
        return "{}: {} {} {} {} {} {}"\
            .format(self.tid, self.text, self.anno_labels, self.anno_times,
                    self.conf_labels, self.conf_times, self.ds_name)

    def update(self, tid, anno_times, conf_times, anno_labels, conf_labels,
               text, ds_name):
        """Updates the data stored for the current Tweet object."""
        self.tid = tid
        self.anno_times = anno_times
        self.conf_times = conf_times
        self.anno_labels = anno_labels
        self.conf_labels = conf_labels
        self.text = text
        self.ds_name = ds_name

    def get_anno_labels(self):
        """Returns the sentiment labels of the different hierarchy levels
        for that tweet"""
        return self.anno_labels

    def get_anno_times(self):
        """Returns the time needed to assign the sentiment labels for the
        different hierarchy levels"""
        return self.anno_times

    def get_conf_labels(self):
        """Returns the confidence labels of the different hierarchy levels
        for that tweet"""
        return self.conf_labels

    def get_conf_times(self):
        """Returns the time needed to assign the sentiment labels for the
                different hierarchy levels"""
        return self.conf_times

    def get_tid(self):
        """Returns tweet ID"""
        return self.tid

    def get_text(self):
        """Returns tweet message"""
        return self.text

    def get_dataset_name(self):
        """Returns the name of the dataset to which the tweet belongs."""
        return self.ds_name


class Tweets(object):
    """Represents the collection of all annotations for each tweet."""
    def __init__(self):
        # Index of current tweet
        self.idx = 0
        # Tweet IDs that exist and their index in the list
        # {tid: index}
        self.tids = {}
        # List of TweetColl objects
        self.tweet_coll = []

    def __len__(self):
        return len(self.tids)

    def add_tweet_coll(self, tc):
        """Adds a TweetColl object."""
        self.tids[tc.tid] = self.idx
        self.idx += 1
        self.tweet_coll.append(tc)

    def get_tweet(self, tid):
        """Returns the collection of the tweet with ID <tid>."""
        idx = self.tids[tid]
        return self.tweet_coll[idx]

    def get_all_tweet_ids(self):
        """List of all labeled tweet IDs"""
        return self.tids.keys()

    #############
    # Iterators #
    #############
    def all_tweets(self):
        """
        Returns an iterator over all TweetColl objects.

        """
        for tc in self.tweet_coll:
            yield tc


class Annotator(object):
    def __init__(self):
        # Annotator ID
        self.aid = -1
        # ID of the author of the tweet - necessary to download this tweet using
        # the Twitter API
        self.uid = -1
        # List of tweet Objects labeled by this annotator in the ascending
        # order, i.e. tweet at index 0 was labeled first etc.
        self.tweets = []
        # Annotator group
        self.group = "None"
        # Dataset to which annotator belongs
        self.ds = "None"
        # List of tids this annotator labeled - first tid was labeled first
        self.labeled = []

    def __str__(self):
        return "Anno {} ({}) labeled tweets ({}) in {}"\
            .format(self.aid, self.group, len(self.tweets), self.ds)

    def update(self, aid, uid, group, ds, labeled):
        """Updates an Annotator object."""
        self.aid = aid
        self.uid = uid
        self.group = group
        self.ds = ds
        self.labeled = labeled

    def add_tweet(self, t):
        """Add Tweet object for annotator"""
        # for tw in self.tweets:
        #     if tw.tid == t.tid:
        #         print "already exists!!!!!!!!!!!!!!!!", t.tid, self.aid
        self.tweets.append(t)

    def get_tweet(self, tid):
        """Returns a Tweet object with tweet ID <tid>."""
        idx = self.labeled[tid]
        return self.tweets[idx]

    def get_ith_tweet(self, i):
        """
        Gets the i-th labeled tweet by this annotator.

        Parameters
        ----------
        i: int - index of tweet.

        Returns
        -------
        Tweet object or None if the i-th tweet doesn't exist.

        """
        if 0 <= i < len(self.tweets):
            return self.tweets[i]
        else:
            # For annotators 1, 2, 10 a bug occurred s.t. they labeled the same
            # tweet twice, so their 2nd annotations are ignored and hence they
            # only labeled 149 instead of 150 tweets. For the sake of
            # convenience, if anyone tries to access their 150th tweet, it'll be
            # ignored silently
            if self.aid not in ["1", "2", "10"] and i != 150:
                warning = "For Anno {} a tweet at index {} doesn't exist!"\
                    .format(self.aid, i)
                warnings.warn(warning)
        return None

    def get_ith_anno_labels(self, i):
        """
        Gets the annotation labels of the i-th tweet by this annotator.

        Parameters
        ----------
        i: int - index of tweet.

        Returns
        -------
        Tweet object or None if the i-th tweet doesn't exist.

        """
        if 0 <= i < len(self.tweets):
            return self.tweets[i].anno_labels
        else:
            # For annotators 1, 2, 10 a bug occurred s.t. they labeled the same
            # tweet twice, so their 2nd annotations are ignored and hence they
            # only labeled 149 instead of 150 tweets. For the sake of
            # convenience, if anyone tries to access their 150th tweet, it'll be
            # ignored silently
            if self.aid not in ["1", "2", "10"] and i != 150:
                warning = "For Anno {} a tweet at index {} doesn't exist!" \
                    .format(self.aid, i)
                warnings.warn(warning)
        return None

    def get_ith_conf_labels(self, i):
        """
        Gets the confidence labels of the i-th tweet by this annotator.

        Parameters
        ----------
        i: int - index of tweet.

        Returns
        -------
        Tweet object or None if the i-th tweet doesn't exist.

        """
        if 0 <= i < len(self.tweets):
            return self.tweets[i].conf_labels
        else:
            # For annotators 1, 2, 10 a bug occurred s.t. they labeled the same
            # tweet twice, so their 2nd annotations are ignored and hence they
            # only labeled 149 instead of 150 tweets. For the sake of
            # convenience, if anyone tries to access their 150th tweet, it'll be
            # ignored silently
            if self.aid not in ["1", "2", "10"] and i != 150:
                warning = "For Anno {} a tweet at index {} doesn't exist!" \
                    .format(self.aid, i)
                warnings.warn(warning)
        return None

    def get_ith_anno_times(self, i):
        """
        Gets the annotation times of the i-th tweet by this annotator.

        Parameters
        ----------
        i: int - index of tweet.

        Returns
        -------
        Tweet object or None if the i-th tweet doesn't exist.

        """
        if 0 <= i < len(self.tweets):
            return self.tweets[i].anno_times
        else:
            # For annotators 1, 2, 10 a bug occurred s.t. they labeled the same
            # tweet twice, so their 2nd annotations are ignored and hence they
            # only labeled 149 instead of 150 tweets. For the sake of
            # convenience, if anyone tries to access their 150th tweet, it'll be
            # ignored silently
            if self.aid not in ["1", "2", "10"] and i != 150:
                warning = "For Anno {} a tweet at index {} doesn't exist!" \
                    .format(self.aid, i)
                warnings.warn(warning)
        return None

    def get_ith_conf_times(self, i):
        """
        Gets the confidence times of the i-th tweet by this annotator.

        Parameters
        ----------
        i: int - index of tweet.

        Returns
        -------
        Tweet object or None if the i-th tweet doesn't exist.

        """
        if 0 <= i < len(self.tweets):
            return self.tweets[i].conf_times
        else:
            # For annotators 1, 2, 10 a bug occurred s.t. they labeled the same
            # tweet twice, so their 2nd annotations are ignored and hence they
            # only labeled 149 instead of 150 tweets. For the sake of
            # convenience, if anyone tries to access their 150th tweet, it'll be
            # ignored silently
            if self.aid not in ["1", "2", "10"] and i != 150:
                warning = "For Anno {} a tweet at index {} doesn't exist!" \
                    .format(self.aid, i)
                warnings.warn(warning)
        return None

    def get_group(self):
        """Returns the annotator group of the annotator."""
        return self.group

    def get_id(self):
        """Returns the ID of the annotator."""
        return self.aid

    def get_labeled_tweets_ids(self):
        """Returns a list of tweet IDs that this annotator labeled."""
        return self.labeled

    def get_labeled_tweets(self):
        """Returns the list of labeled Tweet objects."""
        return self.tweets

    def get_name(self):
        """Returns the name of the annotator."""
        return "Annotator {}".format(self.aid)

    def get_dataset_name(self):
        """Returns the name of the dataset to which the annotator belongs."""
        return self.ds

    def sort_by_label_order(self):
        """Sort Tweet objects in self.tweets according to the order in which
        the annotator labeled them, namely according to self.labeled."""
        ordered_tweets = []
        for tid in self.labeled:
            for t in self.tweets:
                if t.tid == tid:
                    ordered_tweets.append(t)
        self.tweets = ordered_tweets

    #############
    # Iterators #
    #############
    def labeled_tweets(self):
        """Returns an iterator over the annotator's labeled Tweet objects"""
        for t in self.tweets:
            yield t


class Annotators(object):
    def __init__(self):
        # Index of current annotator
        self.idx = 0
        # IDs of annotators that exist and their index in the list
        # {aid: index}
        self.aids = {}
        # List of Annotator objects
        self.annos = []

    def __len__(self):
        return len(self.aids)

    def add_annotator(self, anno):
        """Adds an Annotator <anno> to Annotators"""
        self.aids[anno.aid] = self.idx
        self.idx += 1
        self.annos.append(anno)

    def add_tweet(self, aid, t):
        """Adds a tweet <t> for the annotator with ID <aid>."""
        idx = self.aids[aid]
        self.annos[idx].add_tweet(t)

    def get_ith_annotator(self, i):
        """Gets the i-th annotator object"""
        return self.annos[i]

    def get_annotator(self, aid):
        """Returns the annotator with ID <aid>."""
        idx = self.aids[aid]
        return self.annos[idx]

    def anno_exists(self, aid):
        """Returns true if an annotator with ID <aid> exists."""
        return True if aid in self.aids else False

    #############
    # Iterators #
    #############
    def all_annos(self):
        """
        Returns an iterator over all Annotator objects.

        """
        for anno in self.annos:
            yield anno


if __name__ == "__main__":
    # Get the absolute path to the parent directory of /scripts/
    base_dir = os.path.abspath(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), os.pardir))
    DS_DIR = os.path.join(base_dir, "results", "anonymous")
    # Store MD
    fname = "tweets_hierarchical_md.csv"
    dst_tweets_md = os.path.join(DS_DIR, fname)
    fname = "annotators_hierarchical_md.csv"
    dst_workers_md = os.path.join(DS_DIR, fname)

    # Test functions for reading the training datasets
    # MD
    data = Data(dst_tweets_md, dst_workers_md)
    print "#workers in MD", len(data.annotators)
    all_votes = 0
    for w in data.annotators.all_annos():
        print w.aid, w.uid, len(w.tweets), w.group, w.ds
        # Just for testing
        print "collection"
        tid = w.tweets[0].tid
        print data.tweets.get_tweet(tid)
        all_votes += len(w.tweets)
    print "total votes", all_votes
    print "tweet collections", len(data.tweets)

    for t in data.tweets.all_tweets():
        print t
        # Just to try out some more functions
        if t.tid == "780594886311411712":
            print "labeled by", t.get_labelers()
            for aid in t.get_labelers():
                print data.annotators.get_annotator(aid)
