"""
Tests significance for tables manually created in patterns_simulation.txt

Tests if the observed effect is significant.

Treatments: easy/difficult tweets
Outcomes: early/late stage
Null hypothesis H0:
The proportions of the encoded outcomes 1's (0's) [1's] and 2's (2's) [0's]
is the same in the early and late stage.

http://www.physics.csbsju.edu/stats/exact.html

How to visualize results:
http://www.biostathandbook.com/chiind.html

"""
import scipy.stats

if __name__ == "__main__":
    # 1 vs. 2:
    #         | early   late
    # --------------------------
    # 1       |   11     33
    # 2       |   12      9
    print "using 2-10 tweets in training set"
    print "1 vs. 2"
    odds_ratio, p = scipy.stats.fisher_exact([[12, 35], [8, 4]])
    print odds_ratio, p
    if p < 0.05:
        print "reject H0"
        print "1 occurs preferably in late stage, 2 in early stage. " \
              "probability for more extreme distribution is {:.3f}%".format(
                p * 100)
    else:
        print "accept H0"
    # 1 vs. 0:
    #         | early   late
    # --------------------------
    # 0       |   31     12
    # 1       |   11     33

    print "0 vs. 1"
    odds_ratio, p = scipy.stats.fisher_exact([[34, 15], [13, 35]])
    print odds_ratio, p
    if p < 0.05:
        print "reject H0"
        print "1 occurs preferably in late stage, 0 in early stage. " \
              "probability for more extreme distribution is {:.3f}%".format(
                p * 100)
    else:
        print "accept H0"
    print "0 vs. 2"
    #  2 vs. 0:
    #         | early   late
    # --------------------------
    # 0       |   31     12
    # 2       |   12      9
    odds_ratio, p = scipy.stats.fisher_exact([[34, 15], [8, 4]])
    print odds_ratio, p
    if p < 0.05:
        print "reject H0"
    else:
        print "accept H0"
        print "there's no significant difference in the proportions between " \
              "0 and 2. " \
              "probability for more extreme distribution is {:.3f}%".format(
                 p * 100)

