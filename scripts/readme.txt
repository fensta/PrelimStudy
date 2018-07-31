Order in which the scripts must be run to obtain results of our technical report
1. rank_tweets_by_difficulty_via_classifier_certainty_for_early_late_stage.py --> rank tweets according to classifier certainty using only early/late tweets
2. rank_tweets_by_difficulty_via_label_agreement_for_early_late_stage.py --> rank tweets according to label agreement using only early/late tweets
3. rank_tweets_by_difficulty_via_labeling_costs_for_early_late_stage.py --> rank tweets according to labeling costs using only early/late tweets
4. compute_difficulty_score.py --> computes per tweet a difficulty score
5. add_difficulty_labels.py --> assigns each tweet an easy/difficult label --> we have established a ground truth (separately for 4 strata: a) easy tweets labeled in early stage, b) difficult tweets labeled in early stage, c) easy tweets labeled in late stage, d) difficult tweets labeled in late stage
6. effect_of_difficulty_on_reliability.py --> perform simulation and obtain plots
7. patterns_simulation.txt manually --> data needed for statistical test
8. analyze_label_distribution_easy_difficult.py (independent of all other scripts) --> plots the label distributions per stratum
9. effect_of_difficulty_on_reliability_significance.py --> perform Fisher's exact test to analyze the ratios of the distributions from 7.