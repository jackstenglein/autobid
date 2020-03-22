#!/usr/bin/python

from common import *

import aws
import argparse
import os
import glob
import re
import math
import csv
import sys

import gensim                       # sudo pip install -U --ignore-installed gensim
from gensim import corpora, models

def delete_file(filename):
    try:
        os.remove(filename)
    except:
        print("Unexpected error when removing file:", sys.exc_info()[0])

class Bid:
    def __init__(self, score, submission):
        self.score = score
        self.submission = submission

def get_mu():
    return 1 #0.25

def normalize_word_count(reviewer, corpus, word):
    # Calculated using Dirichlet smoothing
    mu = get_mu()
    normalized_word_count = (reviewer.num_words / float(reviewer.num_words + mu)) * \
                            (reviewer.feature_vector[word] / float(reviewer.num_words)) \
                            + \
                            (mu / float(reviewer.num_words + mu)) * \
                            (corpus.feature_vector[word] / float(corpus.num_words)) 
    return normalized_word_count

def create_bid(reviewer, corpus, submission, method):
    # Calculate s_{rp}, i.e., the score for reviewer r on paper p, from the TPMS paper
    s_rp = 0

    stop_words = set()
    if method == "stop":
        stop_words = get_stop_words()
    elif method == "smallstop":
        stop_words = get_small_stop_words()

    for word,count in submission.feature_vector.iteritems():
#        # Try something simpler
#        s_rp += count * (reviewer.feature_vector[word] / float(reviewer.num_words))
#        continue
        if word in stop_words:
            continue

        # Calculated using Dirichlet smoothing
        normalized_word_count = normalize_word_count(reviewer, corpus, word) 
        if normalized_word_count > 0:
            s_rp += count * math.log(normalized_word_count, 2)
        elif normalized_word_count < 0:
            print("WARNING: Got negative normalized_word_count of", normalize_word_count, "for", word)

    # Normalize score by length (i.e., number of words)
    s_rp = s_rp / float(submission.num_words)
    return Bid(s_rp, submission)

def compare_submission_bids(reviewer, corpus, s1, s2, method):
    diffs = []
    pos_diff_total = 0
    neg_diff_total = 0
    
    stop_words = set()
    if method == "stop":
        stop_words = get_stop_words()
    elif method == "smallstop":
        stop_words = get_small_stop_words()

    for word in reviewer.feature_vector.keys():
        if word in stop_words:
            continue
        normalized_word_count = normalize_word_count(reviewer, corpus, word)
        score1 = s1.feature_vector[word] * normalized_word_count / float(s1.num_words)
        score2 = s2.feature_vector[word] * normalized_word_count / float(s2.num_words)
        diff = score1 - score2
        if diff > 0:
            pos_diff_total += diff
        else:
            neg_diff_total += diff
        diffs.append((word, diff))

    sorted_diffs = sorted(diffs, key=lambda t: t[1], reverse=True)
    for word, diff in sorted_diffs:
        percent = 100 * (diff / pos_diff_total if diff > 0 else diff / neg_diff_total)
        print("%s\t%f (%0.1f%%)" % (word.ljust(30), diff, percent))

def normalize_bids(bids):
    sorted_bids = sorted(bids, key=lambda bid: bid.score, reverse=True)
    max_bid = sorted_bids[0].score
    min_bid = sorted_bids[-1].score
    norm_bids = []
    target_min = -90
    target_max = 100
    for bid in bids:
        new_score = int(round(target_min + (bid.score - min_bid) / \
                    float((max_bid - min_bid) / \
                          float(target_max - target_min))))
        norm_bid = Bid(score=new_score, submission=bid.submission)
        norm_bids.append(norm_bid)
    return norm_bids

def write_bid_file(filename, bids):
    bid_out = "preference,paper\n"
    for bid in sorted(bids, key=lambda bid: bid.score, reverse=True):
        #print "%0.2f,%s" % (bid.score, bid.submission.id)
        bid_out += "%d,%s\n" % (bid.score, bid.submission.name)

    with open(filename, 'w') as bid_file:
        bid_file.write(bid_out)

    aws.upload(filename, 'cs380s-security-project', 'bids/' + filename)
    delete_file(filename)

def create_reviewer_bid(reviewer, submissions, lda_model):
    print("Creating bid for reviewer", reviewer.name())

    # Analyze topics for the reviewer 
    reviewer_topic_list = lda_model[lda_model.id2word.doc2bow(reviewer.words)]
    reviewer_topic_dict = dict(reviewer_topic_list)

    # Create the raw bid
    bids = []
    for submission in submissions.values():
        # Analyze topics in the submission 
        submission_topics = lda_model[lda_model.id2word.doc2bow(submission.words)]

        score = 0
        for topic_id, topic_prob in submission_topics:
            reviewer_prob = reviewer_topic_dict.get(topic_id, 0)

            score += topic_prob * reviewer_prob

        b = Bid(score, submission)
        bids.append(b)

    bids = normalize_bids(bids)
    write_bid_file(reviewer.name() +  ".csv", bids)
    print("Creating bid for reviewer", reviewer.name(), "complete!")


def parse_real_prefs(realprefs_csvfile):
    prefs = {}
    print("Parsing real preferences...")
    with open(realprefs_csvfile, 'rb') as csv_file:
        reader = csv.DictReader(csv_file, delimiter="\t")
        for row in reader:
            id = row['contactId']
            bid = Bid(int(row['preference']), Submission("ignore", row['paperId']))
            if not(id in prefs):
                prefs[id] = [bid]
            else:
                prefs[id].append(bid)
    print("Parsing real preferences complete!")
    return prefs 

def dump_real_prefs(realprefs_csvfile, pc_ids_file, reviewers):
    id_mapping = match_reviewers_to_ids(reviewers, pc_ids_file)
    real_prefs = parse_real_prefs(realprefs_csvfile)

    print("Dumping reviewers' real preferences...")
    for reviewer in reviewers:
        if not reviewer in id_mapping:
            print("Couldn't find an id for %s.  Skipping." % reviewer)
            continue
        id = id_mapping[reviewer]

        if not id in real_prefs:
            print("Couldn't find an id %s for reviewer %s in set of real preferences.  Skipping." % (id, reviewer))
            continue
        real_bids = real_prefs[id]

        write_bid_file(reviewers[reviewer].dir(), "real_bid.csv", real_bids)
    print("Dumping reviewers' real preferences complete!")


def load_2017_prefs(reviewers):
    print("Loading reviewers bids from 2017...")
    reviewers2017 = {}
    for reviewer in reviewers.values():
        prefs_file = glob.glob('%s/oakland17-revprefs*.csv' % reviewer.dir())
        if not(prefs_file == []):
            prefs_file = prefs_file[0]
            prefs = []
            with open(prefs_file, 'rb') as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    pref = -200
                    if row['preference'] == 'conflict':
                        pref = -100
                    elif is_digit(row['preference']):
                        pref = int(row['preference'])
                    else:
                        print("WARNING: Unknown preference %s for 2017 reviewer %s" % (row['preference'], reviewer))

                    b = Bid(score=pref, submission=row['paper'])
                    prefs.append(b)
            reviewers2017[reviewer] = prefs

    print("Loaded 2017 preferences for %d reviewers" % len(reviewers2017))
    print("Loading reviewers bids from 2017 complete!")
    return reviewers2017

def load_submissions(submissions_dir):
    print("Loading submissions...")
    pickle_file = "%s/submissions.dat" % submissions_dir
    submissions = None
    with open(pickle_file, "rb") as pickler:
        submissions = pickle.load(pickler)

    print("Loading submissions complete!")
    return submissions

def load_model(corpus_dir):
    print("Loading LDA model...")
    pickle_file = "%s/lda.model" % corpus_dir
    lda_model = gensim.models.ldamodel.LdaModel.load(pickle_file)
    print("Loading LDA model complete")
    return lda_model


def create_bids(pc, submissions, lda_model):
    for reviewer in pc.reviewers():
        create_reviewer_bid(reviewer, submissions, lda_model)

def main():
    parser = argparse.ArgumentParser(description='Generate reviewer bids')
    parser.add_argument('-c', '--cache', help="Use the specified file for caching reviewer data", required=True)
    parser.add_argument('--submissions', action='store', help="Directory of submissions", required=True)
    parser.add_argument('--bid', action='store', help="Calculate bids for one reviewer", required=False)
    parser.add_argument('--corpus', action='store', help="Directory of PDFs from which to build a topic (LDA) model", required=True)
    parser.add_argument('--realprefs', action='store', help="File containing real preferences from the MySQL db", required=False)
    parser.add_argument('--s1', action='store', help="First submission to compare a reviewer's calculated bid", required=False)
    parser.add_argument('--s2', action='store', help="Second submission to compare a reviewer's calculated bid", required=False)
    parser.add_argument('--b2017', action='store_true', default=False, help="Load 2017 bids", required=False)
    
    args = parser.parse_args()

    pc = PC()
    pc.load(args.cache)

    submissions = load_submissions(args.submissions)
    
    lda_model = load_model(args.corpus)

#    if args.b2017:
#        load_2017_prefs(reviewers)
#        sys.exit()

#    if not (args.s1 == None or args.s2 == None) and (is_digit(args.s1) and is_digit(args.s2)):
#        corpus = build_corpus(reviewers)
#        compare_submission_bids(reviewers[args.bid], corpus, submissions[int(args.s1)], submissions[int(args.s2)], args.bidmethod)
#        sys.exit(0)
#
#    if not args.realprefs == None:
#        if not args.pcids:
#            print "Mapping reviewer preferences to individual bids requires PC IDs.  Use --pcids filename.csv"
#        else:
#            dump_real_prefs(args.realprefs, args.pcids, reviewers)


    if not args.bid == None:
        create_reviewer_bid(pc.reviewer(args.bid), submissions, lda_model)
    else:
        create_bids(pc, submissions, lda_model)


if (__name__=="__main__"):
  main()

