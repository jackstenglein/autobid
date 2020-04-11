from tqdm import tqdm
from tqdm.auto import trange
import numpy as np
import argparse
import matplotlib.pyplot as plt
import multiprocessing
import pickle

import common
import bid

class TopicData:
    """
    Class to save topic related data to speed up the experiments.
    """
    def __init__(self):
        self.rev_top = {}
        self.sub_top = {}
        self.top_wds = {}

    def populate(self, m, reviewers, submissions):
        for rev in tqdm(reviewers):
            self.rev_top[rev.name()] = m[m.id2word.doc2bow(rev.words)]
        for sub in tqdm(submissions):
            self.sub_top[sub.name] = m[m.id2word.doc2bow(sub.words)]
        for t_id, wds in tqdm(m.show_topics(num_topics=m.num_topics, formatted=False)):
            self.top_wds[t_id] = wds

    @classmethod
    def load(cls, cache_dir):
        print("Loading topic data...")
        with open("%s/topic_data.dat" % cache_dir, "rb") as pickler:
            res = pickle.load(pickler)
        print("Loading topic data complete!")
        return res

    def save(self, cache_dir):
        with open("%s/topic_data.dat" % cache_dir, "wb") as pickler:
            pickle.dump(self, pickler)

def bids_for_rev(rev, td, submissions):
    """
    Generate bids for the reviewer for all provided submissions
    """
    bids = []
    rev_top_dict = dict(td.rev_top[rev.name()])
    for sub in submissions:
        doc_top_list = td.sub_top[sub.name]
        score = 0
        for t_id, t_prob in doc_top_list:
            score += rev_top_dict.get(t_id, 0) * t_prob
        bids.append(score)
    return bids

def bids_for_doc(doc_top_list, td, reviewers):
    """
    Generate bids for the document for all provided reviewers
    """
    bids = {}
    for rev in reviewers:
        rev_top_dict = dict(td.rev_top[rev.name()])
        score = 0
        for t_id, t_prob in doc_top_list:
            score += rev_top_dict.get(t_id, 0) * t_prob
        bids[rev.name()] = score
    return bids

def save_unchanged_bids(reviewers, submissions, td, cache_dir):
    """
    Pre-calculate all the bids and cache the raw results to speedup the
    experiments
    """
    old_bids = np.zeros((len(reviewers), len(submissions)), dtype=np.float32)
    for r_idx in trange(len(reviewers)):
        rev = reviewers[r_idx]
        old_bids[r_idx, :] = bids_for_rev(rev, td, submissions)
    with open("%s/old_bids.dat" % cache_dir, "wb") as pickler:
        pickle.dump(old_bids, pickler)

def load_unchanged_bids(cache_dir):
    """
    Loads the pre-calculated bids from cache
    """
    print("Loading unchanged bids...")
    with open("%s/old_bids.dat" % cache_dir, "rb") as pickler:
        old_bids = pickle.load(pickler)
    print("Loading unchanged bids complete!")
    return old_bids

def save_adv_word_probs(reviewers, td, cache_dir):
    """
    Pre-calculate all adversarial word probabilities and cache the results to
    speedup the experiments
    """
    rev_word_prob = {}
    pool = multiprocessing.Pool()

    inputs = [(rev, td) for rev in reviewers]
    print("Starting parallel work")
    outputs = pool.starmap(adv_word_probs_for_rev, inputs)
    print("Done")
    for i, rev in enumerate(reviewers):
        rev_word_prob[rev.name()] = outputs[i]
    with open("%s/adv_words.dat" % cache_dir, "wb") as pickler:
        pickle.dump(rev_word_prob, pickler)
    pool.close()

def load_adv_word_probs(cache_dir):
    print("Loading adversarial word probabilities...")
    rev_word_prob = {}
    with open("%s/adv_words.dat" % cache_dir, "rb") as pickler:
        rev_word_prob = pickle.load(pickler)
    print("Loading adversarial word probabilities complete!")
    return rev_word_prob

def adv_word_probs_for_rev(rev, td):
    """
    Return adversarial word probabilities for the reviewer
    """
    # Get topics for the reviewer
    rev_top_list = td.rev_top[rev.name()]
    wds_prob = {}
    s = 0
    for t_id, t_prob in rev_top_list:
        # Get words contributing to each topic
        t_wds = td.top_wds[t_id]
        for w, w_prob in t_wds:
            # Weight each word by the topic's probability and the word's
            # contribution to that topic
            if w not in wds_prob:
                wds_prob[w] = 0
            wds_prob[w] += t_prob * w_prob
            s += t_prob * w_prob
    for w in wds_prob:
        wds_prob[w] = wds_prob[w] / s
    return wds_prob

def words_from_probs(wds_prob, num_words):
    """
    Return list of nearly `num_words` words as per their probability
    distribution
    """
    wds = []
    for w in wds_prob:
        for _ in range(int(round(wds_prob[w] * num_words))):
            wds.append(w)
    return wds

def main():
    parser = argparse.ArgumentParser(description='Attack Autobid')
    parser.add_argument('-c', '--cache', help="Directory storing the pickled data about reviewers, submissions, and LDA model", required=True)

    args = parser.parse_args()

    # Load all the cached data
    pc = common.PC()
    pc.load("%s/pc.dat" % args.cache)
    reviewers = list(pc.reviewers())
    submissions = list(bid.load_submissions(args.cache).values())
    m = bid.load_model(args.cache)
    td = TopicData.load(args.cache)
    rev_word_prob = load_adv_word_probs(args.cache)

    old_ranks = []
    new_ranks = []

    # Select random set of reviewers and submissions
    reviewers = np.random.choice(reviewers, 50)
    submissions = np.random.choice(submissions, 10)

    for s_idx in trange(len(submissions), desc="Submissions"):
        sub = submissions[s_idx]
        # Calculate bids for unchanged submission
        old_bids = bids_for_doc(td.sub_top[sub.name], td, reviewers)
        # Find rank of each reviewer for this submission
        old_bids_sorted = sorted(old_bids.keys(),
                key = lambda k: -1 * old_bids[k])
        old_bids_rank = {}
        for i, rev in enumerate(old_bids_sorted):
            old_bids_rank[rev] = i + 1
        # For each reviewer, try to modify the submission so that it gets high
        # bid score for that reviewer
        for r_idx in trange(len(reviewers), desc="Reviewers"):
            rev = reviewers[r_idx]
            old_ranks.append(old_bids_rank[rev.name()])
            # Generate new doc based on adversarial word probs for the reviewer
            new_doc = words_from_probs(rev_word_prob[rev.name()], len(sub.words))
            # Generate new bids
            new_bids = bids_for_doc(m[m.id2word.doc2bow(new_doc + sub.words)], td, reviewers)
            # Find new rank of the reviewer
            rank = 1
            for k in new_bids:
                if new_bids[k] > new_bids[rev.name()]:
                    rank += 1
            new_ranks.append(rank)
    old_ranks = np.array(old_ranks)
    new_ranks = np.array(new_ranks)
    print("# reviewers: %d, # submissions: %d" % (len(reviewers),
        len(submissions)))
    print("Stat\t\told_rank\tnew_rank")
    print("Avg. rank\t%f\t%f" % (np.mean(old_ranks), np.mean(new_ranks)))
    print("Rank 1\t\t%d\t%d" % (np.count_nonzero(old_ranks == 1), np.count_nonzero(new_ranks == 1)))
    print("Top 3\t\t%d\t%d" % (np.count_nonzero(old_ranks <= 3), np.count_nonzero(new_ranks <= 3)))

if __name__ == "__main__":
    main()

