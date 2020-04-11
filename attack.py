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
    bids = []
    for rev in reviewers:
        rev_top_dict = dict(td.rev_top[rev.name()])
        score = 0
        for t_id, t_prob in doc_top_list:
            score += rev_top_dict.get(t_id, 0) * t_prob
        bids.append(score)
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
    old_bids = load_unchanged_bids(args.cache)

    n = 1000

    old_sub_rank_in_rev = np.zeros(n, dtype=int)
    old_rev_rank_in_sub = np.zeros(n, dtype=int)

    new_sub_rank_in_rev = np.zeros(n, dtype=int)
    new_rev_rank_in_sub = np.zeros(n, dtype=int)

    for i in trange(n, desc="Trials"):
        r_idx = np.random.randint(0, len(reviewers))
        s_idx = np.random.randint(0, len(submissions))

        sub = submissions[s_idx]
        rev = reviewers[r_idx]

        # Generate new doc based on adversarial word probs for the reviewer
        new_doc = words_from_probs(rev_word_prob[rev.name()], len(sub.words))
        # Generate new bids for this updated submission
        new_bids = bids_for_doc(m[m.id2word.doc2bow(new_doc + sub.words)],
                td, reviewers)

        # Find old rank of sub in rev's list
        rank = 1
        for b in old_bids[r_idx, :]:
            if b > old_bids[r_idx, s_idx]:
                rank += 1
        old_sub_rank_in_rev[i] = rank

        # Find old rank of rev in sub's list
        rank = 1
        for b in old_bids[:, s_idx]:
            if b > old_bids[r_idx, s_idx]:
                rank += 1
        old_rev_rank_in_sub[i] = rank

        # Find new rank of sub in rev's list
        rank = 1
        for b in old_bids[r_idx, :]:
            if b > new_bids[r_idx]:
                rank += 1
        new_sub_rank_in_rev[i] = rank

        # Find new rank of rev in sub's list
        rank = 1
        for b in new_bids:
            if b > new_bids[r_idx]:
                rank += 1
        new_rev_rank_in_sub[i] = rank

    print("# reviewers: %d, # submissions: %d" % (len(reviewers),
        len(submissions)))
    print("# trials: %d" % n)
    print("\nRank of submission in reviewer's list:")
    print("---------------------------------------")
    print("Stat\t\told\tnew")
    print("Avg\t%f\t%f" % (np.mean(old_sub_rank_in_rev),
        np.mean(new_sub_rank_in_rev)))
    print("Top 1\t\t%d\t%d" % (np.count_nonzero(old_sub_rank_in_rev == 1),
        np.count_nonzero(new_sub_rank_in_rev == 1)))
    print("Top 3\t\t%d\t%d" % (np.count_nonzero(old_sub_rank_in_rev <= 3),
        np.count_nonzero(new_sub_rank_in_rev <= 3)))
    print("\nRank of reviewer in submission's list:")
    print("---------------------------------------")
    print("Stat\t\told\tnew")
    print("Avg\t%f\t%f" % (np.mean(old_rev_rank_in_sub),
        np.mean(new_rev_rank_in_sub)))
    print("Top 1\t\t%d\t%d" % (np.count_nonzero(old_rev_rank_in_sub == 1),
        np.count_nonzero(new_rev_rank_in_sub == 1)))
    print("Top 3\t\t%d\t%d" % (np.count_nonzero(old_rev_rank_in_sub <= 3),
        np.count_nonzero(new_rev_rank_in_sub <= 3)))

if __name__ == "__main__":
    main()

