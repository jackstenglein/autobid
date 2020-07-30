from tqdm import tqdm
from tqdm.auto import trange
import numpy as np
from numpy.linalg import norm
import argparse
import matplotlib.pyplot as plt
import multiprocessing
import pickle
import math

import common
import bid

def filter_reviewers(reviewers, min_publications):
    result = []
    for rev in reviewers:
        if len(rev.pdf_names) >= min_publications:
            result.append(rev)
    return result

def filter_submissions(submissions):
    filtered_submissions = []
    for sub in submissions:
        if sub is None or sub.num_words == 0:
            continue
        filtered_submissions.append(sub)
    return filtered_submissions


class BidData:
    """
    Class to save bid related data to speed up the experiments.
    """
    def __init__(self):
        self.normalized_bids = []
        self.rev_max_raw = []
        self.rev_min_raw = []
        self.target_min = -90
        self.target_max = 100

    def get_normalizer(self, min_bid, max_bid):
        return lambda score: int(round(self.target_min + \
                (score - min_bid) / float((max_bid - min_bid) / \
                            float(self.target_max - self.target_min)
                        )))


    def populate(self, td, reviewers, submissions):
        reviewers = filter_reviewers(list(pc.reviewers()), 5)
        self.normalized_bids = np.zeros((len(reviewers), len(submissions)), dtype=np.int)
        for r_idx in trange(len(reviewers)):
            rev = reviewers[r_idx]
            raw_bids, min_bid, max_bid = bids_for_rev(rev, td, submissions)
            self.rev_min_raw.append(min_bid)
            self.rev_max_raw.append(max_bid)
            self.normalized_bids[r_idx, :] = np.vectorize(
                    self.get_normalizer(min_bid, max_bid)
                )(raw_bids)

    @classmethod
    def load(cls, cache_dir):
        print("Loading bid data...")
        with open("%s/bid_data.dat" % cache_dir, "rb") as pickler:
            res = pickle.load(pickler)
        print("Loading bid data complete!")
        return res

    def save(self, cache_dir):
        with open("%s/bid_data.dat" % cache_dir, "wb") as pickler:
            pickle.dump(self, pickler)

class TopicData:
    """
    Class to save topic related data to speed up the experiments.
    """
    def __init__(self):
        self.rev_top = {}
        self.sub_top = {}
        self.top_wds = {}
        self.avg_top_probs = []

    def populate(self, m, reviewers, submissions):
        self.avg_top_probs = [0 for _ in range(m.num_topics)]
        num_revs = len(reviewers)
        for rev in tqdm(reviewers):
            self.rev_top[rev.name()] = sorted(m[m.id2word.doc2bow(rev.words)],
                    key = lambda a: -1 * a[1])
            for t_id, t_prob in self.rev_top[rev.name()]:
                self.avg_top_probs[t_id] += t_prob / num_revs
        for sub in tqdm(submissions):
            self.sub_top[sub.name] = m[m.id2word.doc2bow(sub.words)]
        for t_id, wds in tqdm(m.show_topics(num_topics=m.num_topics, num_words=20, formatted=False)):
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
    bids = np.zeros(len(submissions), dtype=np.float32)
    rev_top_dict = dict(td.rev_top[rev.name()])
    min_bid = math.inf
    max_bid = -1
    for s_idx, sub in enumerate(submissions):
        doc_top_list = td.sub_top[sub.name]
        score = 0
        for t_id, t_prob in doc_top_list:
            score += rev_top_dict.get(t_id, 0) * t_prob
        min_bid = min(min_bid, score)
        max_bid = max(max_bid, score)
        bids[s_idx] = score
    return bids, min_bid, max_bid

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
    for t_id, t_prob in rev_top_list[:3]:
        # Get words contributing to each topic
        t_wds = td.top_wds[t_id]
        for w, w_prob in t_wds[:10]:
            if w not in wds_prob:
                wds_prob[w] = 0
            # Weight each word by the topic's probability 
            prob = t_prob
            wds_prob[w] = max(wds_prob[w], prob)
    s = 0
    for w in wds_prob:
        s += wds_prob[w]
    for w in wds_prob:
        wds_prob[w] = wds_prob[w] / s
    return wds_prob


def adversarialWords(favoredReviewer, allReviewers, topicData, avoid=False):

    probabilityDifferences = []

    # Go through each topic for the favored reviewer
    favoredTopicList = topicData.rev_top[favoredReviewer.name()]
    for topicId, favoredTopicProbability in favoredTopicList:

        probabilityDifference = 0

        # Go through each of the other reviewers and accumulate the differences in 
        # probability for this topic
        for reviewer in allReviewers:
            reviewerTopicDict = dict(topicData.rev_top[reviewer.name()])
            reviewerTopicProbability = reviewerTopicDict.get(topicId, 0)
            probabilityDifference += (favoredTopicProbability - reviewerTopicProbability)
        
        probabilityDifferences.append( (topicId, probabilityDifference) )

    probabilityDifferences = sorted(probabilityDifferences, reverse=(not avoid), key=lambda topic: topic[1])
    # print("Probability differences: ", probabilityDifferences)
    
    wordProbabilities = {}
    for topicId, probabilityDifference in probabilityDifferences:
        if (avoid and probabilityDifference >= 0) or (not avoid and probabilityDifference <= 0):
            break
        
        topicWords = topicData.top_wds[topicId]
        for word, _ in topicWords[:10]:
            if word not in wordProbabilities:
                wordProbabilities[word] = 0
            # Weight each word by the topic's probability difference
            wordProbabilities[word] = max(wordProbabilities[word], abs(probabilityDifference))

    total = 0
    for word in wordProbabilities:
        total += wordProbabilities[word]
    
    for word in wordProbabilities:
        wordProbabilities[word] = wordProbabilities[word] / total

    # print("Word probabilities: ", wordProbabilities)
    return wordProbabilities
    

def numbered_words_from_probs(wds_prob, num_words):
    """
    Return list of adversarial words of size `num_words` as per their probability distribution.
    """
    wds = []
    for w in wds_prob:
        n = int(round(wds_prob[w] * num_words))
        if n <= 0:
            continue
        for _ in range(n):
            wds.append(w)
    return wds


def words_from_probs(wds_prob, sub, size=1):
    """
    Return list of adversarial words for `sub` as per their probability
    distribution
    """
    return numbered_words_from_probs(wds_prob, size * sub.num_words)


def experiment1(allSubmissions, allReviewers, model, bidData, topicData, reviewerWordProbability):

    # Select only submissions with close enough bids
    # Top 3 reviewers must have a score greater than 96
    submissions = []
    oldBids = []
    topReviewers = []
    for subIndex, submission in enumerate(allSubmissions):
        submissionBids = []
        for revIndex, bid in enumerate(bidData.normalized_bids[:, subIndex]):
            submissionBids.append( (revIndex, bid) )
        submissionBids = sorted(submissionBids, reverse=True, key=lambda subBid: subBid[1])
        if submissionBids[2][1] > 96:
            submissions.append(allSubmissions[subIndex])
            oldBids.append(submissionBids[0:3])
            topReviews = []
            for submissionBid in submissionBids[0:10]:
                topReviews.append(allReviewers[submissionBid[0]])
            topReviewers.append(topReviews)

    print("Number of selected submissions: ", len(submissions))
    # print("oldBids: ", oldBids)

    # Go through each selected submission
    # Favor each of the top 3 reviewers and record new bids
    trials = 0
    newBids = []
    num100 = 0
    for subIndex, submission in enumerate(submissions):
        submissionBids = []

        for reviewerBid in oldBids[subIndex]:
            trials += 1

            revIndex = reviewerBid[0]
            reviewer = allReviewers[revIndex]

            # Generate new doc based on adversarial word probs for the reviewer
            fullWordProbabilities = adversarialWords(reviewer, allReviewers, topicData)
            topWordProbabilities = adversarialWords(reviewer, topReviewers[subIndex], topicData)

            topMultiplier = 0.69
            finalWordProbabilities = {}
            for word in fullWordProbabilities:
                finalWordProbabilities[word] = fullWordProbabilities[word]
            for word in topWordProbabilities:
                if word not in finalWordProbabilities:
                    finalWordProbabilities[word] = topWordProbabilities[word] * topMultiplier
                else:
                    finalWordProbabilities[word] = max(finalWordProbabilities[word], topWordProbabilities[word] * topMultiplier)

            new_doc = words_from_probs(finalWordProbabilities, submission) 

            # Generate new bids for this updated submission
            new_bids = bids_for_doc(model[model.id2word.doc2bow(new_doc)], topicData, allReviewers)

            # Normalize new bid using new min and max because we need to compare across different reviewers
            normalizedNewBids = []
            for ri, b in enumerate(new_bids):
                b = bidData.get_normalizer(
                    min(bidData.rev_min_raw[ri], b),
                    max(bidData.rev_max_raw[ri], b)
                )(b)
                normalizedNewBids.append( (ri, b) )
                if b == 100:
                    num100 += 1

            # Get the top 3 reviewers for the updated submission
            normalizedNewBids = sorted(normalizedNewBids, reverse=True, key=lambda bid: bid[1])
            submissionBids.append(normalizedNewBids[0:3])
        newBids.append(submissionBids)

    if len(oldBids) != len(newBids):
        print("Reviewer lengths do not match.")
        return

    # Print the old bids and the new bids per paper for manual comparison
    # for oldPaperBids, newPaperBids in zip(oldBids, newBids):
    #     print("Old bids: ", oldPaperBids, "  New bids: ", newPaperBids)

    # Count stats
    successful = 0     # Number of trials where selected reviewer is in top 3 and other two aren't
    otherReviewer = 0  # Number of trials where selected reviewer is in top 3, but others also are there
    reviewerNotTop = 0 # Number of trials where selected reviewer is not in the top 3, but one or more of the other two are
    groupNotTop = 0    # Number of trials where none of the old top reviewers are in the top 3

    for paperIndex, submissionBids in enumerate(newBids):
        for experimentIndex, newPaperBids in enumerate(submissionBids):
            favoredReviewer = oldBids[paperIndex][experimentIndex][0]
            oldReviewers = list(map(lambda oldBid: oldBid[0], oldBids[paperIndex]))
            survivingReviewers = list(map(lambda bid: bid[0], filter(lambda newBid: newBid[0] in oldReviewers, newPaperBids)))
            #print("Old reviewers: ", oldReviewers)
            #print("Surviving reviewers: ", survivingReviewers)
            #print("Favored reviewer: ", favoredReviewer)

            if len(survivingReviewers) == 0:
                groupNotTop += 1
                #print("Group not top")
            elif favoredReviewer not in survivingReviewers:
                reviewerNotTop += 1
                #print("Reviewer not top")
            elif len(survivingReviewers) == 1:
                successful += 1
                #print("Successful")
            elif len(survivingReviewers) > 1:
                otherReviewer += 1
                #print("Other reviewer")
            #print("")

    print("")
    print("Total trials:       ", trials)
    print("Successful:         ", successful)
    print("Other reviewer:     ", otherReviewer)
    print("Reviewer not top 3: ", reviewerNotTop)
    print("Group not top 3:    ", groupNotTop)
    print("Avg. number of 100 bids: ", num100/trials)


def experiment5(allSubmissions, allReviewers, model, bidData, topicData):

    for i in trange(1, 11, desc="Word sizes"):
        totalTrials = 0
        trialsWithTopRfs = 0    # Number of trials where the reviewer is the top in the submission's list. Will be equal to total trials.
        trialsWithTopSfr = 0    # Number of trials where the submission is the in top 5 bids in the reviewer's list. May not be equal to total trials.
        trialsWithNewTopRfs = 0 # Number of trials where the reviewer is in the top 3 in the new submission's list.
        trialsWithNewTopSfr = 0 # Number of trials where the new submission is in the 5 in the reviewer's list.
        submissionRankSum = 0   # The sum of the ranks of the new submissions in their reviewer's lists.
        reviewerRankSum = 0     # The sum of the ranks of the reviewers in the new submissions' lists.
        newSize = 0             # The sum of the new submission sizes as a fraction of the old size

        a, b = bidData.normalized_bids.shape
        assert a == len(allReviewers)
        assert b == len(allSubmissions)

        for subIndex, submission in tqdm(enumerate(allSubmissions), total=len(allSubmissions), desc="Submissions"):
            totalTrials += 1
            
            # Find the original top reviewer for the submission
            maxBid = -90
            originalReviewerIndex = -1
            for revIndex, bid in enumerate(bidData.normalized_bids[:, subIndex]):
                if bid > maxBid:
                    maxBid = bid
                    originalReviewerIndex = revIndex
            originalReviewer = allReviewers[originalReviewerIndex]
            trialsWithTopRfs += 1
            
            # Find the original rank of the submission in the reviewer's list
            originalSubmissionRank = 1
            for bid in bidData.normalized_bids[originalReviewerIndex, :]:
                if bid > bidData.normalized_bids[originalReviewerIndex, subIndex]:
                    originalSubmissionRank += 1
            if originalSubmissionRank <= 5:
                trialsWithTopSfr += 1
            
            # Get word probabilits to avoid the reviewer and generate the new doc
            wordProbabilities = adversarialWords(originalReviewer, allReviewers, topicData, avoid=True)
            newDoc = words_from_probs(wordProbabilities, submission, size=i)

            # Get the new document size
            newSize += 1 + len(newDoc) / submission.num_words 

            # Generate the new bids
            newBids = bids_for_doc(model[model.id2word.doc2bow(newDoc + submission.words)], topicData, allReviewers)

            # Find the new rank of the reviewer in the submission's list
            # Normalize new bid using new min and max because we need to
            # compare across different reviewers
            newReviewerRank = 1
            normalizedReviewerBid = bidData.get_normalizer(
                min(bidData.rev_min_raw[originalReviewerIndex], newBids[originalReviewerIndex]),
                max(bidData.rev_max_raw[originalReviewerIndex], newBids[originalReviewerIndex])
            )(newBids[originalReviewerIndex])
            for revIndex, bid in enumerate(newBids):
                bid = bidData.get_normalizer(
                    min(bidData.rev_min_raw[revIndex], bid),
                    max(bidData.rev_max_raw[revIndex], bid)
                )(bid)
                if bid > normalizedReviewerBid:
                    newReviewerRank += 1

            # Find the new rank of the submission in the reviewer's list
            # Normalize new bid using old min and max because we just need to
            # compare with the old values of same reviewer
            newSubmissionRank = 1
            normalizedReviewerBid = bidData.get_normalizer(
                bidData.rev_min_raw[originalReviewerIndex], bidData.rev_max_raw[originalReviewerIndex]
            )(newBids[originalReviewerIndex])
            for sIndex, bid in enumerate(bidData.normalized_bids[originalReviewerIndex, :]):
                if sIndex != subIndex and bid > normalizedReviewerBid:
                    newSubmissionRank += 1

            # Record the stats for this trial
            submissionRankSum += newSubmissionRank
            reviewerRankSum += newReviewerRank
            if newReviewerRank <= 3:
                trialsWithNewTopRfs += 1
            if newSubmissionRank <= 5:
                trialsWithNewTopSfr += 1

        # Print the final stats for all experiments
        print()
        print("**** Experiment 5: Avoiding specific reviewers ****")
        print()
        print("# submissions: %d, # reviewers: %d" % (len(allSubmissions), len(allReviewers)))
        print("New Doc size: %.2f" % (newSize / totalTrials,))
        print("# trials: %d" % totalTrials)
        print("# trials with top reviewer for original submission: %d" % trialsWithTopRfs)
        print("# trials with top original submission for reviewer: %d" % trialsWithTopSfr)
        print("# trials with top reviewer for new submission: %d" % trialsWithNewTopRfs)
        print("# trials with top new submission for reviewer: %d" % trialsWithNewTopSfr)
        print("Avg. rank of reviewer in submission's list: %.2f" % (reviewerRankSum / totalTrials,))
        print("Avg. rank of submission in reviewer's list: %.2f" % (submissionRankSum / totalTrials,))
        print()

        # Write results to a tsv file
        with open("data/experiment5-originaltext.tsv", 'a') as f:
            f.write(f"%d\t%.2f\t%d\t%d\t%d\t%d\t%.2f\t%.2f\n" % (
                totalTrials, 
                newSize / totalTrials,
                trialsWithTopRfs,
                trialsWithTopSfr,
                trialsWithNewTopRfs,
                trialsWithNewTopSfr,
                reviewerRankSum / totalTrials,
                submissionRankSum / totalTrials
            ))


def get_ranks_with_maximal_diff(rev1, rev2, numWordsInSubmission, model, bidData, topicData):
    # Return (rank1, rank2) with maximal difference for (rev1, rev2) using an adversarial
    # submission with `numWordsInSubmission` words.

    # Sort the topics in order of the greatest probability difference
    probabilityDifferences = []
    topicList = topicData.rev_top[rev1.name()]
    for topicId, topicProbability in topicList:
        probabilityDifference = topicProbability - topicData.rev_top[rev2.name()].get(topicId, 0)
        probabilityDifferences.append( (topicId, probabilityDifference) )
    probabilityDifferences = sorted(probabilityDifferences, reverse=True, key=lambda topic: topic[1])

    # Get the top words from all topics with positive probability difference. IE: favor rev1.
    # Weight each word by its topic's probability difference
    wordProbabilities = {}
    for topicId, probabilityDifference in probabilityDifferences:
        if probabilityDifference <= 0:
            break
        topicWord = topicData.top_wds[topicId][0]
        wordProbabilities[topicWord] = max(wordProbabilities.get(topicWord, 0), probabilityDifference)

    # Normalize the word probabilities
    total = 0
    for word in wordProbabilities:
        total += wordProbabilities[word]
    for word in wordProbabilities:
        wordProbabilities[word] = wordProbabilities[word] / total

    # Use the word probabilities to generate a fake submission
    # Create bids for rev1 and rev2 on the fake submission
    newDoc = numbered_words_from_probs(wordProbabilities, numWordsInSubmission)
    return bids_for_doc(model[model.id2word.doc2bow(newDoc)], topicData, (rev1, rev2))


def get_similarity(rev1, rev2, m):
    v1 = np.array([ x[1] for x in m.get_document_topics(m.id2word.doc2bow(rev1.words), 0) ])
    v2 = np.array([ x[1] for x in m.get_document_topics(m.id2word.doc2bow(rev2.words), 0) ])
    return np.dot(v1, v2)/(norm(v1) * norm(v2))


def experiment6(allReviewers, numWordsInSubmission, model, bidData, topicData):
    reviewers = filter_reviewers(allReviewers, 5)
    for rev1 in tqdm(reviewers):
        for rev2 in tqdm(reviewers):
            similartiy = get_similarity(rev1, rev2, model)
            rev1rank, rev2rank = get_ranks_with_maximal_diff(rev1, rev2,
                    numWordsInSubmission, model, bidData, topicData)
            with open("data/experiment6.tsv", 'a') as f:
                f.write(f"%d\t%s\t%s\t%f\t%d\t%d\n" % (
                    numWordsInSubmission,
                    rev1,
                    rev2,
                    similarity,
                    rev1rank,
                    rev2rank
                ))


def main():
    # :(
    return 0
    parser = argparse.ArgumentParser(description='Attack Autobid')
    parser.add_argument('-c', '--cache', help="Directory storing the pickled data about reviewers, submissions, and LDA model", required=True)

    args = parser.parse_args()

    # Load all the cached data
    pc = common.PC()
    pc.load("%s/pc.dat" % args.cache)
    reviewers = filter_reviewers(list(pc.reviewers()), 5)
    submissions = list(bid.load_submissions(args.cache).values())
    m = bid.load_model(args.cache)
    td = TopicData.load(args.cache)
    bd = BidData.load(args.cache)
    # rev_word_prob = load_adv_word_probs(args.cache)
    # experiment1(submissions, reviewers, m, bd, td, rev_word_prob)
    # adversarialWords(reviewers[0], reviewers, td)
    experiment5(submissions, reviewers, m, bd, td)
    return 0

    n = 1000

    old_sub_rank_in_rev = np.zeros(n, dtype=int)
    old_rev_rank_in_sub = np.zeros(n, dtype=int)

    new_sub_rank_in_rev = np.zeros(n, dtype=int)
    new_rev_rank_in_sub = np.zeros(n, dtype=int)

    old_size = 0
    new_size = 0

    with open("%s/experiments_ava.tsv" % args.cache, 'a') as f:
        # for i in trange(n, desc="Trials"):
        for s_idx in trange(len(submissions), desc="Submissions"):
            sub = submissions[s_idx]
            for r_idx in trange(len(reviewers), desc="Reviewers"):
            # r_idx = np.random.randint(0, len(reviewers))
                rev = reviewers[r_idx]

            # sub = None
            # while (sub is None) or (sub.num_words == 0):
            #     s_idx = np.random.randint(0, len(submissions))
            #     sub = submissions[s_idx]

            # Generate new doc based on adversarial word probs for the reviewer
                new_doc = words_from_probs(rev_word_prob[rev.name()], sub)

                # old_size += sub.num_words
                # new_size += 1 + len(new_doc) / sub.num_words 

            # # Generate new bids for this updated submission
            # new_bids = bids_for_doc(m[m.id2word.doc2bow(new_doc + sub.words)],
            #         td, reviewers)

            # Generate new bids for this updated submission
                new_bids = bids_for_doc(m[m.id2word.doc2bow(new_doc)],
                        td, reviewers)


                # Find old rank of sub in rev's list
                osir = 1
                for b in bd.normalized_bids[r_idx, :]:
                    if b > bd.normalized_bids[r_idx, s_idx]:
                        osir += 1
                # old_sub_rank_in_rev[i] = rank

                # Find old rank of rev in sub's list
                oris = 1
                for b in bd.normalized_bids[:, s_idx]:
                    if b > bd.normalized_bids[r_idx, s_idx]:
                        oris += 1
                # old_rev_rank_in_sub[i] = rank

                # Find new rank of sub in rev's list
                sir = 1
                # Normalize new bid using old min and max because we just need to
                # compare with the old values of same reviewer
                normalized_new_bid = bd.get_normalizer(
                        bd.rev_min_raw[r_idx], bd.rev_max_raw[r_idx]
                    )(new_bids[r_idx])
                for si, b in enumerate(bd.normalized_bids[r_idx, :]):
                    if si != s_idx and b > normalized_new_bid:
                        sir += 1
                # new_sub_rank_in_rev[i] = rank

                # Find new rank of rev in sub's list
                ris = 1
                # Normalize new bid using new min and max because we need to
                # compare across different reviewers
                normalized_new_bid = bd.get_normalizer(
                        min(bd.rev_min_raw[r_idx], new_bids[r_idx]),
                        max(bd.rev_max_raw[r_idx], new_bids[r_idx])
                    )(new_bids[r_idx])
                for ri, b in enumerate(new_bids):
                    # Normalize new bid using new min and max because we need to
                    # compare across different reviewers
                    b = bd.get_normalizer(
                            min(bd.rev_min_raw[ri], b),
                            max(bd.rev_max_raw[ri], b)
                        )(b)
                    if b > normalized_new_bid:
                        ris += 1
                # new_rev_rank_in_sub[i] = rank
                f.write(f"%d\t%d\t%d\t%d\t%d\t%d\n" % (
                    s_idx, r_idx,
                    osir, sir,
                    oris, ris
                    ))

    # print()
    # print("# reviewers: %d, # submissions: %d" % (len(reviewers),
    #     len(submissions)))
    # print("# trials: %d" % n)
    # print("Avg. old size (# words): %.2f" % (old_size / n,))
    # print("Avg. new size: %.2fx" % (new_size / n,))
    # print("\nRank of submission in reviewer's list:")
    # print("---------------------------------------")
    # print("Stat\t\told\tnew")
    # print("Avg\t\t%.2f\t%.2f" % (np.mean(old_sub_rank_in_rev),
    #     np.mean(new_sub_rank_in_rev)))
    # print("Top 1\t\t%.2f%%\t%.2f%%" % (np.count_nonzero(old_sub_rank_in_rev == 1) * 100 / n,
    #     np.count_nonzero(new_sub_rank_in_rev == 1) * 100 / n))
    # print("Top 5\t\t%.2f%%\t%.2f%%" % (np.count_nonzero(old_sub_rank_in_rev <= 5) * 100 / n,
    #     np.count_nonzero(new_sub_rank_in_rev <= 5) * 100 / n))
    # print("\nRank of reviewer in submission's list:")
    # print("---------------------------------------")
    # print("Stat\t\told\tnew")
    # print("Avg\t\t%.2f\t%.2f" % (np.mean(old_rev_rank_in_sub),
    #     np.mean(new_rev_rank_in_sub)))
    # print("Top 1\t\t%.2f%%\t%.2f%%" % (np.count_nonzero(old_rev_rank_in_sub == 1) * 100 / n,
    #     np.count_nonzero(new_rev_rank_in_sub == 1) * 100 / n))
    # print("Top 3\t\t%.2f%%\t%.2f%%" % (np.count_nonzero(old_rev_rank_in_sub <= 3) * 100 / n,
    #     np.count_nonzero(new_rev_rank_in_sub <= 3) * 100 / n))

    # Write results to a tsv file
    # with open("%s/experiments.tsv" % args.cache, 'a') as f:
    #     f.write(f"%d\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n" % (
    #         n, new_size / n,
    #         np.count_nonzero(new_sub_rank_in_rev == 1) * 100 / n,
    #         np.count_nonzero(new_sub_rank_in_rev <= 5) * 100 / n,
    #         np.count_nonzero(new_rev_rank_in_sub == 1) * 100 / n,
    #         np.count_nonzero(new_rev_rank_in_sub <= 3) * 100 / n,
    #         ))

if __name__ == "__main__":
    main()
