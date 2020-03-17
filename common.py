#!/usr/bin/python

import os
#from urlparse import urlparse
import csv
import pickle
from collections import Counter

class Reviewer:
    def __init__(self, gs_id):
        self.gs_id = gs_id
        self.num_words = 0
        self.pdf_names = set()
        self.feature_vector = None
        self.words = []
        #self.html = None
        self.status = "PDFs"
        #self.sql_id = None

        if len(self.gs_id) == 0:
            print("\n\nWarning!  Missing information for %s\n\n" % self)
    
    def name(self):
        return "%s" % (self.gs_id)

    def dir(self):
        return 'reviewer_papers/' + self.gs_id + '/'

    def s3_filename(self, citation_id):
        return 'reviewer_papers/' + self.gs_id + '/' + self.gs_id + ':' + citation_id + '.pdf'

    def __str__(self):
        return self.name()

    def __eq__(self, other):
        return isinstance(other, Reviewer) and \
               self.gs_id == other.gs_id

    def __hash__(self):
        return hash("%s" % self.name())

    def make_pdf_weights(self):
        weights = {}
        for index, name in enumerate(self.pdf_names):
            weights[name] = (len(self.pdf_names) - index) / float(len(self.pdf_names))
            if len(self.pdf_names) < 15:  # If someone only has a small number of pubs, don't apply gradient weights
                weights[name] = 1
        return weights

    def display_status(self):
        print(self.gs_id, self.words)

    def set_status(self, status):
        self.status = status
        if status == "Init" or status == "HTML":
            self.pdf_names = set()

        if status == "Init" or status == "HTML" or status == "PDFs":
            self.num_words = 0
            self.feature_vector = None
            self.words = []


class PDF:
    def __init__(self, index, url):
        self.index = int(index)
        self.url = url

    def __str__(self):
        return "%d %s" % (self.index, self.url)

    def __eq__(self, other):
        return isinstance(other, PDF) and \
               self.url == other.url 

    def __hash__(self):
        return hash("%s" % self.url)


class Submission:
    def __init__(self, name):
        self.name = name
        self.num_words = 0
        self.feature_vector = Counter()
        self.words = []


class PC:
    def __init__(self):
        self.__reviewers = {}

    def reviewers(self):
        return self.__reviewers.values()

    def names(self):
        return self.__reviewers.keys()

    def count(self):
        return len(self.__reviewers)

    def reviewer(self, reviewer_name):
        return self.__reviewers[reviewer_name]

    def remove_reviewer(self, reviewer_name):
        if reviewer_name in self.__reviewers:
            del self.__reviewers[reviewer_name]
            return True
        else:
            print("Sorry, %s is not recognized as a current PC member" % reviewer_name)
            return False

    def save(self, filename):
        with open(filename, "wb") as pickler:
            pickle.dump(self.__reviewers, pickler)

    def load(self, filename):
        if os.path.exists(filename):
            print("Loading reviewer information...")
            with open(filename, "rb") as pickler:
                self.__reviewers = pickle.load(pickler)
            print("Loading reviewer information complete!")
        else:
            print("Unable to find file of saved reviewers info: %s" % filename)

    def status(self):
        max_width = 0
        for reviewer_name in self.names():
            if len(reviewer_name) > max_width:
                max_width = len(reviewer_name)

        for reviewer_name in sorted(self.names()):
            self.__reviewers[reviewer_name].display_status(max_width)

    def set_status(self, status):
        for reviewer in self.reviewers():
            reviewer.set_status(status)

    def parse_csv(self, csv_file_name):
        with open(csv_file_name, 'rb') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                if row['Do you agree to be a member of the IEEE S&P 2018 program committee?'] == "Yes":
                    reviewer = Reviewer(row['First Name'],
                                        row['Last Name'],
                                        row['E-Mail Address'],
                                        row['Link to publications web page'])
                    if reviewer.name() in self.names():
                        existing_reviewer = self.reviewer(reviewer.name())
                        if not existing_reviewer == reviewer:
                            # Update the reviewer and mark for reprocessing
                            self.__reviewers[reviewer.name()] = reviewer
                            print("Reviewer %s's info has been updated from\n\t%s\nto\n\t%s\n" % \
                                (reviewer.name(), existing_reviewer, reviewer))
                            reviewer.status = "Init"
                    else:
                        print("Found a new reviewer: %s" % reviewer.name())
                        self.__reviewers[reviewer.name()] = reviewer

    def parse_citations(self, citations_file_name):
        with open(citations_file_name, 'r') as citations_file:
            for line in citations_file:
                tokens = line.split(':')
                gs_id = tokens[0].strip()
                citation_id = tokens[1].strip()
                reviewer = self.__reviewers.get(gs_id, None)

                # Add the reviewer if it doesn't already exist
                if reviewer == None:
                    print('Creating new reviewer:', gs_id)
                    reviewer = Reviewer(gs_id)
                    self.__reviewers[gs_id] = reviewer

                # Add the citation id to the list of PDFs
                print('Adding citation', citation_id, 'to reviewer', gs_id)
                reviewer.pdf_names.add(citation_id)

    def __match_based_on_first_name(self, first):
        matches = []
        for reviewer in self.reviewers():
            if reviewer.first == first:
                matches.append(reviewer)

        if len(matches) == 1:
            return matches[0]
        elif len(matches) == 0:
            #print "No matches for %s based on first name!" % reviewer.name()
            return None
        else:
            #print "Too many matches for %s based on first name!" % reviewer.name()
            return None
        
    def __match_based_on_last_name(self, last):
        matches = []
        for reviewer in self.reviewers():
            if reviewer.last == last:
                matches.append(reviewer)

        if len(matches) == 1:
            return matches[0]
        elif len(matches) == 0:
            #print "No matches for %s based on last name!" % reviewer.name()
            return None
        else:
            #print "Too many matches for %s based on last name!" % reviewer.name()
            return None

    def assign_sql_ids(self, pc_ids_file):
        print("Matching reviewers to SQL IDs...")

        with open(pc_ids_file, 'rb') as csv_file:
            reader = csv.DictReader(csv_file, delimiter="\t")
            for row in reader:
                first = row['firstName']
                last = row['lastName']
                id = row['contactId']

                name = "%s %s" % (first, last)
                if name in self.__reviewers:
                    self.reviewer(name).sql_id = id
                else:
                    match = self.__match_based_on_last_name(last)
                    if not match == None:
                        match.sql_id = id
                    else:
                        match = self.__match_based_on_first_name(first)
                        if not match == None:
                            match.sql_id = id
                        else: 
                            print("\nWARNING: Couldn't find a reviewer with name: %s %s!\n" % (first, last))
        print("Matching reviewers to SQL IDs complete!")

    def print_words(self):
        for reviewer in self.reviewers():
            reviewer.display_status()



#def main():
#    parser = argparse.ArgumentParser(description='Fetch PC member papers and analyze them')
#    parser.add_argument('--csv', action='store', required=False,
#        help='CSV file containing author <first name, last name, email, ID> from HotCRP')
#    parser.add_argument('-c', '--cache', help="Use the specified file for caching reviewer status and information", required=False)
#    parser.add_argument('--html', action='store_true', default=False, help="Only fetch and analyze HTML pages", required=False)
#    parser.add_argument('--submissions', action='store', help="Directory of submissions", required=False)
#    parser.add_argument('--submissionsc', action='store', help="Directory of submissions", required=False)
#    parser.add_argument('--reviewersc', action='store_true', default=False, help="Clean reviewer corpus", required=False)
#    parser.add_argument('--pc', action='store_true', help="Display PC status", required=False)
#    parser.add_argument('--reviewer', action='store', help="Display status of one reviewer", required=False)
#    parser.add_argument('--status', action='store', help="Update reviewer specified with --reviewer to the provided status", required=False)
#    parser.add_argument('--pcstatus', action='store', help="Set entire PC's status", required=False)
#    parser.add_argument('--bid', action='store', help="Calculate bids for one reviewer", required=False)
#    parser.add_argument('--bids', action='store_true', help="Calculate bids for the entire PC", required=False)
#    parser.add_argument('--bidmethod', action='store', default="default", help="Method used when calculating bids", required=False)
#    parser.add_argument('--words', action='store_true', default=False, help="(Re)calculate number of words for each reviewer", required=False)
#    parser.add_argument('-j', action='store', help="Number of processes to use", required=False)
#    parser.add_argument('--feature', action='store', help="Pass a submission number or reviewer name to display their feature vector", required=False)
#    parser.add_argument('--top_k', action='store', help="Restrict feature vector printing to top k features", required=False)
#    parser.add_argument('--pcids', action='store', help="File containing PC IDs in the MySQL db", required=False)
#    parser.add_argument('--realprefs', action='store', 
#        help="File containing real preferences from the MySQL db", required=False)
#    parser.add_argument('--s1', action='store', help="First submission to compare a reviewer's calculated bid", required=False)
#    parser.add_argument('--s2', action='store', help="Second submission to compare a reviewer's calculated bid", required=False)
#    parser.add_argument('--b2017', action='store_true', default=False, help="Load 2017 bids", required=False)
#    parser.add_argument('--lda', action='store', help="Directory of old submissions to build an LDA model for", required=False)
#    parser.add_argument('--ldabids', action='store_true', default=False, help="Calculate bids using LDA", required=False)
#    
#    args = parser.parse_args()
#
#    reviewers = {}
#
#    # Pull in previous data, if it exists
#    if not (args.cache == None) and os.path.exists(args.cache):
#        print "Loading reviewer information..."
#        with open(args.cache, "rb") as pickler:
#            reviewers = pickle.load(pickler)
#        print "Loading reviewer information complete!"
#
#    if args.words and not (args.cache == None):
#        calculate_reviewer_words(reviewers)
#        pickle_reviewers(args.cache, reviewers)
#        sys.exit(0)
#
#    if args.pc:
#        display_reviewers_status(reviewers)
#        sys.exit(0)
#
#    if args.b2017:
#        load_2017_prefs(reviewers)
#        sys.exit()
#
#    if not args.reviewer == None:
#        if args.status == None:
#            display_reviewer_status(reviewers[args.reviewer])
#        else:
#            set_reviwer_status(reviewers[args.reviewer], args.status)
#            pickle_reviewers(args.cache, reviewers)
#        sys.exit(0)
#
#    if not args.pcstatus == None:
#        for reviewer in reviewers.values():
#            set_reviwer_status(reviewer, args.pcstatus)
#        pickle_reviewers(args.cache, reviewers)
#        sys.exit(0)
#
#    submissions = None
#    if not args.submissions == None:
#        pickle_file = "%s/submissions.dat" % args.submissions
#        if not os.path.isfile(pickle_file):
#            submissions = analyze_submissions(args.submissions, args.j)
#            with open(pickle_file, "wb") as pickler:
#                pickle.dump(submissions, pickler)
#        else:
#            with open(pickle_file, "rb") as pickler:
#                submissions = pickle.load(pickler)
#
#    if args.ldabids: 
#        for reviewer in reviewers.values():
#            create_reviewer_bid(reviewer, None, submissions, None, "lda", args.submissions, args.lda)
#        sys.exit(0)
#
#    if not args.lda == None:
#        build_lda_model(args.lda, args.j)
#        sys.exit(0)
#
#    if not args.submissionsc == None:
#        make_clean_submissions_corpus(args.submissionsc, args.j)
#        sys.exit(0)
#
#    if args.reviewersc:
#        make_clean_reviewers_corpus(reviewers)
#        sys.exit(0)
#
#    if not (args.s1 == None or args.s2 == None) and (is_digit(args.s1) and is_digit(args.s2)):
#        corpus = build_corpus(reviewers)
#        compare_submission_bids(reviewers[args.bid], corpus, submissions[int(args.s1)], submissions[int(args.s2)], args.bidmethod)
#        sys.exit(0)
#
#    if not args.feature == None:
#        display_feature_vector(args.feature, args.top_k, reviewers, submissions)
#        sys.exit(0)
#
#    if not args.csv == None:
#        # Update based on csv file
#        parse_reviewers(args.csv, reviewers)
#
#    if not args.realprefs == None:
#        if not args.pcids:
#            print "Mapping reviewer preferences to individual bids requires PC IDs.  Use --pcids filename.csv"
#        else:
#            dump_real_prefs(args.realprefs, args.pcids, reviewers)
#
#    if args.bid == None and not args.bids:
#        try:
#            process_reviewers(reviewers, args.html)
#            analyze_reviewers_papers(reviewers, args.j)
#            if not args.cache == None:
#                pickle_reviewers(args.cache, reviewers)
#        except:
#            print "\nUnexpected Error!\n%s" % traceback.format_exc()
#            if not args.cache == None:
#                pickle_reviewers(args.cache, reviewers)
#    else:
#        corpus = build_corpus(reviewers)
#        id_mapping = None
#        if not args.pcids == None:
#            id_mapping = match_reviewers_to_ids(reviewers, args.pcids)
#        if not args.bid == None:
#            create_reviewer_bid(reviewers[args.bid], corpus, submissions, id_mapping, args.bidmethod)
#        elif args.bids:
#            for reviewer in reviewers.values():
#                create_reviewer_bid(reviewer, corpus, submissions, id_mapping, args.bidmethod)
#
#
#if (__name__=="__main__"):
#  main()
#
