# An "election" for features
# For voting methods, see: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1030908
from typing import Dict, List, Set
from abc import ABC, abstractmethod

import voter


class Election(ABC):
    def __init__(self, voters: List[voter.Voter]):
        self.voters = voters
        self.vote_queue = self.make_vote_queue()  # not really a queue

    def make_vote_queue(self) -> List[str]:
        # use the voters' legal features to have a list of features in the
        # voting queue
        seen_fts = set()

        for v in self.voters:
            seen_fts.update(v.legal_fts)

        return list(seen_fts)

    @abstractmethod
    def decision_rule(self, token: str, ft: str) -> int:
        # decide if a given token has a feature
        # outputs: -1 (abstain), 0 (no feature), 1 (has feature)
        pass

    def run_election(self, tgt_token) -> List[str]:
        # given some target token, output a list of feature according
        # to the voting methods

        pred_fts: Set[str] = set()
        for ft in self.vote_queue:
            v = self.decision_rule(tgt_token, ft)
            if v == 1:
                pred_fts.add(ft)

        return list(pred_fts)


class Plurality(Election):
    # a plurality election gives every voter one vote
    # the candidate with the highest number of votes wins

    def decision_rule(self, token: str, ft: str) -> int:

        ballot = {-1: 0, 0: 0, 1: 0}
        for voter in self.voters:
            vote = voter.vote(token, ft)
            ballot[vote] += 1

        # abstrain doesn't actually matter here
        if ballot[0] > ballot[1]:
            return 0
        elif ballot[1] > ballot[0]:
            return 1
        else:
            # we have a tie...
            # assume we have don't have the feature
            return 0


class Majority(Election):
    # majority election only accepts the voter with the highest vote
    # if no class has a majority (e.g., too many voters abstrain), abstain

    def decision_rule(self, token: str, ft: str) -> int:

        ballot = {-1: 0, 0: 0, 1: 0}
        for voter in self.voters:
            vote = voter.vote(token, ft)
            ballot[vote] += 1

        total = ballot[-1] + ballot[0] + ballot[1]
        half = total / 2

        if ballot[0] > half:
            return 0
        elif ballot[1] > half:
            return 1
        else:
            # abstrain
            return -1


class WeightedElection(Election):

    # basis class for using language distance as a weight
    # convert each src iso's vote to a weight according to dictionary
    # entry

    # more of a base class for confidence voting methods (see 2.3 in above paper)

    def __init__(self, voters: List[voter.Voter], src_iso_weights: Dict[str, float]):
        self.voters = voters
        self.vote_queue = self.make_vote_queue()  # not really a queue
        self.weights = src_iso_weights


class SumRule(WeightedElection):
    # Each voter gives confidence value for each candidate (yes/no)
    # sum together all confidence values for each candidate and
    # take the highest that wins the election

    def decision_rule(self, token: str, ft: str) -> int:
        ballot = {-1: 0, 0: 0, 1: 0}

        for voter in self.voters:
            vote = voter.vote(token, ft)
            confidence = self.weights[voter.src_iso]
            other_confidence = 1 - confidence
            ballot[vote] += confidence
            if vote == 0:
                other = 1
            elif vote == 1:
                other = 0
            else:
                # abstain
                continue
            ballot[other] += other_confidence

        if ballot[0] > ballot[1]:
            return 0
        elif ballot[1] > ballot[0]:
            return 1
        else:
            # abstain
            return -1


class ProductRule(WeightedElection):
    # Each voter gives confidence value for each candidate (yes/no)
    # sum together all confidence values for each candidate and
    # take the highest that wins the election

    def decision_rule(self, token: str, ft: str) -> int:
        ballot = {-1: 0, 0: 0, 1: 0}

        for voter in self.voters:
            vote = voter.vote(token, ft)
            confidence = self.weights[voter.src_iso]
            other_confidence = 1 - confidence
            ballot[vote] *= confidence
            if vote == 0:
                other = 1
            elif vote == 1:
                other = 0
            else:
                # abstain
                continue
            ballot[other] *= other_confidence

        if ballot[0] > ballot[1]:
            return 0
        elif ballot[1] > ballot[0]:
            return 1
        else:
            # abstain
            return -1


class UnionVote(Election):
    # take each vote and keep it if a single one has the feature

    def decision_rule(self, token: str, ft: str) -> int:
        ballot = {0: 0, -1: 0}
        for voter in self.voters:
            vote = voter.vote(token, ft)
            if vote == 1:
                return 1
            ballot[vote] += 1

        if ballot[0] > ballot[-1]:
            return 0
        return -1


class ThresholdVote(Election):
    # take each vote and keep it if a single one has the feature
    def __init__(self, voters: List[voter.Voter], threshold=2):
        super().__init__(voters)
        self.threshold = threshold

    def decision_rule(self, token: str, ft: str) -> int:
        ballot = {0: 0, 1: 0, -1: 0}
        for voter in self.voters:
            vote = voter.vote(token, ft)
            ballot[vote] += 1

        if ballot[1] > self.threshold:
            return 1
        else:
            if ballot[0] > ballot[-1]:
                return 0
        return -1


class IntersectVote(Election):
    # take each vote and keep it if a single one has the feature

    def decision_rule(self, token: str, ft: str) -> int:

        ballot = {1: 0, -1: 0}
        for voter in self.voters:
            vote = voter.vote(token, ft)
            if vote == 0:
                return 0
            ballot[vote] += 1

        if ballot[1] > ballot[-1]:
            return 1
        return -1
