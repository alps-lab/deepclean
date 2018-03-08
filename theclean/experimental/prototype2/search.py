from six.moves import xrange

from collections import deque, namedtuple
from itertools import product
import heapq
from copy import deepcopy


class State(object):

    def __init__(self, components, masks, score, step=0, external=False):
        self.components = components
        self.masks = masks
        self.score = score
        self.checked = False
        self.external = external

        self.step = 0

    def __lt__(self, other):
        return isinstance(other, State) and (self.score < other.score or (self.score == other.score
                                    and self.external))

    def __gt__(self, other):
        return isinstance(other, State) and (self.score > other.score or
                                             (self.score == other.score and not self.external))

    def __eq__(self, other):
        return (isinstance(other, State) and self.components == other.components
            and self.external == other.external)

    def __hash__(self):
        h = 432452343
        for comp in self.components:
            h = h ^ hash(comp) ^ hash(self.external)
        return h

    def __str__(self):
        return str(self.components)


class BeamSearch(object):

    def __init__(self, num_tokens, num_variants):
        self.k = -1

        self.num_tokens = num_tokens
        self.num_variants = num_variants
        self.candidate_states = [State([], set(), 0.0)]

    def generate_next_candidates(self):
        new_states = []

        for state in self.candidate_states:
            if state.checked:
                continue
            state.checked = True
            for i in xrange(max({-1} | (state.masks - {state.components[0][0]})) + 1, self.num_tokens):
                if i in state.masks:
                    continue

                new_state = deepcopy(state)
                new_state.checked = False
                new_state.components.append((i, 0))
                new_state.masks.add(i)

                new_states.append(new_state)

        return new_states

    def set_k(self, k):
        self.k = k
        if self.k >= 0:
            while len(self.candidate_states) > self.k:
                heapq.heappop(self.candidate_states)

    def update_candidate_scores(self, new_candidate_states):
        for candidate_state in new_candidate_states:
            heapq.heappush(self.candidate_states, candidate_state)

        if self.k >= 0:
            while len(self.candidate_states) > self.k:
                heapq.heappop(self.candidate_states)
