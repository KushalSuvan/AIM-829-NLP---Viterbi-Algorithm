from typing import Iterable
import numpy as np

class Viterbi:
    def __init__(self, initial_prob, transition_mat, emission_mat):
        '''

        :param initial_prob: Initial probability of each PoS
        :param transition_mat: Transition probability matrix
        :param emission_mat: Emission probability matrix
        '''

        self.transition_mat = transition_mat
        self.emission_mat = emission_mat
        self.initial_prob = initial_prob

        self.number_of_pos = transition_mat.shape[0]
        self.number_of_tok = emission_mat.shape[1]

    def tag_pos(self, tokens):
        l = len(tokens)

        probabilities = np.zeros((l, self.number_of_pos))
        probabilities[0] = self.initial_prob * self.emission_mat[tokens[0]]

        backpointer = np.zeros(l)

        for t in range(1, l):
            for pos in range(self.number_of_pos):
                probs = probabilities[t-1] * self.transition_mat[pos] * self.emission_mat[tokens[t]]
                probabilities[t][pos] = np.max(probs)
                backpointer[t] = np.argmax(probs)

        pos_tags = np.zeros(l)
        for t in range(1, l):
            pos_tags[t-1] = backpointer[t]

        return pos_tags