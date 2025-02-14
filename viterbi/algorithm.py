from typing import Iterable
import numpy as np
from encode import Tokenizer


class Viterbi:
    def __init__(self, initial_prob, transition_mat, emission_mat, tokenizer:Tokenizer):
        """

        :param initial_prob: Initial probability of each PoS
        :param transition_mat: Transition probability matrix
        :param emission_mat: Emission probability matrix
        """

        self.transition_mat = transition_mat
        self.emission_mat = emission_mat
        self.initial_prob = initial_prob
        self.tokenizer = tokenizer

        self.number_of_pos = transition_mat.shape[0]
        self.number_of_tok = emission_mat.shape[1]

    def tag_pos(self, tokens):
        l = len(tokens)

        probabilities = np.zeros((l, self.number_of_pos))
        probabilities[0] = self.initial_prob * self.emission_mat[tokens[0]]

        backpointer= np.zeros((l, self.number_of_pos))
        for i in range(self.number_of_pos):
            backpointer[0][i] = i

        for t in range(1, l):
            for pos in range(self.number_of_pos):
                probs = probabilities[t-1] * self.transition_mat[pos] * self.emission_mat[tokens[t]][pos]
                probabilities[t][pos] = np.max(probs)
                backpointer[t][pos] = np.argmax(probs)


        pos_tags = np.zeros(l)
        pos_tags[l-1] = np.argmax(probabilities[l-1])

        for t in range(l-2, -1, -1):
            pos_tags[t] = backpointer[t+1][int(pos_tags[t+1])]

        pos_seq = []
        for tag in pos_tags:
            pos_seq.append(self.tokenizer.decode_pos(tag))

        return pos_seq
