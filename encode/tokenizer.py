import string


class Tokenizer:
    def __init__(self):
        self.vocabulary_encoding = None
        self.grammar_encoding = None
        self.inverse_grammar_encoding = None

    def encode_sentence(self, sentence):
        encoding = []
        for token in sentence:
            print(token)
            token = token.strip(string.punctuation)
            token = token.lower()
            encoding.append(self.vocabulary_encoding.get(token, 0))

        print(encoding)
        return encoding

    def encode(self, token):
        token = token.strip(string.punctuation)
        token = token.lower()
        return self.vocabulary_encoding.get(token, 0)

    def encode_pos(self, pos):
        return self.grammar_encoding.get(pos, 0)

    def decode_pos(self, encoding):
        return self.inverse_grammar_encoding[encoding]

