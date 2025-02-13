from viterbi import Viterbi
from encode import Tokenizer
import pickle


def main():
    with open('initial.pkl', 'rb') as handle:
        initial = pickle.load(handle)

    with open('transition.pkl', 'rb') as handle:
        transition = pickle.load(handle)

    with open('emission.pkl', 'rb') as handle:
        emission = pickle.load(handle)

    with open('tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)

    print(tokenizer.encode('God'))

    model = Viterbi(initial, transition, emission)
    model.tag_pos(tokenizer.encode_sentence(["Hello", "God"]))

if __name__ == "__main__":
    main()