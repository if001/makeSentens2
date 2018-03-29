from models.Seq2Seq import Seq2Seq
from lib.DataShaping import DataShaping
from lib.StringOperation import StringOperation
from aozora_text.remove_stopword import Wakachi
from aozora_text.remove_stopword import WakachiMethod
import random as rand
import numpy as np


def get_word_lists(file_path):
    print("make wordlists")
    # lines = open(file_path).read().split("。")
    lines = open(file_path).read().split("\n")
    wordlists = []
    for line in lines:
        wordlists.append(line.split(" "))

    print("wordlist num:", len(wordlists))
    return wordlists[:-1]


def main():
    # train data作成
    word_list = get_word_lists("./aozora_text/files/files_all_tmp.txt")
    # stop_word_list = get_word_lists(
    #     "./aozora_text/files/stop_files_all_tmp.txt")
    ds = DataShaping()
    seq2seq = Seq2Seq("make")

    st = StringOperation()
    start_token = np.array([st.sentens_array_to_vec(["BOS"])])
    sentens = word_list[rand.randint(0, len(word_list) - 1)][1:]
    while('' in sentens):
        sentens.remove('')
    sentens = str(sentens)
    w = WakachiMethod(Wakachi)

    for _ in range(3):
        print("sentens:", sentens)
        sentens_rm_stop_word = w.remove_stopword(sentens)
        print("rm stop word sentens", sentens_rm_stop_word)
        sentens_vec = np.array([st.sentens_array_to_vec(sentens_rm_stop_word)])
        sentens_vec = seq2seq.make_sentens_vec(sentens_vec, start_token)
        sentens_vec = np.array(sentens_vec).reshape(len(sentens_vec), 128)
        sentens_arr = st.sentens_vec_to_sentens_arr_prob(sentens_vec)
        sentens = str(sentens_arr)
        print(sentens)


if __name__ == "__main__":
    main()
