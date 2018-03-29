from models.Seq2Seq import Seq2Seq
from lib.DataShaping import DataShaping


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
    stop_word_list = get_word_lists(
        "./aozora_text/files/stop_files_all_tmp.txt")
    ds = DataShaping()
    seq2seq = Seq2Seq("train")

    for i in range(10):
        train = ds.make_data_train(stop_word_list, i)
        teach, target = ds.make_data_teach_target(word_list, i)
        seq2seq.train(train, teach, target)
        seq2seq.save_model()


if __name__ == "__main__":
    main()
