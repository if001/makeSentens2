import os
from remove_stopword import WakachiMethod
from remove_stopword import Wakachi
import re

WORK_DIR = "./"


def fopen(fname):
    lines = []
    with open(fname, "r") as f:
        for l in f.readlines():
            lines.append(l.split(" ")[:-1])
    return lines


def get_file_list(path):
    file_list = []
    for f in os.listdir(path=path):
        if re.search('reshape_mecab_utf8_.*.txt', f) != None:
            file_list.append(f)
    return file_list


def main():
    # lines = fopen("./files/files_all_rnp.txt")
    wakachi = WakachiMethod(Wakachi)
    file_list = get_file_list(WORK_DIR)
    for f in file_list:
        print("open : stop_" + f)
        write_file = open(WORK_DIR + "stop_" + f, "a")
        with open(WORK_DIR + f) as read_file:
            for line in read_file.readlines():
                if len(line) > 2:
                    d = wakachi.get_default_word_remove_stopword(line)
                    write_file.write(" ".join(d + ["\n"]))

        write_file.close()


if __name__ == "__main__":
    main()
