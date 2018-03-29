import sys
import MeCab

import sys
import MeCab


class Wakachi():
    def __init__(self):
        self.tagger = MeCab.Tagger(
            ' -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')
        self.tagger.parse('')  # これ重要！！！！


class WakachiMethod():
    def __init__(self, w):
        self.w = w()
        self.get_part_list = ["名詞", "動詞", "接続詞", "形容詞", "固有名詞", "副詞", "連体詞"]
        self.stop_word = ["BOS/EOS", "助詞", "助動詞", "記号"]

    def remove_stopword(self, wordline):
        node = self.w.tagger.parseToNode(wordline)
        words = []
        while (node):
            part = node.feature.split(",")[0]
            if part in self.get_part_list:
                words.append(node.surface)
            # nodeを次に送る
            node = node.next
        return words

    def get_default_word_remove_stopword(self, wordline):
        node = self.w.tagger.parseToNode(wordline)
        words = []
        while (node):
            part = node.feature.split(",")[0]
            if part not in self.stop_word:
                # 7番目は単語の原形
                default = node.feature.split(",")[6]
                if default == "*":
                    words.append(node.surface)
                else:
                    words.append(default)
            node = node.next
        return words

    def split_word(self, wordline):
        node = self.w.tagger.parseToNode(wordline)
        words_list = []
        while (node):
            if node.feature.split(",")[0] != 'BOS/EOS':
                words_list.append(node.surface)
            node = node.next
        return words_list


if __name__ == "__main__":
    text = u'午後1時30分。１人で遊ぶ。'
    text = u'私は友達と、美しい公園を歩いたり走ったりした。'
    text = '見ると、名前の上に朱線も引かれていなければ、上欄には隠居届を受附けた旨記載してあるばかりで、死亡の死の字も見えないのであった。'
    text = 'いつも ながら その 部屋 は 、 私 を 、 丁度 とほう も なく 大きな 生物 の 心臓 の 中 に 坐っ て でも いる 様 な 気持 に し た 。'
    wakachi = WakachiMethod(Wakachi)
    t = wakachi.get_default_word_remove_stopword(text)
    t = wakachi.split_word(text)
    print(t)
    # wakachi = Wakachi(split_word)
    # get_words(text)
    # print(get_words(text))
