# from lib import Const
from lib import WordVec as wv

fname = "./aozora_text/files/concat_file_all_rnp.txt"
model = wv.MyWord2Vec().train(fname, "save")

print("corpus: ", model.corpus_count)
voc = model.wv.vocab.keys()


# vec = wv.MyWord2Vec().str_to_vector(model, "冷遇")
# print(vec)
# import pylab as plt
# plt.plot(vec)
# plt.show()
