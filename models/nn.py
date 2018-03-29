class NN():
    def __init__(self, weight_file):
        self.weight_file = weight_file
        # tb_cb = TensorBoard(log_dir="~/tflog/", histogram_freq=1)
        # self.cbks = [tb_cb

    def save_models(self, model):
        print("save" + self.weight_file)
        model.save(self.weight_file)

    def load_models(self):
        print("load" + self.weight_file)
        from keras.models import load_model
        return load_model(self.weight_file)
