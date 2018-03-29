from keras.layers import Input, Dense, merge
from keras.models import Model


class Test():
    def __init__(self):
        encoder = self.test_build()
        encoder.summary()
        decoder = self.test_build()
        decoder.summary()
        ei1, ei2 = encoder.input
        di1, di2 = decoder.output
        autoencoder = Model([ei1, ei2], [di1, di2])
        autoencoder.summary()

    def test_build(self):
        input_layer1 = Input(shape=(None, 10))
        input_layer2 = Input(shape=(None, 10))
        merged_layer = merge([input_layer1, input_layer2])
        output_layer = Dense(20)(merged_layer)
        output_layer1 = Dense(10)(output_layer)
        output_layer2 = Dense(10)(output_layer)
        return Model([input_layer1, input_layer2], [output_layer1, output_layer2])

    # def model_flow(self, model, input_layer):
    #     output_layer = model.layers[1](input_layer)
    #     for layer in model.layers[2:]:
    #         output_layer = layer(output_layer)
    #     return output_layer

    def test_rebuild(self, model, model2):
        pass
        # output_layer = model.get_layer(
        #     index=len(model.layers))(self.input_layer)
        # output_layer = model2.get_layer(index=len(model2.layers))(output_layer)
        # output_layer = self.model_flow(model, self.input_layer)
        # output_layer = self.model_flow(model2, output_layer)
        # return Model(self.input_layer, output_layer)


def main():
    test = Test()


if __name__ == "__main__":
    main()
