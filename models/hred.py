import numpy as np
import matplotlib.pylab as plt


from keras.layers.wrappers import Bidirectional as Bi
from keras.layers import Lambda, Input, Dense, GRU, LSTM, concatenate, Dropout, Bidirectional
from keras import backend as K
from keras.models import Model
from keras.optimizers import Adam, SGD, RMSprop

from keras.engine.topology import Container


class NNVAL():
    input_dim = 10
    output_dim = 10
    encoder_latent_dim = 5
    decoder_latent_dim = 5


class HRED():
    def __init__(self):
        self.encoder = self.build_encoder()
        self.encoder.summary()

        self.decoder = self.build_decoder()
        self.decoder.summary()

        self.autoencoder = self.build_autoencoder()
        self.autoencoder.summary()

    def build_encoder(self):
        K.set_learning_phase(1)
        encoder_input = Input(shape=(None, NNVAL.input_dim))
        encoder_dense_outputs = Dense(
            NNVAL.encoder_latent_dim, activation='sigmoid')(encoder_input)
        encoder_bi_lstm = LSTM(
            NNVAL.encoder_latent_dim, return_sequences=True, dropout=0.4, recurrent_dropout=0.4)
        encoder_bi_outputs = Bi(encoder_bi_lstm)(encoder_dense_outputs)
        _, state_h, state_c = LSTM(NNVAL.encoder_latent_dim, return_state=True,
                                   dropout=0.2, recurrent_dropout=0.2)(encoder_bi_outputs)
        return Container(encoder_input, [state_h, state_c])

    def build_decoder(self):
        K.set_learning_phase(1)
        decoder_input = Input(shape=(None, NNVAL.input_dim))
        state_h = Input(shape=(NNVAL.encoder_latent_dim,))
        state_c = Input(shape=(NNVAL.encoder_latent_dim,))

        decoder_dense_outputs = Dense(
            NNVAL.decoder_latent_dim, activation='sigmoid')(decoder_input)
        decoder_bi_lstm = LSTM(
            NNVAL.decoder_latent_dim, return_sequences=True, dropout=0.6, recurrent_dropout=0.6)
        decoder_bi_outputs = Bi(decoder_bi_lstm)(decoder_dense_outputs)
        decoder_lstm = LSTM(NNVAL.decoder_latent_dim, return_sequences=True,
                            return_state=True, dropout=0.4, recurrent_dropout=0.4)

        encoder_states = [state_h, state_c]
        decoder_output, output_h, output_c = decoder_lstm(
            decoder_bi_outputs, initial_state=encoder_states)

        decoder_output = Dense(
            NNVAL.decoder_latent_dim, activation='tanh')(decoder_output)
        decoder_output = Dropout(0.2)(decoder_output)
        decoder_output = Dense(
            NNVAL.output_dim, activation='linear')(decoder_output)

        return Container([decoder_input, state_h, state_c], [
            decoder_output, output_h, output_c])

    def build_autoencoder(self):
        encoder_input = Input(shape=(None, NNVAL.input_dim))
        state_h, state_c = self.encoder(encoder_input)
        decoder_input = Input(shape=(None, NNVAL.input_dim))
        decoder_output = self.decoder([decoder_input, state_h, state_c])
        return Model([encoder_input, decoder_input], decoder_output)

    # def model_compile(self, model):
    #     """ complie """
    #     optimizer = RMSprop(lr=0.001, rho=0.7, epsilon=1e-08, decay=0.0)
    #     loss = 'mean_squared_error'
    #     # loss = 'kullback_leibler_divergence'
    #     model.compile(optimizer=optimizer,
    #                   loss=loss,
    #                   metrics=['accuracy'])
    #     model.summary()
    #     return model

    # def train_autoencoder(self, model, encoder_input_data, decoder_input_data, decoder_target_data, meta_hh, meta_hc, meta_ch, meta_cc):
    #     """ Run training """
    #     # loss = model.train_on_batch([encoder_input_data, decoder_input_data, meta_hh, meta_hc, meta_ch, meta_cc], decoder_target_data)
    #     loss = model.fit([encoder_input_data, decoder_input_data, meta_hh, meta_hc, meta_ch, meta_cc], decoder_target_data,
    #                      batch_size=self.batch_size,
    #                      epochs=1)
    #     #                  # validation_split=0.2)
    #     return loss

    # def test_autoencoder(self,  model, encoder_input_data, decoder_input_data, decoder_target_data, meta_hh, meta_hc, meta_ch, meta_cc):
    #     loss = model.test_on_batch(
    #         [encoder_input_data, decoder_input_data, meta_hh, meta_hc, meta_ch, meta_cc], decoder_target_data)
    #     return loss

    # def train_context(self, model, train_data, teach_data):
    #     """ Run training """
    #     loss = model.fit(train_data, teach_data,
    #                      batch_size=self.batch_size,
    #                      epochs=1,
    #                      validation_split=0.2)

    #     return loss

    # def save_models(self, fname, model):
    #     print("save" + self.seq2seq_wait_save_dir + fname)
    #     model.save(self.seq2seq_wait_save_dir + fname)

    # def load_models(self, fname):
    #     print("load" + self.seq2seq_wait_save_dir + fname)
    #     from keras.models import load_model
    #     return load_model(self.seq2seq_wait_save_dir + fname)


def main():
    pass


if __name__ == "__main__":
    main()
