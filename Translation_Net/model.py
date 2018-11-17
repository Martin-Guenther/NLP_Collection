from keras.layers import LSTM, Input, Dense, Embedding
from keras.utils.vis_utils import plot_model
from keras.models import Model
import numpy as np

def encdec_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    """
    Build and train an encoder-decoder model on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
    # OPTIONAL: Implement
    num_embedding = 64
    num_lstm_cells = 128
    
    # training encoder
    encoder_inputs = Input(shape=(input_shape[1],), name="encoder_input")
    embed = Embedding(english_vocab_size, num_embedding, input_length=input_shape[1], mask_zero=False)
    embedded_inputs = embed(encoder_inputs)
    encoder = LSTM(num_lstm_cells, return_state=True, name="encoder_lstm")
    encoder_outputs, state_h, state_c = encoder(embedded_inputs)
    encoder_states = [state_h, state_c]
    
    # training decoder
    decoder_inputs = Input(shape=(None, french_vocab_size), name="decoder_input")
    decoder_lstm = LSTM(num_lstm_cells, return_sequences=True, return_state=True, name="decoder_lstm")
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(french_vocab_size, activation="softmax")
    decoder_outputs = decoder_dense(decoder_outputs)
    
    training_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    
    # inference encoder
    inference_encoder = Model(encoder_inputs, encoder_states)
    
    # inference decoder
    decoder_state_input_h = Input(shape=(num_lstm_cells,))
    decoder_state_input_c = Input(shape=(num_lstm_cells,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    
    inference_decoder = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    return training_model, inference_encoder, inference_decoder    


def predict_sequence(infenc, infdec, input_sequence, n_steps, french_vocab_size):
    # encode
    state = infenc.predict(input_sequence)
    
    # start of sequence input
    target_seq = np.array([0.0 for _ in range(french_vocab_size)]).reshape(1, 1, french_vocab_size)
    
    # collect predictions
    output = list()
    for t in range(n_steps):
        # predict next word
        yhat, h, c = infdec.predict([target_seq] + state)
        
        output.append(yhat[0, 0, :])
        
        state = [h, c]
        
        target_seq = yhat
        
    return np.array(output)