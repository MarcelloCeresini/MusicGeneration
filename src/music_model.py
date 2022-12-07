import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision

from transformers import GPT2Config, TFGPT2Model

from config import Config

ROOT_PATH = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
conf = Config("single_instruments_type", ROOT_PATH)

### CUSTOM LAYERS
# Custom intermediate layer for allowing types transformation (no parameters to be learnt)
class SubsequentTypeTransformationLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(SubsequentTypeTransformationLayer, self).__init__()
        # Use a StaticHashTable to map values to their consecutive version within Tensorflow
        self.keys_tensor = tf.range(conf.INPUT_RANGES['type'])
        self.vals_tensor = tf.constant([0,1,2,3,3,3,3,4])
        self.table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(self.keys_tensor, self.vals_tensor), 
            default_value=-1)

    def call(self, inputs):
        return self.table.lookup(inputs)
    

    
# Custom layer that computes masks for type probabilities computation
class MaskTypeProbabilitiesLayer(tf.keras.layers.Layer):
    def __init__(self, trainable=False, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)

    @tf.function
    def create_mask(self, inputs):
        '''
        Takes as input the token types of ONE song --> conf.SEQ_LEN-1 * 1
        Since the decoder creates an output for token i in output i-1 (i.e. the last output is not correlated with any token)
        Creates the mask for the NEXT token (depending on token i it masks output i, that corresponds to scores for token i+1)
        '''
        batch_gt_types = inputs
        mask = tf.TensorArray(tf.bool, size=conf.SEQ_LEN-1)
        for i in tf.range(conf.SEQ_LEN-1):
            token_type = batch_gt_types[i]
            if token_type == 0: # only start of song token: cannot be anything else than instrument choice (1)
                type_mask = tf.constant([False, True, False, False, False, False, False, False], dtype=tf.bool)
            elif token_type == 1: # we reached instrument choice: cannot be anything else than instrument choice (1) or start of events (2)
                type_mask = tf.constant([False, True, True, False, False, False, False, False], dtype=tf.bool)
            elif token_type == 2: # after a 2 there must be at least a 4
                type_mask = tf.constant([False, False, False, False, True, False, False, False], dtype=tf.bool)
            elif token_type == 3: # allow 3,4,5,6,7
                type_mask = tf.constant([False, False, False, True, True, True, True, True], dtype=tf.bool)
            elif token_type >= 4 and token_type <= 6:
                # - if there are at least a 5 and a 6 (there is always a 4)   --> [3, 4, 5, 6, 7]
                # - if a 5 is missing, we only allow 5                        --> [5]
                # - if a 6 is missing, we only allow 6                        --> [6]
                # i+1 is needed because if current token_type is 5 it counts (otherwise it would always put 2 consecutive 5)
                if tf.size(tf.where(batch_gt_types[:i+1] == 5)) == 0:
                    type_mask = tf.constant([False, False, False, False, False, True, False, False], dtype=tf.bool)
                elif tf.size(tf.where(batch_gt_types[:i+1] == 6)) == 0:
                    type_mask = tf.constant([False, False, False, False, False, False, True, False], dtype=tf.bool)
                else:
                    type_mask = tf.constant([False, False, False, True, True, True, True, True], dtype=tf.bool)
            elif token_type == 7: # at the end of the song we can ONLY GUESS "000000000" TODO: change ending token to type 7s -> 7000000000
                type_mask = tf.constant([False, False, False, False, False, False, False, True], dtype=tf.bool)
            else:
                # ERROR. Define a random type mask so that it's defined in all branches for tf.function
                type_mask = tf.constant([False, False, False, False, False, False, False, False], dtype=tf.bool)
            mask = mask.write(i, type_mask)
        return mask.stack()


    def call(self, inputs, training=True):
        '''
        Takes as input the ground truth song (at training time) or the logits (at testing time) 
        and computes a mask for the type probabilities.
        output masks is BATCH_SIZE * SEQ_LEN * 1 --> we mask also the last output even if it's useless
        '''
        if training:
            # Use the groundtruth song as a target
            song        = inputs
            gt_types    = song[:,:,0]       # Get the token types from the song (batch_size x seq_len-1)
            # Iterate over the batch to collect the appropriate masks from the song
            masks = tf.map_fn(fn=self.create_mask, 
                elems=gt_types, 
                fn_output_signature=tf.TensorSpec(
                    (conf.SEQ_LEN-1, conf.INPUT_RANGES['type']), 
                    dtype=tf.bool)
            )
            return masks
        else:
            # Compute the types and their masks one by one based on the type chosen at the previous iteration
            # TODO: implement this branch
            pass

# The main masking layer applying all constraints based on the predicted types 
class MaskingActivationLayer(tf.keras.layers.Layer):
    def __init__(self, trainable=False, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.default_mask = conf.default_mask
        self.full_mask    = conf.full_mask
        self._numerators  = tf.constant(conf.numerators)
        self._tot_numerators = tf.constant(conf.tot_numerators)

    @tf.function
    def get_max_beat_from_time_sign(self, time_sign):
        '''
        Since the time sign is defined (in utils.time_sign_map()) as: 
            conf.numerators.index(time_sign[0]) + conf.denominators.index(time_sign[1])*conf.tot_numerators

        to retrieve the NUMERATOR of the time_sign given the index you need to divide by conf.tot_numerators and take the rest of the division
        that gives you the index of the corresponding numerator in conf.numerators
        then you use gather or, more simply, a slice to get the actual value of the numerator
        '''
        idx = tf.math.floormod(time_sign, self._tot_numerators)
        return self._numerators[idx]

    @tf.function
    def get_mask_for_all_tokens(self, inputs): 
        '''
        Inputs:
        - chosen_types:         (SEQ_LEN-1)*1
        - song_tokens:          (SEQ_LEN-1)*11
        - seq_scores:           (SEQ_LEN-1)*1391

        Returns a list of ndarrays of bool type used for masking
        Inputs are for a SINGLE ELEMENT OF A BATCH of size SEQ_LEN*(1+11+1391) where 1391 is the summed length of logits (minus the type)
        '''
        # Collect inputs from longer tensor
        chosen_types, song_tokens, seq_scores = inputs
        chosen_types = tf.cast(chosen_types, dtype=tf.int32)
        song_tokens  = tf.cast(song_tokens , dtype=tf.int32)
        # Indexes
        index_tensor = tf.range(conf.SEQ_LEN-1, dtype=tf.int32)
        # Define mask (output) using a TensorArray
        mask = tf.TensorArray(dtype=tf.bool, size=conf.SEQ_LEN-1)
        # Iterate over the indexes
        for idx in index_tensor:
            until_idx = idx+1
            ## SETUP ##
            # Define the default variables and flags
            default_token_parts   = [True]*(len(conf.INPUT_RANGES)-1)
            default_flag          = False
            min_measure           = tf.constant(-1, dtype=tf.int32)
            min_beat              = tf.constant(-1, dtype=tf.int32)
            min_position          = tf.constant(-1, dtype=tf.int32)
            # TODO: variable length arrays: can we do it with tensorarrays?
            allowed_instruments   = tf.constant([0]*conf.INPUT_RANGES["instrument"], dtype=tf.int32)
            allowed_key_sign      = tf.constant(-1, dtype=tf.int32)
            allowed_time_sign     = tf.constant(-1, dtype=tf.int32)
            allowed_tempo         = tf.constant(-1, dtype=tf.int32)
            forbidden_instruments_flag = False
            forbidden_instruments = tf.constant([0]*conf.INPUT_RANGES["instrument"], dtype=tf.int32)
            forbidden_key_sign    = tf.constant(-1, dtype=tf.int32)
            forbidden_time_sign   = tf.constant(-1, dtype=tf.int32)
            forbidden_tempo       = tf.constant(-1, dtype=tf.int32)
            # Define the inputs
            chosen_type = chosen_types[idx]
            scores      = seq_scores[idx]
            song = song_tokens # TODO: don't need to mask?
            ### song        = song_tokens * (tf.expand_dims([1]*idx + [0]*(conf.SEQ_LEN-1-idx), axis=-1)) # Mask all tokens after index idx
            ## MAIN BODY ##
            if chosen_type == 0 or chosen_type == 2 or chosen_type == 7:
                default_token_parts = [True, True, True, True, True, True, True, True, True, True]
                default_flag = True
            elif chosen_type == 1: # Instrument selection, false only for type and instrument type (the ones that you can choose)
                if tf.size(tf.where(song[:until_idx, 0] == 1)[:,0]) == 0:
                    # Choice of first instrument
                    default_token_parts = [True, True, True, True, True, False, True, True, True, True]
                    default_flag = True
                else:
                    forbidden_instruments, _ = tf.unique(tf.gather(
                        song[:until_idx, 6], 
                        tf.where(song[:until_idx, 0] == 1)[:,0]        # Cast to 1D array
                    ))
                    forbidden_instruments_flag = True
            elif chosen_type == 3: # Notes: They have the same key_sign, time_sign and tempo as last previous event, everything has to be manually decided
                min_measure = song[idx, 1]   # It has to be >= than the last measure
                # If in the MEASURE SCORES the MAX SCORE between all possible measures == min_measure, the measure is min_measure.
                # In this case, we need to make sure that beat >= last_beat
                if tf.math.argmax(
                    scores[:conf.INPUT_RANGES["measure"]], 
                        output_type=tf.int32) == min_measure:  
                    min_beat = song[idx,2]      # It has to be >= than the last beat when measure is the same

                    if tf.math.argmax(scores[
                        conf.INPUT_RANGES["measure"] : 
                        conf.INPUT_RANGES["measure"] + conf.INPUT_RANGES["beat"]], 
                        output_type=tf.int32) == min_beat:
                        min_position = song[idx,3]  # It has to be >= than the last position (if beat and measure are the same)
                    else:
                        min_position = tf.constant(0, dtype=tf.int32)
                else:
                    min_beat = tf.constant(0, dtype=tf.int32)
                    min_position = tf.constant(0, dtype=tf.int32)

                # Only some instruments, key signs, time signs and tempos are allowed for these events: 
                # - for instruments, the allowed ones are the ones that have been defined previously with type = 1
                # - for the others, the allowed ones are the ones that are collected right before the note from event types 4, 5 and 6
                allowed_instruments, _ = tf.unique(tf.gather(
                    song[:until_idx, 6], 
                    tf.where(song[:until_idx, 0] == 1)[:,0]
                ))
                # We have made it so that the model should output 3s only after at least a 4, 5 and 6.
                allowed_key_sign = tf.gather(
                    song[:until_idx, 8], 
                    tf.where(song[:until_idx, 0] == 4)[:,0]   # if type == 4 --> read the LAST key_sign
                )[-1] 
                allowed_time_sign = tf.gather(
                    song[:until_idx, 9], 
                    tf.where(song[:until_idx, 0] == 5)[:,0]   # if type == 5 --> read the LAST time_sign
                )[-1] 
                allowed_tempo = tf.gather(
                    song[:until_idx, 10], 
                    tf.where(song[:until_idx, 0] == 6)[:,0]   # if type == 6 --> read the LAST tempo
                )[-1] 
            elif chosen_type >= 4 and chosen_type <= 6:     # key_sign, time_sign, tempo
                # If last event is at the beginning of a measure, you can add an event at the same time
                if song[idx, 3] == 0 and song[idx, 2] == 0:  # if beat and position == 0, the event can be at this measure
                    min_measure = song[idx, 1]
                else:
                    min_measure = song[idx, 1] + 1                   # otherwise it goes to the next measure
                # Fine-grain checks
                # Here, there are cases where there is not a LAST key_sign/time_sign (when this is the first 4, 5 or 6). 
                # In these cases we should use the default masks.
                if chosen_type == 4:
                    # Cannot put the same key_sign again
                    tmp = tf.gather(
                        song[:until_idx, 8], 
                        tf.where(song[:until_idx, 0] == 4)[:,0]) # if type == 4 --> read all the key_sign
                    if tf.size(tmp) > 0: # if there is at least one before this
                        forbidden_key_sign = tmp[-1] # cannot choose the LAST key_sign
                elif chosen_type == 5:
                    # Cannot put the same time_sign again
                    tmp = tf.gather(
                        song[:until_idx, 9], 
                        tf.where(song[:until_idx, 0] == 5)[:,0]) # if type == 5 --> read the LAST time_sign
                    if tf.size(tmp) > 0:
                        forbidden_time_sign = tmp[-1]
                elif chosen_type == 6:
                    # Cannot put the same tempo again
                    tmp = tf.gather(
                        song[:until_idx, 10], 
                        tf.where(song[:until_idx, 0] == 6)[:,0]) # if type == 6 --> read the LAST tempo
                    if tf.size(tmp) > 9:
                        forbidden_tempo = tmp[-1]

            ## ENDING PART ##
            # Put together the masks
            if default_flag:                
                # No manual masking required, either "can freely choose this part of the token" (True) or 
                # "can only choose default for this part of the token" (False)
                mask = mask.write(idx, tf.concat(
                    # Default mask only allows to predict a 0
                    # Full mask allows to predict any value
                    [self.default_mask[i] if default_token_parts[i] else self.full_mask[i] 
                        for i in range(len(default_token_parts))], axis=-1)
                )
            elif forbidden_instruments_flag:
                # Default flag is False and forbidden instruments contains some elmeents (which means that the chosen type is 1)
                instruments_mask = tf.sparse.SparseTensor(  # Forbidden instruments
                        indices= tf.expand_dims(tf.cast(forbidden_instruments, tf.int64), axis=-1),
                        values = tf.zeros_like(forbidden_instruments),
                        dense_shape=[conf.INPUT_RANGES["instrument"]]
                    )
                instruments_mask = tf.cast(
                    tf.sparse.to_dense(tf.sparse.reorder(instruments_mask), default_value=1), 
                    dtype=tf.dtypes.bool)
                
                # Only mask the forbidden instruments, all the rest is default
                mask = mask.write(idx, tf.concat(
                    [self.default_mask[i] for i in range(5)] + \
                    [instruments_mask] + \
                    [self.default_mask[i] for i in range(6,len(default_token_parts))], 
                    axis=-1))
            elif chosen_type >= 3 and chosen_type <= 6:
                # General event. What we do depends on which specific event it is, but
                # in general there is always a measure mask.
                
                measure_mask = tf.cast(
                    tf.concat([
                            tf.repeat([False], min_measure),        # Can be equal to or greater than min_measure
                            tf.repeat([True],  conf.INPUT_RANGES["measure"]-min_measure)
                        ], axis=-1
                    ), dtype=tf.bool
                )
                # We need to do manual masking. Define all tensors (default_masks will most probably be changed, full_masks won't)
                beat_mask        = self.default_mask[1]
                position_mask    = self.default_mask[2]
                duration_mask    = self.default_mask[3]
                pitch_mask       = self.default_mask[4]
                instruments_mask = self.default_mask[5]
                velocity_mask    = self.default_mask[6]
                key_sign_mask    = self.default_mask[7]
                time_sign_mask   = self.default_mask[8]
                tempo_mask       = self.default_mask[9]
                # Create more specific masks depending on the type
                if chosen_type == 4:
                    if forbidden_key_sign != -1: ## forbidden_key_sign can only appear if chosen_type = 4
                        # True in all places but the forbidden key signs
                        key_sign_mask = tf.convert_to_tensor([
                            i != forbidden_key_sign 
                            for i in range(conf.INPUT_RANGES["key_sign"])], 
                            dtype=tf.bool)
                    else:
                        key_sign_mask = self.full_mask[7]
                elif chosen_type == 5:
                    if forbidden_time_sign != -1: ## forbidden_time_sign can only appear if chosen_type = 5
                        # True in all places but the forbidden time signs
                        time_sign_mask = tf.convert_to_tensor([
                            i != forbidden_time_sign 
                            for i in range(conf.INPUT_RANGES["time_sign"])], 
                            dtype=tf.bool)
                    else:
                        time_sign_mask = self.full_mask[8]
                elif chosen_type == 6:
                    if forbidden_tempo != -1: ## forbidden_tempo can only appear if chosen_type = 6
                        # True in all places but the forbidden tempos
                        tempo_mask = tf.convert_to_tensor([
                            i != forbidden_tempo 
                            for i in range(conf.INPUT_RANGES["tempo"])], 
                            dtype=tf.bool)
                    else:
                        tempo_mask = self.full_mask[9]
                elif chosen_type == 3:
                    # can always choose any of the below for a note
                    duration_mask    = self.full_mask[3]
                    pitch_mask       = self.full_mask[4]
                    velocity_mask    = self.full_mask[6]

                    # If the event is a note, we have ALLOWED time signs/tempos/key signs, not
                    # forbidden ones. Also, there are many other elements to take into account
                    if min_beat != -1: # it is ALWAYS != -1 if chosen_type == 3
                        # oss: allowed_time_sign is always != None if min_beat != None
                        max_beat = self.get_max_beat_from_time_sign(allowed_time_sign)
                        # allowed beats are only AFTER previous beat and BEFORE max_beat from the numerator of the time_sign
                            
                        beat_mask = tf.cast(tf.concat([
                            tf.repeat([False], min_beat),
                            tf.repeat([True],  max_beat-min_beat), 
                            tf.repeat([False], conf.INPUT_RANGES["beat"]-max_beat)],
                            axis=-1), 
                        dtype=tf.bool)
                        
                    if min_position != -1: # it is ALWAYS != -1 if chosen_type == 3
                            
                        position_mask = tf.cast(tf.concat([
                            tf.repeat([False], min_position), 
                            tf.repeat([True],  conf.INPUT_RANGES["position"]-min_position)],
                            axis=-1), 
                        dtype=tf.dtypes.bool)

                    instruments_mask = tf.sparse.SparseTensor( # Allowed instruments
                        indices=tf.expand_dims(tf.cast(allowed_instruments, tf.int64), axis=-1),
                        values=tf.ones_like(allowed_instruments),
                        dense_shape=[conf.INPUT_RANGES["instrument"]]
                    )
                    instruments_mask = tf.cast(
                        tf.sparse.to_dense(tf.sparse.reorder(instruments_mask), default_value=0),
                        dtype=tf.dtypes.bool)

                    if allowed_key_sign != -1: # it is ALWAYS != -1 if chosen_type == 3
                        key_sign_mask = tf.convert_to_tensor([
                            i == allowed_key_sign 
                            for i in range(conf.INPUT_RANGES["key_sign"])], 
                            dtype=tf.bool)
                    if allowed_time_sign != -1: # it is ALWAYS != -1 if chosen_type == 3
                        time_sign_mask = tf.convert_to_tensor([
                            i == allowed_time_sign 
                            for i in range(conf.INPUT_RANGES["time_sign"])], 
                            dtype=tf.bool)
                    if allowed_tempo != -1: # it is ALWAYS != -1 if chosen_type == 3
                        tempo_mask = tf.convert_to_tensor([
                            i == allowed_tempo 
                            for i in range(conf.INPUT_RANGES["tempo"])], 
                            dtype=tf.bool)
                
                mask = mask.write(idx, tf.concat([
                    measure_mask, beat_mask, position_mask, duration_mask,
                    pitch_mask, instruments_mask, velocity_mask, key_sign_mask,
                    time_sign_mask, tempo_mask], axis=-1))

        return mask.stack()

    def call(self, inputs, training=True):
        '''
        Inputs:
        - songs:                BATCH*(SEQ_LEN-1)*11
        - out_logits:           BATCH*(SEQ_LEN-1)*1391 (all except type)
        - types_probabilities:  BATCH*(SEQ_LEN-1)*8 --> becomes chosen_types through argmax --> BATCH*(SEQ_LEN-1)*1

        passes through map_fn --> get_mask_fro_all_tokens to debatch
        '''
        songs, out_logits, types_probabilities = inputs
        chosen_types  = tf.expand_dims(tf.math.argmax(types_probabilities, axis=2), axis=-1)
        concat_logits = tf.concat(out_logits[1:], axis=-1)                 # Concatenate all logits (except type) into a tensor batch_size x seq_len x 1391
        masks = tf.map_fn(fn=self.get_mask_for_all_tokens, elems=(         # Iterate function over batch dimension 
                tf.cast(chosen_types, concat_logits.dtype),                # BATCH*(SEQ_LEN-1)*1
                tf.cast(songs,   concat_logits.dtype),                     # BATCH*(SEQ_LEN-1)*11
                concat_logits                                              # BATCH*(SEQ_LEN-1)*1391 
            ), fn_output_signature=tf.TensorSpec(                          # Total: a BATCH * SEQ_LEN-1 * 1403 tensor
                (conf.SEQ_LEN-1, conf.input_ranges_sum - conf.INPUT_RANGES['type']),
                dtype=tf.bool
            ))

        return masks



def create_model(input_shape=(conf.SEQ_LEN-1, len(conf.INPUT_RANGES)), num_genres=len(conf.accepted_subgenres), 
                 use_regularization=True, use_masking_layers=True, reg_loss_scale=conf.REG_LOSS_SCALE):
    
    # Get input shapes
    seq_len_no_genre = input_shape[0]
    events_elements = input_shape[1]
    
    # Instantiate transformer decoder (n_emb % n_head must be 0)
    decoder = conf.get_decoder()
    
    # Define inputs
    songs  = tf.keras.Input(shape=input_shape, name='songs',  dtype=tf.int32)
    genres = tf.keras.Input(shape=num_genres , name='genres', dtype=tf.float32)
    
    # Loss utilities
    subsequent_type_transform_layer = SubsequentTypeTransformationLayer()
    reg_scaler = tf.constant(reg_loss_scale, dtype=tf.float32)
    loss_function = tf.keras.losses.SparseCategoricalCrossentropy(reduction="sum")
    
    # Embedding layers
    embedding_layers = [
        # Type embedding
        tf.keras.layers.Embedding(conf.INPUT_RANGES['type'],       conf.SINGLE_EMB_SIZE, 
                                  input_length=seq_len_no_genre, name='type_embeddings'),
        # Measure embedding
        tf.keras.layers.Embedding(conf.INPUT_RANGES['measure'],    conf.SINGLE_EMB_SIZE, 
                                  input_length=seq_len_no_genre, name='measure_embeddings'),
        # Beat embedding
        tf.keras.layers.Embedding(conf.INPUT_RANGES['beat'],       conf.SINGLE_EMB_SIZE, 
                                  input_length=seq_len_no_genre, name='beat_embeddings'),
        # Position embedding
        tf.keras.layers.Embedding(conf.INPUT_RANGES['position'],   conf.SINGLE_EMB_SIZE, 
                                  input_length=seq_len_no_genre, name='position_embeddings'),
        # Duration embedding
        tf.keras.layers.Embedding(conf.INPUT_RANGES['duration'],   conf.SINGLE_EMB_SIZE, 
                                  input_length=seq_len_no_genre, name='duration_embeddings'),
        # Pitch embedding
        tf.keras.layers.Embedding(conf.INPUT_RANGES['pitch'],      conf.SINGLE_EMB_SIZE, 
                                  input_length=seq_len_no_genre, name='pitch_embeddings'),
        # Instrument embedding
        tf.keras.layers.Embedding(conf.INPUT_RANGES['instrument'], conf.SINGLE_EMB_SIZE, 
                                  input_length=seq_len_no_genre, name='instrument_embeddings'),
        # Velocity embedding
        tf.keras.layers.Embedding(conf.INPUT_RANGES['velocity'],   conf.SINGLE_EMB_SIZE, 
                                  input_length=seq_len_no_genre, name='velocity_embeddings'),
        # Key sign embedding
        tf.keras.layers.Embedding(conf.INPUT_RANGES['key_sign'],   conf.SINGLE_EMB_SIZE, 
                                  input_length=seq_len_no_genre, name='key_sign_embeddings'),
        # Time sign embedding
        tf.keras.layers.Embedding(conf.INPUT_RANGES['time_sign'],  conf.SINGLE_EMB_SIZE, 
                                  input_length=seq_len_no_genre, name='time_sign_embeddings'),
        # Tempo embedding
        tf.keras.layers.Embedding(conf.INPUT_RANGES['tempo'],      conf.SINGLE_EMB_SIZE, 
                                  input_length=seq_len_no_genre, name='tempo_embeddings')
    ]
    
    genre_embedding_layer = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(conf.DROPOUT_VALUE),
        tf.keras.layers.Dense(conf.GENRE_DIM)
    ], name='genre_embedding')
    
    # Input processing layers
    input_concat_layer         = tf.keras.layers.Concatenate(axis=2)
    sequence_concat_layer      = tf.keras.layers.Concatenate(axis=1)
    encoding_processing_layer  = tf.keras.layers.Dense(conf.TOKEN_DIM, name='encoding_processing')
    
    # Positional encoding
    positional_encoding_matrix = conf.get_positional_embedding_matrix()
    positional_encoding        = tf.repeat(positional_encoding_matrix[tf.newaxis, :, :], tf.shape(songs)[0], axis=0)
    sum_layer                  = tf.keras.layers.Add(name='final_encoding')

    # Output layers
    output_dense_layers = [
        # Type
        tf.keras.layers.Dense(conf.INPUT_RANGES['type'],       name='type_scores'),
        # Measure
        tf.keras.layers.Dense(conf.INPUT_RANGES['measure'],    name='measure_scores'),
        # Beat
        tf.keras.layers.Dense(conf.INPUT_RANGES['beat'],       name='beat_scores'),
        # Position
        tf.keras.layers.Dense(conf.INPUT_RANGES['position'],   name='position_scores'),
        # Duration
        tf.keras.layers.Dense(conf.INPUT_RANGES['duration'],   name='duration_scores'),
        # Pitch
        tf.keras.layers.Dense(conf.INPUT_RANGES['pitch'],      name='pitch_scores'),
        # Instrument
        tf.keras.layers.Dense(conf.INPUT_RANGES['instrument'], name='instrument_scores'),
        # Velocity
        tf.keras.layers.Dense(conf.INPUT_RANGES['velocity'],   name='velocity_scores'),
        # Key sign
        tf.keras.layers.Dense(conf.INPUT_RANGES['key_sign'],   name='keysign_scores'),
        # Time sign
        tf.keras.layers.Dense(conf.INPUT_RANGES['time_sign'],  name='timesign_scores'),
        # Tempo
        tf.keras.layers.Dense(conf.INPUT_RANGES['tempo'],      name='tempo_scores')
    ]
    
    # OSS: names of these layers must correspond to the keys of dict y_true (in case it's a dict)
    output_probs_layers = [
        # Type
        tf.keras.layers.Softmax(name='type_probabilities'),
        # Measure
        tf.keras.layers.Softmax(name='measure_probabilities'),
        # Beat
        tf.keras.layers.Softmax(name='beat_probabilities'),
        # Position
        tf.keras.layers.Softmax(name='position_probabilities'),
        # Duration
        tf.keras.layers.Softmax(name='duration_probabilities'),
        # Pitch
        tf.keras.layers.Softmax(name='pitch_probabilities'),
        # Instrument
        tf.keras.layers.Softmax(name='instrument_probabilities'),
        # Velocity
        tf.keras.layers.Softmax(name='velocity_probabilities'),
        # Key sign
        tf.keras.layers.Softmax(name='key_sign_probabilities'),
        # Time sign
        tf.keras.layers.Softmax(name='time_sign_probabilities'),
        # Tempo
        tf.keras.layers.Softmax(name='tempo_probabilities')
    ]
    
    # Masking layers
    if use_masking_layers:
        type_masking_layer = MaskTypeProbabilitiesLayer()
        activations_masking =  MaskingActivationLayer()
    
    # Model dynamics
    embeddings        = [embedding_layers[i](songs[:,:,i]) for i in range(events_elements)]
    genre_embedding   = genre_embedding_layer(genres)
    input_embedding   = input_concat_layer(embeddings)
    input_embedding   = encoding_processing_layer(input_embedding)
    input_embedding   = sequence_concat_layer([genre_embedding[:, np.newaxis, :], input_embedding])
    input_embedding   = sum_layer([input_embedding, positional_encoding])
    model_output      = decoder({'inputs_embeds': input_embedding})['last_hidden_state']
    out_scores        = [output_dense_layers[i](model_output)[:,:-1,:] 
                         for i in range(len(output_dense_layers))] # [BATCH_SIZE * SEQ_LEN-1 * events_elements] * INPUT_RANGES (various)

    # We don't care about the last scores, since they refer to a token that's out of bounds.
    if use_masking_layers:
        type_mask           = type_masking_layer(songs, training=True)
        types_probabilities = output_probs_layers[0](out_scores[0], type_mask) # BATCH_SIZE * SEQ_LEN-1 * 8
        full_mask           = activations_masking([songs, out_scores, types_probabilities])
        # Unpack the final masks into a list of masks
        index = 0; masks = []          
        for key in conf.INPUT_RANGES:
            if key != 'type':
                masks.append(full_mask[:, :, index:index+conf.INPUT_RANGES[key]])
                index += conf.INPUT_RANGES[key]
        # Call all the softmax layers
        out_probabilities = [types_probabilities] + [
            output_probs_layers[i](out_scores[i], masks[i-1]) 
            for i in range(1, len(output_dense_layers))]
    else:
        out_probabilities = [output_probs_layers[i](out_scores[i]) 
        for i in range(len(output_dense_layers))]
    
    out_probabilities_dict = {
        key: out_probabilities[i] 
        for i, key in enumerate(conf.INPUT_RANGES.keys())
    }
    
    # Create model
    model = tf.keras.Model(
        inputs=[songs, genres], 
        outputs=out_probabilities_dict, 
        name='music_generation_model'
    )
    
    # Define regularizers
    def custom_regularizers(songs, y_pred):
        # Regularization loss: transform the actual vectors into consecutive-type representation
        max_pred_types = tf.argmax(y_pred[0], axis=2, output_type=tf.int32)
        consecutive_pred_types = subsequent_type_transform_layer(max_pred_types)
        # Compute difference
        differences = consecutive_pred_types[:, 1:] - consecutive_pred_types[:, :-1]
        # Compute regularization terms
        # Difference between one element's type and the next is >= 0
        reg_term_1 = tf.math.reduce_sum(tf.math.maximum(0, -differences))
        # Difference between one element's type and the next is < 1
        reg_term_2 = tf.math.reduce_sum(tf.math.maximum(0, tf.math.maximum(1, differences) - 1))
        return reg_scaler * tf.cast(reg_term_1, tf.float32) + reg_scaler * tf.cast(reg_term_2, tf.float32)
    
    # Loss function for each separate part of the tokens
    def custom_loss(y_true, y_pred):
        y_true_concat_batch = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred_concat_batch = tf.reshape(y_pred, [-1, y_pred.shape[2]])
        loss = loss_function(y_true_concat_batch, y_pred_concat_batch) * \
                (1. / (conf.GLOBAL_BATCH_SIZE * (conf.SEQ_LEN-1)))
        return loss
    
    # Cannot do it with standard framework, so we add a different loss for each part of the tokens
    for i in range(len(conf.INPUT_RANGES.keys())):
        loss_name = list(conf.INPUT_RANGES.keys())[i] + '_loss'
        loss = custom_loss(y_true = songs[:,:,i], y_pred = out_probabilities[i])
        model.add_loss(loss)
        model.add_metric(loss, name=loss_name)
    
    if use_regularization:
        reg_loss = custom_regularizers(songs, out_scores)
        model.add_loss(reg_loss)
        model.add_metric(reg_loss, name='regularization_loss')
    
    # Compile and return
    model.compile(optimizer="adam")
    return model