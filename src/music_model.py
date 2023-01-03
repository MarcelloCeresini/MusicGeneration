import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision

from config import Config

### CUSTOM LAYERS
# Custom intermediate layer for allowing types transformation (no parameters to be learnt)
class SubsequentTypeTransformationLayer(tf.keras.layers.Layer):
    def __init__(self, conf:Config):
        super(SubsequentTypeTransformationLayer, self).__init__()
        # Use a StaticHashTable to map values to their consecutive version within Tensorflow
        self.keys_tensor = tf.range(conf.INPUT_RANGES['type'])
        self.vals_tensor = tf.constant([0,1,2,3,3,3,3,4])
        self.table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(self.keys_tensor, self.vals_tensor), 
            default_value=-1)

    def call(self, inputs):
        return self.table.lookup(inputs)


# Custom intermediate layer for regularization that computes the loss related to 
# miscellaneous type errors that could happen in the generated song
class MiscTypeChecker(tf.keras.layers.Layer):
     def __init__(self, conf:Config):
        super(MiscTypeChecker, self).__init__()
        self.conf = conf
    
     def call(self, inputs):
        max_pred_types = inputs
        # 1) First token must have type 0 (each batch element times 4 to keep it comparable)
        rg1 = tf.math.reduce_sum(tf.cast(max_pred_types[:, 0] != 0, tf.int32)*4)
        # 2) Second token must have type 1 (each batch element times 4 to keep it comparable)
        rg2 = tf.math.reduce_sum(tf.cast(max_pred_types[:, 1] != 1, tf.int32)*4)
        rg3s = tf.TensorArray(dtype=tf.int32, size=tf.shape(max_pred_types)[0])
        rg4s = tf.TensorArray(dtype=tf.int32, size=tf.shape(max_pred_types)[0])
        for b in tf.range(tf.shape(max_pred_types)[0]):
            ones = tf.cast(tf.where(max_pred_types[b] == 1), tf.int32)
            last_1 = -1
            if tf.size(ones)  > 0: last_1 = tf.squeeze(ones[-1])
            # 3) There should be at least one of each type (squared to be comparable to other losses)
            rg3s = rg3s.write(b, (self.conf.INPUT_RANGES['type'] - tf.size(tf.unique(max_pred_types[b])[0]))**2)
            # 4) From the last 1 type token there should be the following types pattern:
            #    ..., 1, 2, 4, 5, 6, 3, ...
            if 0 < last_1 < (tf.shape(max_pred_types)[1] - 5):
                rg4s = rg4s.write(b, (tf.cast(max_pred_types[b, last_1 + 1] != 2, tf.int32) + \
                                      tf.cast(max_pred_types[b, last_1 + 2] != 4, tf.int32) + \
                                      tf.cast(max_pred_types[b, last_1 + 3] != 5, tf.int32) + \
                                      tf.cast(max_pred_types[b, last_1 + 4] != 6, tf.int32) + \
                                      tf.cast(max_pred_types[b, last_1 + 5] != 3, tf.int32)))
            else:
                # Something has gone wrong, so the error would be the maximum + 1
                rg4s = rg4s.write(b, 6)
        return rg1 + rg2 + tf.math.reduce_sum(rg3s.stack()) + tf.math.reduce_sum(rg4s.stack())


# Custom intermediate layer for regularization that computes the loss related to duplicate instruments
# definition and instruments that are used wrongly in the notes.
class InstrumentsChecker(tf.keras.layers.Layer):
     def __init__(self, conf:Config):
        super(InstrumentsChecker, self).__init__()
        self.conf = conf
    
     def call(self, inputs):
        max_pred_types, instrument_scores = inputs
        reg_term_2_list = tf.TensorArray(dtype=tf.int32, size=tf.shape(max_pred_types)[0])
        for b in tf.range(tf.shape(max_pred_types)[0]):
            instruments_in_batch = tf.argmax(
                tf.gather(instrument_scores[b], tf.where(max_pred_types[b] == 1)[:, 0]),
                axis=-1)
            unique_instruments_in_batch, _ = tf.unique(instruments_in_batch)
            instruments_in_notes = tf.argmax(
                tf.gather(instrument_scores[b], tf.where(max_pred_types[b] == 3)[:, 0]),
                axis=-1)
            unique_instruments_in_notes, _, count_of_instruments_in_notes = \
                tf.unique_with_counts(instruments_in_notes)
            undefined_instruments_in_notes = tf.sparse.to_dense(
                  tf.sets.difference(tf.expand_dims(unique_instruments_in_notes, axis=0), 
                                     tf.expand_dims(unique_instruments_in_batch, axis=0)))[0]
            indices_of_undefined_instruments = tf.where(
                tf.expand_dims(undefined_instruments_in_notes, axis=1) == unique_instruments_in_notes)[:, 1]
            count_of_undefined_instruments = tf.gather(count_of_instruments_in_notes, indices_of_undefined_instruments)
            # Difference between the number of selected instruments and the number of unique instruments
            # (AKA: number of duplicates)
            reg_term_2_1 = tf.shape(instruments_in_batch)[0] - tf.shape(unique_instruments_in_batch)[0]
            # Sum the number of undefined instruments in notes
            reg_term_2_2 = tf.math.reduce_sum(count_of_undefined_instruments)
            reg_term_2_list = reg_term_2_list.write(b, reg_term_2_1 + reg_term_2_2)
        return tf.math.reduce_sum(reg_term_2_list.stack())
    

# Custom layer that computes masks for type probabilities computation
class MaskTypeProbabilitiesLayer(tf.keras.layers.Layer):
    def __init__(self, conf:Config, trainable=False, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.conf = conf

    @tf.function
    def mask_single_token_in_song(self, inputs):
        i, batch_gt_types = inputs
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
            # - if there are at least a 5 and a 6 (there is always a 4 from ==3)    --> [3, 4, 5, 6, 7]
            # - if a 5 is missing, we only allow 5                                  --> [5]
            # - if a 6 is missing, we only allow 6                                  --> [6]
            # i+1 is needed because if current token_type is 5 it counts (otherwise it would always put 2 consecutive 5)
            # create mask for subsequent operations: 1 if idx<=i+1, 0 otherwise
            tmp_token_mask = tf.cast(tf.less_equal(
                tf.range(self.conf.SEQ_LEN-1),
                i+1
            ), dtype="float32")
            tmp_token_mask = tf.ensure_shape(tmp_token_mask, self.conf.SEQ_LEN-1)
            # create vector of 1s when type == 5 --> eliminate the ones after i+1 with multiply, then count them by summing 
            time_sign_occurrences = tf.reduce_sum(tf.math.multiply(
                tf.cast(tf.math.equal(batch_gt_types, 5), dtype="float32"),
                tmp_token_mask
            ))
            tempo_occurrences = tf.reduce_sum(tf.math.multiply(
                tf.cast(tf.math.equal(batch_gt_types, 6), dtype="float32"),
                tmp_token_mask
            ))
            if time_sign_occurrences == 0:
                type_mask = tf.constant([False, False, False, False, False, True, False, False], dtype=tf.bool)
            elif tempo_occurrences == 0:
                type_mask = tf.constant([False, False, False, False, False, False, True, False], dtype=tf.bool)
            else:
                type_mask = tf.constant([False, False, False, True, True, True, True, True], dtype=tf.bool)
        elif token_type == 7: # at the end of the song we can ONLY GUESS "700000000"
            type_mask = tf.constant([False, False, False, False, False, False, False, True], dtype=tf.bool)
        else:
            # ERROR. Define a random type mask so that it's defined in all branches for tf.function
            type_mask = tf.constant([False, False, False, False, False, False, False, False], dtype=tf.bool)
        return tf.ensure_shape(type_mask, 8)


    def mask_single_song_in_batch(self, inputs):
        '''
        Takes as input the token types of ONE song --> conf.SEQ_LEN-1 * 1
        Since the decoder creates an output for token i in output i-1 (i.e. the last output is not correlated with any token)
        Creates the mask for the NEXT token (depending on token i it masks output i, that corresponds to scores for token i+1)
        '''
        batch_gt_types = inputs
        mask = tf.vectorized_map(
            fn=self.mask_single_token_in_song,
            elems=(
                tf.range(self.conf.SEQ_LEN-1),
                tf.expand_dims(batch_gt_types, axis=0) # if first_dim = 1 --> passes always the same tensor
            ),        
        )
        return tf.ensure_shape(mask, (self.conf.SEQ_LEN-1, 8))


    def call(self, inputs, training=True):
        '''
        Takes as input the ground truth song (at training time) or the logits (at testing time) 
        and computes a mask for the type probabilities.
        output masks is BATCH_SIZE * SEQ_LEN * 1 --> we mask also the last output even if it's useless
        '''
        if training:
            # Use the groundtruth song as a target
            gt_types    = inputs[:,:,0]       # Get the token types from the song (batch_size x seq_len-1)
            # Iterate over the batch to collect the appropriate masks from the song
            batch_size = tf.shape(gt_types)[0]
            masks = tf.map_fn(
                fn=self.mask_single_song_in_batch, 
                elems=gt_types,
                fn_output_signature=(
                    tf.TensorSpec(shape=(self.conf.SEQ_LEN-1, 8), dtype="bool")
                ),
                # parallel_iterations=batch_size
            )
            return masks
        else:
            # Compute the types and their masks one by one based on the type chosen at the previous iteration
            # TODO: implement this branch
            pass


# The main masking layer applying all constraints based on the predicted types 
class MaskingActivationLayer(tf.keras.layers.Layer):
    def __init__(self, conf:Config, trainable=False, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.conf = conf
        self.default_mask = self.conf.default_mask
        self.full_mask    = self.conf.full_mask
        self._numerators  = tf.constant(self.conf.numerators)
        self._tot_numerators = tf.constant(self.conf.tot_numerators)


    @tf.function
    def get_max_beat_from_time_sign(self, time_sign):
        '''
        Since the time sign is defined (in utils.time_sign_map()) as: 
            conf.numerators.index(time_sign[0]) + conf.denominators.index(time_sign[1])*conf.tot_numerators

        to retrieve the NUMERATOR of the time_sign given the index you need to divide by conf.tot_numerators and take the rest of the division
        that gives you the index of the corresponding numerator in conf.numerators
        then you use gather or, more simply, a slice to get the actual value of the numerator
        '''
        idx = tf.math.floormod(tf.cast(time_sign, dtype=tf.dtypes.int32), self._tot_numerators)
        return self._numerators[idx]


    @tf.function
    def mask_tokens_if_type_is_desired_type(self, song, tmp_token_mask, desired_type):
        # useful if chosen_type == 1 and if chosen_type==3
        # 1 if token is instrument, 0 otherwise
        token_is_desired_type = tf.math.multiply(
            tf.cast(tf.math.equal(song[:,0], desired_type), dtype="float32"),
            tmp_token_mask
        )
        token_is_desired_type = tf.ensure_shape(token_is_desired_type, self.conf.SEQ_LEN-1)
        return token_is_desired_type


    @tf.function
    def get_mask_of_present_values_from_desired_type(self, 
        song, tmp_token_mask, 
        desired_type, desired_type_string, desired_position_in_token, 
        reverse_mask=False,
        only_last=False):
        """ 
        Given the song, the mask for tokens up until i+1 and the desired type, 
        returns the mask with the right shape for all values present in the song of the desired type
        
        ex. If desired_type == 1: the function will return a binary mask with as many elements as defined 
        in `conf.INPUT_RANGES["instrument"]` with 1 if an instrument is present in the song, 0 otherwise.
        
        If reverse_mask=True, the 0s become 1s and vice-versa

        Inputs:
        - song:                         6143*11     all tokens of the song
        - tmp_token_mask:               6143        a binary mask, 1 if token is <= i+1, 0 otherwise
        - desired_type:                 1           in [0,7], get only the tokens of desired type
        - desired_type_string           1           string used to get the shape of the final mask (ex. "instrument" mask has shape 129)
        - desired_position_in_token     1           where to look in the size 11 token (ex. instruments are defined in position 6)
        - reverse_mask                  1           if True, return the complementary mask

        Returns:
        - mask                          INPUT_LENGTH["desired_type_string"] 
        The mask has 1s where the tokens UP TO i+1 contain that specific feature value 
        (ex. a specific instrument) and 0 otherwise
        """
        token_is_desired_type = self.mask_tokens_if_type_is_desired_type(song, tmp_token_mask, desired_type)
        
        if not only_last:
            # need to be bool to be put as the condition of a tf.where
            token_is_desired_type = tf.cast(token_is_desired_type, dtype="bool")

            # if there is a token of desired_type, save its value, otherwise put last "fake" index
            indexes_masked = tf.expand_dims(tf.cast(tf.where(
                condition=token_is_desired_type,
                x=song[:,desired_position_in_token],          # a vector of feature indexes
                y=self.conf.INPUT_RANGES[desired_type_string]    # a scalar = last feature index + 1
                # y=-1    # a scalar = -1
            ), dtype="int32"), axis=-1) # need to be int32/64 to be inside indices of scatter_nd 

            if not reverse_mask:
                # create a tensor with one more spot than needed
                type_mask_tmp = tf.zeros(self.conf.INPUT_RANGES[desired_type_string] + 1)

                # for each index in the indexes_masked tensor above, put 1 in the mask
                # many of the values of the above tensor will be the fake index
                # It the end, we should have 1 for instruments that are defined (and can be chosen), 
                # and 0 for the ones that are not defined (and cannot be chosen)
                type_mask_tmp = tf.tensor_scatter_nd_max(
                    tensor=type_mask_tmp,
                    indices=indexes_masked,
                    updates=tf.ones(self.conf.SEQ_LEN-1)
                )

            else:
                # ones instead of zeros
                type_mask_tmp = tf.ones(self.conf.INPUT_RANGES[desired_type_string] + 1)
                # min instead of max
                type_mask_tmp = tf.tensor_scatter_nd_min(
                    tensor=type_mask_tmp,
                    indices=indexes_masked,
                    # zeros instead of ones
                    updates=tf.zeros(self.conf.SEQ_LEN-1)
                )

            # Remove the last fake token
            tmp_return = tf.cast(type_mask_tmp[:-1], "bool")
            tmp_return = tf.ensure_shape(tmp_return, self.conf.INPUT_RANGES[desired_type_string])
            return tmp_return

        else:
            # we need a rank 2 tensor to feed into "indices" of tensor_scatter_nd_max --> since we have a scalar, we expand_dims 2 times
            feature_index = tf.expand_dims(tf.expand_dims(song[
                tf.math.argmax(tf.math.multiply( # multiply for the range so that argmax takes the last one that has value 1 (the zeros remain 0 when you multiply them)
                    self.mask_tokens_if_type_is_desired_type(song, tmp_token_mask, desired_type),
                    tf.cast(tf.range(self.conf.SEQ_LEN-1), "float32")
                )),
                desired_position_in_token
            ], axis=-1), axis=-1)

            if not reverse_mask:
                simpler_type_mask_tmp = tf.zeros(self.conf.INPUT_RANGES[desired_type_string])
                simpler_type_mask_tmp = tf.tensor_scatter_nd_max(
                    tensor=simpler_type_mask_tmp,
                    indices=feature_index,
                    updates=tf.ones(1)
                )
            else:
                simpler_type_mask_tmp = tf.ones(self.conf.INPUT_RANGES[desired_type_string])
                simpler_type_mask_tmp = tf.tensor_scatter_nd_min(
                    tensor=simpler_type_mask_tmp,
                    indices=feature_index,
                    updates=tf.zeros(1)
                )

            tmp_return = tf.cast(simpler_type_mask_tmp, "bool")
            tmp_return = tf.ensure_shape(tmp_return, self.conf.INPUT_RANGES[desired_type_string])
            return tmp_return


    @tf.function
    def get_mask_for_each_token_in_elem(self, inputs): 
        '''
        Inputs:
        - idx:                  1
        - chosen_type:          1
        - song:                 (SEQ_LEN-1)*11
        - scores:               1391

        Returns a list of ndarrays of bool type used for masking
        Inputs are for a SINGLE ELEMENT OF A BATCH of size SEQ_LEN*(1+11+1391) where 1391 is the summed length of logits (minus the type)
        '''
        # Collect inputs from longer tensor
        idx, chosen_type, song = inputs
        song = tf.ensure_shape(song, (self.conf.SEQ_LEN-1, 11))

        default_mask = self.default_mask.copy()
        instruments_mask = default_mask[5]

        # Mask for tokens in the song until idx+1 (current index+1 because in the input space it's shifted)
        tmp_token_mask = tf.cast(tf.range(self.conf.SEQ_LEN-1) <= (idx + 1), dtype="float32")
        tmp_token_mask = tf.ensure_shape(tmp_token_mask, self.conf.SEQ_LEN-1)

        ## MAIN BODY ##
        if chosen_type == 0 or chosen_type == 2 or chosen_type == 7:
            return tf.concat(default_mask, axis=-1)

        elif chosen_type == 1: # Instrument selection, false only for type and instrument type (the ones that you can choose)
            instruments_masks_list = default_mask
            token_is_instrument = self.mask_tokens_if_type_is_desired_type(song, tmp_token_mask, 1)
            instruments_occurrences = tf.reduce_sum(token_is_instrument)
            if instruments_occurrences == 0:
                # Choice of first instrument
                instruments_masks_list[5] = self.full_mask[5]
                return tf.concat(instruments_masks_list, axis=-1)
            else:
                # Only mask the forbidden instruments, all the rest is default
                instruments_mask = self.get_mask_of_present_values_from_desired_type(
                    song, tmp_token_mask, 
                    desired_type=1, 
                    desired_type_string="instrument", 
                    desired_position_in_token=6,
                    reverse_mask=True)
                instruments_masks_list[5] = instruments_mask
                return tf.concat(instruments_masks_list, axis=-1)

        else:
            # All other types are free
            return tf.concat(self.full_mask, axis=-1) 


    def get_mask_for_one_elem_in_batch(self, inputs):
        '''
        Inputs:
        - chosen_types:         (SEQ_LEN-1)*1
        - song_tokens:          (SEQ_LEN-1)*11

        Returns a list of ndarrays of bool type used for masking
        Inputs are for a SINGLE ELEMENT OF A BATCH of size SEQ_LEN*(1+11+1391) 
        where 1391 is the summed length of logits (minus the type)
        '''
        # Collect inputs from longer tensor
        chosen_types, song_tokens = inputs
        chosen_types = tf.cast(chosen_types, dtype=tf.int32)
        song_tokens  = tf.cast(song_tokens , dtype=tf.int32)
        # Indexes
        mask = tf.vectorized_map(
            fn=self.get_mask_for_each_token_in_elem,
            elems=(
                tf.range(self.conf.SEQ_LEN-1),
                chosen_types,
                tf.expand_dims(song_tokens, axis=0),    # expand_dims = 0 allows to pass the song_tokens every time to vectorized_map
            )
        )
        mask = tf.ensure_shape(mask, (self.conf.SEQ_LEN-1, 1391), name="ensure_shape_mask_batch")
        return mask


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
        
        masks = tf.map_fn(
            fn=self.get_mask_for_one_elem_in_batch, 
            elems=(         # Iterate function over batch dimension 
                tf.cast(chosen_types, concat_logits.dtype),                 # BATCH*(SEQ_LEN-1)*1
                tf.cast(songs,   concat_logits.dtype),                      # BATCH*(SEQ_LEN-1)*11
            ),
            fn_output_signature=tf.TensorSpec(                              
                (
                    self.conf.SEQ_LEN-1, 
                    self.conf.input_ranges_sum - self.conf.INPUT_RANGES['type']
                ), dtype="bool"), 
            name="map_fn_complex"
        )
        return masks

####################################################################################################################

# Model creation function (to be called within a scope in case of MultiGPU training)
def create_model(conf:Config, input_shape=None, num_genres=None, 
                 use_regularization=True, use_masking_layers=True, 
                 reg_loss_scale=None):
    '''
    General model creation function. By default it generates a model which uses regulariazion and doesn't use
    masking, but this can be changed by manipulating the boolean flags `use_regularization` and `use_masking_layers`.
    It requires a `Config` object containing all hyperparameters of the model and optionally accepts an input shape, 
    the number of possible genres for classification, and a regularization scaling factor.
    '''
    # Defaults taken from conf object
    if input_shape is None:
        input_shape=(conf.SEQ_LEN-1, len(conf.INPUT_RANGES))
    if num_genres is None:
        num_genres=len(conf.accepted_subgenres)
    if reg_loss_scale is None:
        reg_loss_scale = conf.REG_LOSS_SCALE
    
    # Get input shapes
    seq_len = input_shape[0]
    events_elements = input_shape[1]
    
    # Instantiate transformer decoder (n_emb % n_head must be 0)
    decoder = conf.get_decoder()
    
    # Define inputs
    songs  = tf.keras.Input(shape=input_shape, name='songs',  dtype=tf.int32)
    genres = tf.keras.Input(shape=num_genres , name='genres', dtype=tf.float32)
    
    # Define loss
    loss_function = tf.keras.losses.SparseCategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE)

    # Regularization layers
    if use_regularization:
        subsequent_type_transform_layer = SubsequentTypeTransformationLayer(conf)
        misc_type_checker = MiscTypeChecker(conf)
        instruments_checker = InstrumentsChecker(conf)
        reg_scaler = tf.constant(reg_loss_scale, dtype=tf.float32)
    
    # Masking layers
    if use_masking_layers:
        type_masking_layer = MaskTypeProbabilitiesLayer(conf)
        activations_masking = MaskingActivationLayer(conf)
    
    # Embedding layers
    embedding_layers = [
        # Type embedding
        tf.keras.layers.Embedding(conf.INPUT_RANGES['type'],       conf.SINGLE_EMB_SIZE, input_length=conf.SEQ_LEN, name='type_embeddings'),
        # Measure embedding
        tf.keras.layers.Embedding(conf.INPUT_RANGES['measure'],    conf.SINGLE_EMB_SIZE, input_length=conf.SEQ_LEN, name='measure_embeddings'),
        # Beat embedding
        tf.keras.layers.Embedding(conf.INPUT_RANGES['beat'],       conf.SINGLE_EMB_SIZE, input_length=conf.SEQ_LEN, name='beat_embeddings'),
        # Position embedding
        tf.keras.layers.Embedding(conf.INPUT_RANGES['position'],   conf.SINGLE_EMB_SIZE, input_length=conf.SEQ_LEN, name='position_embeddings'),
        # Duration embedding
        tf.keras.layers.Embedding(conf.INPUT_RANGES['duration'],   conf.SINGLE_EMB_SIZE, input_length=conf.SEQ_LEN, name='duration_embeddings'),
        # Pitch embedding
        tf.keras.layers.Embedding(conf.INPUT_RANGES['pitch'],      conf.SINGLE_EMB_SIZE, input_length=conf.SEQ_LEN, name='pitch_embeddings'),
        # Instrument embedding
        tf.keras.layers.Embedding(conf.INPUT_RANGES['instrument'], conf.SINGLE_EMB_SIZE, input_length=conf.SEQ_LEN, name='instrument_embeddings'),
        # Velocity embedding
        tf.keras.layers.Embedding(conf.INPUT_RANGES['velocity'],   conf.SINGLE_EMB_SIZE, input_length=conf.SEQ_LEN, name='velocity_embeddings'),
        # Key sign embedding
        tf.keras.layers.Embedding(conf.INPUT_RANGES['key_sign'],   conf.SINGLE_EMB_SIZE, input_length=conf.SEQ_LEN, name='key_sign_embeddings'),
        # Time sign embedding
        tf.keras.layers.Embedding(conf.INPUT_RANGES['time_sign'],  conf.SINGLE_EMB_SIZE, input_length=conf.SEQ_LEN, name='time_sign_embeddings'),
        # Tempo embedding
        tf.keras.layers.Embedding(conf.INPUT_RANGES['tempo'],      conf.SINGLE_EMB_SIZE, input_length=conf.SEQ_LEN, name='tempo_embeddings')
    ]
    
    genre_embedding_layer = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(conf.GENRE_DIM)
    ], name='genre_embedding')
    
    # Input processing layers
    input_concat_layer         = tf.keras.layers.Concatenate(axis=2)
    sequence_concat_layer      = tf.keras.layers.Concatenate(axis=1)
    encoding_processing_layer  = tf.keras.layers.Dense(conf.TOKEN_DIM, name='encoding_processing')
    
    # Absolute positional encoding for GPT model
    if conf.model_type == 'GPT':
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
        tf.keras.layers.Softmax(name='keysign_probabilities'),
        # Time sign
        tf.keras.layers.Softmax(name='timesign_probabilities'),
        # Tempo
        tf.keras.layers.Softmax(name='tempo_probabilities')
    ]
    
    # Model dynamics
    embeddings        = [embedding_layers[i](songs[:,:,i]) for i in range(events_elements)]
    genre_embedding   = genre_embedding_layer(genres)
    input_embedding   = input_concat_layer(embeddings)
    input_embedding   = encoding_processing_layer(input_embedding)
    input_embedding   = sequence_concat_layer([genre_embedding[:, np.newaxis, :], input_embedding])
    if conf.model_type == 'GPT':
        input_embedding   = sum_layer([input_embedding, positional_encoding])
    model_output      = decoder({'inputs_embeds': input_embedding})['last_hidden_state']
    out_scores        = [output_dense_layers[i](model_output)[:,:-1,:] 
                         for i in range(len(output_dense_layers))]
    
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
        for i, key in enumerate(conf.INPUT_RANGES)
    }

    # Create model
    model = tf.keras.Model(inputs=[songs, genres], 
                           outputs=out_probabilities_dict, 
                           name='music_generation_model')
    
    # Before computing losses, mask probabilities so that nothing after the first 7
    # in the original song counts.
    @tf.function
    def find_type_7(songs):
        mask = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        for i in tf.range(tf.shape(songs)[0]):
            end_song_idx = tf.math.reduce_min(tf.where(songs[i, :, 0] == 7))
            mask = mask.write(i, tf.concat([
                tf.ones(end_song_idx), 
                tf.zeros(conf.SEQ_LEN - 1 - end_song_idx)], axis=-1))
        return mask.stack()

    end_song_mask = tf.keras.layers.Lambda(find_type_7)(songs)
    end_song_mask = tf.cast(end_song_mask, tf.bool)
    
    # Define loss
    def custom_loss(y_true, y_pred):
        return tf.math.reduce_sum(
            loss_function(y_true, y_pred) * \
            (1. / (conf.GLOBAL_BATCH_SIZE * tf.cast(tf.shape(y_true)[0], tf.float32)))
        )
    
    # Define regularizers
    def custom_regularizers(y_pred):
        # Regularization loss: transform the actual vectors into consecutive-type representation
        max_pred_types = tf.argmax(y_pred[0], axis=2, output_type=tf.int32)
        
        ####### 0: MISC CONSTRAINTS ABOUT TOKEN TYPES ORDER #######
        reg_term_0 = misc_type_checker(max_pred_types) * 20   # *20 to keep it comparable to other losses
        
        ####### 1: PUNISHMENT FOR NON-CONSECUTIVE TYPES ##########
        consecutive_pred_types = subsequent_type_transform_layer(max_pred_types)
        # Compute difference
        differences = consecutive_pred_types[:, 1:] - consecutive_pred_types[:, :-1]
        # Compute regularization terms
        # Difference between one element's type and the next is >= 0
        reg_term_1_1 = tf.math.reduce_sum(tf.math.maximum(0, -differences))
        # Difference between one element's type and the next is < 1
        reg_term_1_2 = tf.math.reduce_sum(tf.math.maximum(0, tf.math.maximum(1, differences) - 1))  
        reg_term_1 = reg_term_1_1 + reg_term_1_2
        
        ####### 2: PUNISHMENT FOR NOTES WHOSE INSTRUMENT IS NOT DEFINED AND FOR DUPLICATE INSTRUMENTS ########
        reg_term_2 = instruments_checker([max_pred_types, y_pred[6]])
        
        ####### 3: PUNISHMENT FOR CONSECUTIVE EVENTS WITH NON-INCREASING TIMINGS ########
        # Get the predicted measures, beats and positions
        max_pred_measures = tf.argmax(y_pred[1], axis=2, output_type=tf.int32)
        max_pred_beats = tf.argmax(y_pred[2], axis=2, output_type=tf.int32)
        max_pred_positions = tf.argmax(y_pred[3], axis=2, output_type=tf.int32)
        # Use them to compute the "times" matrix
        times = max_pred_measures*conf.INPUT_RANGES['beat']*conf.INPUT_RANGES['position'] + \
            max_pred_beats*conf.INPUT_RANGES['position'] + \
            max_pred_positions
        # Normalize times
        times = times / ((conf.INPUT_RANGES['measure']+1)*conf.INPUT_RANGES['beat']*conf.INPUT_RANGES['position'])
        # Only consider the time matrix when the type is between 3 and 6
        times = tf.cast(tf.where(tf.logical_and(max_pred_types >= 3, max_pred_types <= 6), times, 0), tf.float32)
        # For type 7 fill with a very large value
        times = tf.where(max_pred_types == 7, 1e10, times)
        # Compute time differences between consecutive time steps
        time_sep = times[:, 1:] - times[:, :-1]
        # Count negative time seps
        reg_term_3 = tf.math.reduce_sum(tf.cast(time_sep < 0, tf.int32))
        
        ####### PUT TOGETHER THE REGULARIZATION TERMS #######
        return tf.math.reduce_sum(
            reg_scaler * ((tf.cast(reg_term_0, tf.float32)) + (tf.cast(reg_term_1, tf.float32)) + \
                          (tf.cast(reg_term_2, tf.float32)) + (tf.cast(reg_term_3, tf.float32)))
        )
    
    # Add losses
    for i, k in enumerate(conf.INPUT_RANGES):
        loss_name = f'{k}_loss'
        gt = tf.boolean_mask(songs[:,:,i], end_song_mask)
        pred = tf.boolean_mask(out_probabilities[i], end_song_mask)
        loss = custom_loss(y_true = gt, y_pred = pred)
        model.add_loss(loss)
        model.add_metric(loss, name=loss_name)
    
    if use_regularization:
        # Note: we don't mask in regularization, because we don't use a ground truth
        # Here we just make the model learn how to produce a syntactically good output.
        reg_loss = custom_regularizers(out_probabilities)
        model.add_loss(reg_loss)
        model.add_metric(reg_loss, name='regularization_loss')
    
    # Compile and return
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=conf.LEARNING_RATE))
    return model