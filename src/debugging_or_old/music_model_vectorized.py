import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision

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
                tf.range(conf.SEQ_LEN-1),
                i+1
            ), dtype="float32")
            tmp_token_mask = tf.ensure_shape(tmp_token_mask, conf.SEQ_LEN-1)
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
                tf.range(conf.SEQ_LEN-1),
                tf.expand_dims(batch_gt_types, axis=0) # if first_dim = 1 --> passes always the same tensor
            ),        
        )

        return tf.ensure_shape(mask, (conf.SEQ_LEN-1, 8))


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
                    tf.TensorSpec(shape=(conf.SEQ_LEN-1, 8), dtype="bool")
                ),
                # parallel_iterations=batch_size
            )

            """ 

            masks = tf.TensorArray(
                dtype="bool",
                size=batch_size,
                clear_after_read=True,
                element_shape=(conf.SEQ_LEN-1, 8)
            )

            for i in range(batch_size):
                masks = masks.write(
                    index=i,
                    value=self.mask_single_song_in_batch(gt_types[i, :])
                )

            masks = masks.stack()
             """

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


    def mask_tokens_if_type_is_desired_type(self, song, tmp_token_mask, desired_type):
        # useful if chosen_type == 1 and if chosen_type==3
        # 1 if token is instrument, 0 otherwise
        token_is_desired_type = tf.math.multiply(
            tf.cast(tf.math.equal(song[:,0], desired_type), dtype="float32"),
            tmp_token_mask
        )
        token_is_desired_type = tf.ensure_shape(token_is_desired_type, conf.SEQ_LEN-1)
        return token_is_desired_type


    def get_mask_of_present_values_from_desired_type(self, 
            song, tmp_token_mask, 
            desired_type, desired_type_string, desired_position_in_token, 
            reverse_mask=False,
            only_last=False
        ):
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
                y=conf.INPUT_RANGES[desired_type_string]    # a scalar = last feature index + 1
                # y=-1    # a scalar = -1
            ), dtype="int32"), axis=-1) # need to be int32/64 to be inside indices of scatter_nd 

            if not reverse_mask:
                # create a tensor with one more spot than needed
                type_mask_tmp = tf.zeros(conf.INPUT_RANGES[desired_type_string] + 1)

                # for each index in the indexes_masked tensor above, put 1 in the mask
                # many of the values of the above tensor will be the fake index
                # It the end, we should have 1 for instruments that are defined (and can be chosen), 
                # and 0 for the ones that are not defined (and cannot be chosen)
                type_mask_tmp = tf.tensor_scatter_nd_max(
                    tensor=type_mask_tmp,
                    indices=indexes_masked,
                    updates=tf.ones(conf.SEQ_LEN-1)
                )

            else:
                # ones instead of zeros
                type_mask_tmp = tf.ones(conf.INPUT_RANGES[desired_type_string] + 1)
                # min instead of max
                type_mask_tmp = tf.tensor_scatter_nd_min(
                    tensor=type_mask_tmp,
                    indices=indexes_masked,
                    # zeros instead of ones
                    updates=tf.zeros(conf.SEQ_LEN-1)
                )

            # Remove the last fake token
            tmp_return = tf.cast(type_mask_tmp[:-1], "bool")
            tmp_return = tf.ensure_shape(tmp_return, conf.INPUT_RANGES[desired_type_string])
            return tmp_return

        else:
            # we need a rank 2 tensor to feed into "indices" of tensor_scatter_nd_max --> since we have a scalar, we expand_dims 2 times
            feature_index = tf.expand_dims(tf.expand_dims(song[
                tf.math.argmax(tf.math.multiply( # multiply for the range so that argmax takes the last one that has value 1 (the zeros remain 0 when you multiply them)
                    self.mask_tokens_if_type_is_desired_type(song, tmp_token_mask, desired_type),
                    tf.cast(tf.range(conf.SEQ_LEN-1), "float32")
                )),
                desired_position_in_token
            ], axis=-1), axis=-1)

            if not reverse_mask:
                simpler_type_mask_tmp = tf.zeros(conf.INPUT_RANGES[desired_type_string])
                simpler_type_mask_tmp = tf.tensor_scatter_nd_max(
                    tensor=simpler_type_mask_tmp,
                    indices=feature_index,
                    updates=tf.ones(1)
                )
            else:
                simpler_type_mask_tmp = tf.ones(conf.INPUT_RANGES[desired_type_string])
                simpler_type_mask_tmp = tf.tensor_scatter_nd_min(
                    tensor=simpler_type_mask_tmp,
                    indices=feature_index,
                    updates=tf.zeros(1)
                )

            tmp_return = tf.cast(simpler_type_mask_tmp, "bool")
            tmp_return = tf.ensure_shape(tmp_return, conf.INPUT_RANGES[desired_type_string])
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
        idx, chosen_type, song, scores = inputs
        song = tf.ensure_shape(song, (conf.SEQ_LEN-1, 11))
        default_mask = self.default_mask.copy()

        ## SETUP ##
        # Define the variables and flags (to avoid ValueError: 'variable' must also be initialized in the main branch)
        # min_measure           = tf.constant(0, dtype=tf.int32)
        # min_beat              = tf.constant(0, dtype=tf.int32)
        # min_position          = tf.constant(0, dtype=tf.int32)

        # min_measure_check     = tf.constant(False, dtype=tf.bool)

        # measure_num           = tf.constant(conf.INPUT_RANGES['measure'], tf.int32)
        # position_num          = tf.constant(conf.INPUT_RANGES["position"], tf.int32)
        # beat_num              = tf.constant(conf.INPUT_RANGES["beat"], tf.int32)

        # measure_mask, beat_mask,      \
        # position_mask, duration_mask, \
        # pitch_mask, instruments_mask, \
        # velocity_mask, key_sign_mask, \
        # time_sign_mask, tempo_mask    = default_mask
        instruments_mask = default_mask[5]

        # Mask for tokens in the song until idx+1 (current index+1 because in the input space it's shifted)
        tmp_token_mask = tf.cast(tf.range(conf.SEQ_LEN-1) <= (idx + 1), dtype="float32")
        tmp_token_mask = tf.ensure_shape(tmp_token_mask, conf.SEQ_LEN-1)

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
            return tf.concat(self.full_mask, axis=-1) 

        # else:   # Types 3 to 6
        #     if chosen_type == 3:
        #         # min_measure = song[idx, 1]   # It has to be >= than the last measure
        #         # min_measure = tf.ensure_shape(min_measure, tf.TensorShape(()))
        #         # If in the MEASURE SCORES the MAX SCORE between all possible measures == min_measure, the measure is min_measure.
        #         # In this case, we need to make sure that beat >= last_beat

        #         # min_measure_check = tf.math.argmax(scores_measure, output_type=tf.int32) == min_measure
        #         # tf.print(tf.math.argmax(scores_measure, output_type=tf.int32))
        #         # tf.print(tf.shape(tf.math.argmax(scores_measure, output_type=tf.int32)))
        #         # tf.print(min_measure)
        #         # tf.print(tf.shape(min_measure))
        #         # tf.print(min_measure_check)
        #         # tf.print(tf.shape(min_measure_check))
                
        #         # if False:
        #         #     min_beat = song[idx, 2] # It has to be >= than the last beat when measure is the same
        #         #     min_beat = tf.ensure_shape(min_beat, tf.TensorShape(()))
        #         # else:
        #         #     min_beat = tf.constant(0, tf.int32)

        #         # if  False: # min_measure_check and tf.math.argmax(
        #         #         # tf.slice(scores, [measure_num], [beat_num]), output_type=tf.int32) == min_beat:
        #         #     min_position = song[idx, 3] # It has to be >= than the last position (if beat and measure are the same)
        #         #     min_position = tf.ensure_shape(min_position, tf.TensorShape(()))
        #         # else:
        #         #     min_position = tf.constant(0, tf.int32)

        #         # tf.print(min_beat)
        #         # tf.print(min_position)
        #         # tf.print("---")

        #         # Only some instruments, key signs, time signs and tempos are allowed for these events: 
        #         # - for instruments, the allowed ones are the ones that have been defined previously with type = 1
        #         # - for the others, the allowed ones are the ones that are collected right before the note from event types 4, 5 and 6
                
        #         instruments_mask = self.get_mask_of_present_values_from_desired_type(
        #             song, tmp_token_mask, 
        #             desired_type=1, 
        #             desired_type_string="instrument", 
        #             desired_position_in_token=6,
        #             reverse_mask=False,
        #             only_last=False
        #         )
        #         instruments_mask = tf.ensure_shape(instruments_mask, conf.INPUT_RANGES["instrument"])

        #         # We have made it so that the model should output 3s only after at least a 4, 5 and 6.
        #         key_sign_mask = self.get_mask_of_present_values_from_desired_type(
        #             song, tmp_token_mask,
        #             desired_type=4,
        #             desired_type_string="key_sign",
        #             desired_position_in_token=8,
        #             reverse_mask=False,
        #             only_last=True
        #         )
        #         key_sign_mask = tf.ensure_shape(key_sign_mask, conf.INPUT_RANGES["key_sign"])

        #         time_sign_mask = self.get_mask_of_present_values_from_desired_type(
        #             song, tmp_token_mask,
        #             desired_type=5,
        #             desired_type_string="time_sign",
        #             desired_position_in_token=9,
        #             reverse_mask=False,
        #             only_last=True
        #         )
        #         time_sign_mask = tf.ensure_shape(time_sign_mask, conf.INPUT_RANGES["time_sign"])

        #         tempo_mask = self.get_mask_of_present_values_from_desired_type(
        #             song, tmp_token_mask,
        #             desired_type=6,
        #             desired_type_string="tempo",
        #             desired_position_in_token=10,
        #             reverse_mask=False,
        #             only_last=True
        #         )
        #         tempo_mask = tf.ensure_shape(tempo_mask, conf.INPUT_RANGES["tempo"])

        #         # max_beat = self.get_max_beat_from_time_sign(tf.argmax(time_sign_mask)) # max_beat is the numerator of time_sign_mask
        #         # # Allowed beats are only AFTER previous beat and BEFORE max_beat from the numerator of the time_sign
        #         # beat_mask = tf.math.logical_and(
        #         #     tf.range(beat_num) >= min_beat,
        #         #     tf.range(beat_num) < max_beat)
        #         # # beat_mask = tf.ones(conf.INPUT_RANGES['beat'], dtype=tf.bool)
        #         # beat_mask = tf.ensure_shape(beat_mask, conf.INPUT_RANGES["beat"])

        #         # position_mask = tf.range(position_num) >= min_position
        #         # # position_mask = tf.ones(conf.INPUT_RANGES['position'], dtype=tf.bool)
        #         # position_mask = tf.ensure_shape(position_mask, conf.INPUT_RANGES["position"])

        #         measure_mask  = self.full_mask[0]
        #         measure_mask  = tf.ensure_shape(measure_mask, conf.INPUT_RANGES["measure"])

        #         beat_mask     = self.full_mask[1]
        #         beat_mask     = tf.ensure_shape(beat_mask, conf.INPUT_RANGES["beat"])

        #         position_mask = self.full_mask[2]
        #         position_mask = tf.ensure_shape(position_mask, conf.INPUT_RANGES["position"])

        #         # measure_mask = tf.range(measure_num) >= min_measure
        #         # measure_mask = tf.ensure_shape(measure_mask, conf.INPUT_RANGES['measure'])

        #         tf.print(tf.shape(measure_mask)) 
        #         tf.print(tf.shape(beat_mask)) 
        #         tf.print(tf.shape(position_mask)) 
        #         tf.print(tf.shape(duration_mask))
        #         tf.print(tf.shape(pitch_mask)) 
        #         tf.print(tf.shape(instruments_mask)) 
        #         tf.print(tf.shape(velocity_mask)) 
        #         tf.print(tf.shape(key_sign_mask))
        #         tf.print(tf.shape(time_sign_mask)) 
        #         tf.print(tf.shape(tempo_mask))

        #         tmp_return = tf.concat([
        #             measure_mask, 
        #             beat_mask, 
        #             position_mask, 
        #             duration_mask,
        #             pitch_mask, 
        #             instruments_mask, 
        #             velocity_mask, 
        #             key_sign_mask,
        #             time_sign_mask, 
        #             tempo_mask
        #         ], axis=-1, name="concat_3456")

        #         tf.print(tf.shape(tmp_return))

        #     # else:
        #     #     # Type between 4 and 6: key_sign, time_sign, tempo
        #     #     # If last event is at the beginning of a measure, you can add an event at the same time
        #     #     if song[idx, 2] == 0 and song[idx, 3] == 0:  # if beat and position == 0, the event can be at this measure
        #     #         min_measure = song[idx, 1]
        #     #     else:
        #     #         min_measure = song[idx, 1] + 1           # otherwise it goes to the next measure
            
        #     #     # Fine-grain checks

        #     #     # Here, there are cases where there is not a LAST key_sign/time_sign (when this is the first 4, 5 or 6). 
        #     #     # In these cases we should use the default masks.
        #     #     if chosen_type == 4:
        #     #         # Cannot put the same key_sign again
        #     #         token_is_key_sign = self.mask_tokens_if_type_is_desired_type(song, tmp_token_mask, 4)
        #     #         key_sign_occurrences = tf.reduce_sum(token_is_key_sign)

        #     #         if key_sign_occurrences == 0:
        #     #             key_sign_mask = tf.cast(tf.ones(conf.INPUT_RANGES["key_sign"]), "bool")
        #     #         else:
        #     #             key_sign_mask = self.get_mask_of_present_values_from_desired_type(
        #     #                 song, tmp_token_mask,
        #     #                 desired_type=4,
        #     #                 desired_type_string="key_sign",
        #     #                 desired_position_in_token=8,
        #     #                 reverse_mask=True, ###
        #     #                 only_last=True
        #     #             )
        #     #             key_sign_mask = tf.ensure_shape(key_sign_mask, conf.INPUT_RANGES["key_sign"])
                        
        #     #     elif chosen_type == 5:
        #     #         # Cannot put the same time_sign again
        #     #         token_is_time_sign = self.mask_tokens_if_type_is_desired_type(song, tmp_token_mask, 5)
        #     #         time_sign_occurrences = tf.reduce_sum(token_is_time_sign)

        #     #         if time_sign_occurrences == 0:
        #     #             time_sign_mask = tf.cast(tf.ones(conf.INPUT_RANGES["time_sign"]), "bool")
        #     #         else:
        #     #             time_sign_mask = self.get_mask_of_present_values_from_desired_type(
        #     #                 song, tmp_token_mask,
        #     #                 desired_type=5,
        #     #                 desired_type_string="time_sign",
        #     #                 desired_position_in_token=9,
        #     #                 reverse_mask=True, ###
        #     #                 only_last=True
        #     #             )
        #     #             time_sign_mask = tf.ensure_shape(time_sign_mask, conf.INPUT_RANGES["time_sign"])

        #     #     elif chosen_type == 6:
        #     #         # Cannot put the same tempo again
        #     #         token_is_tempo = self.mask_tokens_if_type_is_desired_type(song, tmp_token_mask, 6)
        #     #         tempo_occurrences = tf.reduce_sum(token_is_tempo)

        #     #         if tempo_occurrences == 0:
        #     #             tempo_mask = tf.cast(tf.ones(conf.INPUT_RANGES["tempo"]), "bool")
        #     #         else:
        #     #             tempo_mask = self.get_mask_of_present_values_from_desired_type(
        #     #                 song, tmp_token_mask,
        #     #                 desired_type=6,
        #     #                 desired_type_string="tempo",
        #     #                 desired_position_in_token=10,
        #     #                 reverse_mask=True, ###
        #     #                 only_last=True
        #     #             )
        #     #             tempo_mask = tf.ensure_shape(tempo_mask, conf.INPUT_RANGES["tempo"])

        #         return tmp_return
        #         # return tf.concat([
        #         #     measure_mask, 
        #         #     beat_mask, 
        #         #     position_mask, 
        #         #     duration_mask,
        #         #     pitch_mask, 
        #         #     instruments_mask, 
        #         #     velocity_mask, 
        #         #     key_sign_mask,
        #         #     time_sign_mask, 
        #         #     tempo_mask
        #         # ], axis=-1, name="concat_3456")
        #     else:
        #         return tf.concat(self.full_mask, axis=-1)


    def get_mask_for_one_elem_in_batch(self, inputs):
        '''
        Inputs:
        - chosen_types:         (SEQ_LEN-1)*1
        - song_tokens:          (SEQ_LEN-1)*11
        - seq_scores:           (SEQ_LEN-1)*1391

        Returns a list of ndarrays of bool type used for masking
        Inputs are for a SINGLE ELEMENT OF A BATCH of size SEQ_LEN*(1+11+1391) 
        where 1391 is the summed length of logits (minus the type)
        '''
        # Collect inputs from longer tensor
        chosen_types, song_tokens, seq_scores = inputs
        chosen_types = tf.cast(chosen_types, dtype=tf.int32)
        song_tokens  = tf.cast(song_tokens , dtype=tf.int32)
        # seq_scores_measure = seq_scores[:, :conf.INPUT_RANGES['measure']]
        # seq_scores_beat = seq_scores[:, conf.INPUT_RANGES['measure'] : 
        #                                 conf.INPUT_RANGES['measure'] + conf.INPUT_RANGES['beat']]
        # Indexes
        mask = tf.vectorized_map(
            fn=self.get_mask_for_each_token_in_elem,
            elems=(
                tf.range(conf.SEQ_LEN-1),
                chosen_types,
                tf.expand_dims(song_tokens, axis=0),    # expand_dims = 0 allows to pass the song_tokens every time to vectorized_map
                seq_scores
            )
        )

        mask = tf.ensure_shape(mask, (conf.SEQ_LEN-1, 1391), name="ensure_shape_mask_batch")
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
                concat_logits                                               # BATCH*(SEQ_LEN-1)*1391 
            ),                                                              # Total: a BATCH * SEQ_LEN-1 * 1403 tensor
            fn_output_signature=tf.TensorSpec(                              
                (
                    conf.SEQ_LEN-1, 
                    conf.input_ranges_sum - conf.INPUT_RANGES['type']
                ), dtype="bool"), 
            name="map_fn_complex"
        )

        return masks


#################################################################################
def create_debugging_model(song_shape, decoder_output_shape):
    
    input_layer = tf.keras.Input(shape=decoder_output_shape)
    songs  = tf.keras.Input(shape=song_shape, name='songs',  dtype=tf.int32)

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
    
    type_masking_layer  = MaskTypeProbabilitiesLayer()
    activations_masking =  MaskingActivationLayer()

    out_scores        = [output_dense_layers[i](input_layer)[:,:-1,:] 
                         for i in range(len(output_dense_layers))] # [BATCH_SIZE * SEQ_LEN-1 * events_elements] * INPUT_RANGES (various)

    # We don't care about the last scores, since they refer to a token that's out of bounds.
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
        for i in range(1, len(output_probs_layers))]
    
    out_probabilities_dict = {
        key: out_probabilities[i] 
        for i, key in enumerate(conf.INPUT_RANGES.keys())
    }
    
    # Create model
    model = tf.keras.Model(
        inputs=[input_layer, songs], 
        outputs=out_probabilities_dict, 
        name='debugging_model'
    )

    return model
