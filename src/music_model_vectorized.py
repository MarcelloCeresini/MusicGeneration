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

            tf.ensure_shape(tmp_token_mask, conf.SEQ_LEN-1)

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

        tf.ensure_shape(type_mask, 8)

        return type_mask


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

        tf.ensure_shape(mask, (conf.SEQ_LEN-1, 8))

        return mask


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
        tf.ensure_shape(token_is_desired_type, conf.SEQ_LEN-1)

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
        
        ex. if desired_type == 1:
        it will return a mask conf.INPUT_RANGES["instrument"] long with 1 if an instrument is present in the song, 0 otherwise
        
        if reverse_mask=True, the 0s become 1s and vice-versa


        Inputs:
        - song:                         6143*11     all tokens
        - tmp_token_mask:               6143        1 if token is <= i+1, 0 otherwise
        - desired_type:                 1           in [0,7], get only the tokens of desired type
        - desired_type_string           1           string used to get the shape of the final mask (ex. "instrument" mask is 129 long)
        - desired_position_in_token     1           where to look in the size 11 token (ex. instruments are defined in position 6)
        - reverse_mask                  1           if True, return the complementary mask

        Returns:
        - mask                          INPUT_LENGTH["desired_type_string"] 
        the mask has 1 if the tokens UP TO i+1 contain that specific feature value (ex. contain a specific instrument) and 0 otherwise
        """

        token_is_desired_type = self.mask_tokens_if_type_is_desired_type(song, tmp_token_mask, desired_type)

        if not only_last:
            # need to be bool to be put as the condition of a tf.where
            token_is_desired_type = tf.cast(token_is_desired_type, dtype="bool")

            # if there is a token of desired_type, save its value, otherwise put last "fake" index
            indexes_masked = tf.expand_dims(tf.cast(tf.where(
                condition=token_is_desired_type,
                x=song[:,desired_position_in_token],        # a vector of feature indexes
                # y=conf.INPUT_RANGES[desired_type_string]    # a scalar = last feature index + 1
                y=-1    # a scalar = -1
            ), dtype="int32"), axis=-1) # need to be int32/64 to be inside indices of scatter_nd 

            if not reverse_mask:
                # create a tensor with one more spot than needed
                type_mask_tmp = tf.zeros(conf.INPUT_RANGES[desired_type_string])

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
                type_mask_tmp = tf.ones(conf.INPUT_RANGES[desired_type_string])
                # min instead of max
                type_mask_tmp = tf.tensor_scatter_nd_min(
                    tensor=type_mask_tmp,
                    indices=indexes_masked,
                    # zeros instead of ones
                    updates=tf.zeros(conf.SEQ_LEN-1)
                )

            # remove the last fake token
            return tf.cast(type_mask_tmp, "bool")

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
            
            return tf.cast(simpler_type_mask_tmp, "bool")


    @tf.function
    def get_mask_for_each_token_in_elem(self, inputs): 
        '''
        Inputs:
        - idx:                  1
        - chosen_type:          1
        - song:          (SEQ_LEN-1)*11
        - scores:               1391

        Returns a list of ndarrays of bool type used for masking
        Inputs are for a SINGLE ELEMENT OF A BATCH of size SEQ_LEN*(1+11+1391) where 1391 is the summed length of logits (minus the type)
        '''

        # Collect inputs from longer tensor
        idx, chosen_type, song, scores = inputs

        ## SETUP ##
        # Define the variables and flags (to avoid ValueError: 'variable' must also be initialized in the main branch)
        default_token_parts   = [True, True, True, True, True, True, True, True, True, True]
        min_measure           = tf.constant(-1, dtype=tf.int32)
        min_beat              = tf.constant(-1, dtype=tf.int32)
        min_position          = tf.constant(-1, dtype=tf.int32)

        measure_mask = self.default_mask[0]
        beat_mask = self.default_mask[1]
        position_mask = self.default_mask[2]
        duration_mask = self.default_mask[3]
        pitch_mask = self.default_mask[4]
        instruments_mask = self.default_mask[5]
        velocity_mask = self.default_mask[6]
        key_sign_mask = self.default_mask[7]
        time_sign_mask = self.default_mask[8]
        tempo_mask = self.default_mask[9]

        default_flag                = False
        forbidden_instruments_flag  = False
        
        # mask tokens until idx+1 (current index+1 because in the input space it's shifted)
        tmp_token_mask = tf.cast(
            tf.less_equal(
                tf.range(conf.SEQ_LEN-1),
                idx+1
            ), dtype="float32"
        )
        tf.ensure_shape(tmp_token_mask, conf.SEQ_LEN-1)

        ## MAIN BODY ##
        if chosen_type == 0 or chosen_type == 2 or chosen_type == 7:
            # tf.print("chosen_type 027")
            default_token_parts = [True, True, True, True, True, True, True, True, True, True]
            default_flag = True

        elif chosen_type == 1: # Instrument selection, false only for type and instrument type (the ones that you can choose)
            # tf.print("chosen_type 1")
            token_is_instrument = self.mask_tokens_if_type_is_desired_type(song, tmp_token_mask, 1)
            instruments_occurrences = tf.reduce_sum(token_is_instrument)
            if instruments_occurrences == 0:
                # Choice of first instrument
                default_token_parts = [True, True, True, True, True, False, True, True, True, True]
                default_flag = True
            else:
                instruments_mask = self.get_mask_of_present_values_from_desired_type(
                    song, tmp_token_mask, 
                    desired_type=1, 
                    desired_type_string="instrument", 
                    desired_position_in_token=6,
                    reverse_mask=True)
                tf.ensure_shape(instruments_mask, conf.INPUT_RANGES["instrument"])
                forbidden_instruments_flag = True

        elif chosen_type == 3: # Notes: They have the same key_sign, time_sign and tempo as last previous event, everything has to be manually decided
            # tf.print("chosen_type 3")
            min_measure = song[idx, 1]   # It has to be >= than the last measure
            # min_measure = tf.reshape(tf.gather_nd(
            #     params=song,
            #     indices=[[idx, 1]],
            #     name="gather_min_measure_first"
            # ), shape=[])

            # tf.print("min_measure_shape", min_measure.shape)
            # tf.print("min_measure", min_measure)
            # If in the MEASURE SCORES the MAX SCORE between all possible measures == min_measure, the measure is min_measure.
            # In this case, we need to make sure that beat >= last_beat
            if tf.math.argmax(scores[:conf.INPUT_RANGES["measure"]], output_type=tf.int32) == min_measure:  
                
                min_beat = song[idx, 2] # It has to be >= than the last beat when measure is the same

                if tf.math.argmax(scores[
                        conf.INPUT_RANGES["measure"] : 
                        conf.INPUT_RANGES["measure"] + conf.INPUT_RANGES["beat"]
                    ], output_type=tf.int32) == min_beat:

                    min_position = song[idx, 3] # It has to be >= than the last position (if beat and measure are the same)

                else:
                    min_position = tf.constant(0, dtype=tf.int32)
            else:
                min_beat = tf.constant(0, dtype=tf.int32)
                min_position = tf.constant(0, dtype=tf.int32)

            # Only some instruments, key signs, time signs and tempos are allowed for these events: 
            # - for instruments, the allowed ones are the ones that have been defined previously with type = 1
            # - for the others, the allowed ones are the ones that are collected right before the note from event types 4, 5 and 6
            
            instruments_mask = self.get_mask_of_present_values_from_desired_type(
                song, tmp_token_mask, 
                desired_type=1, 
                desired_type_string="instrument", 
                desired_position_in_token=6,
                reverse_mask=False,
                only_last=False
            )
            tf.ensure_shape(instruments_mask, conf.INPUT_RANGES["instrument"])
            

            # We have made it so that the model should output 3s only after at least a 4, 5 and 6.
            key_sign_mask = self.get_mask_of_present_values_from_desired_type(
                song, tmp_token_mask,
                desired_type=4,
                desired_type_string="key_sign",
                desired_position_in_token=8,
                reverse_mask=False,
                only_last=True
            )
            tf.ensure_shape(key_sign_mask, conf.INPUT_RANGES["key_sign"])

            time_sign_mask = self.get_mask_of_present_values_from_desired_type(
                song, tmp_token_mask,
                desired_type=5,
                desired_type_string="time_sign",
                desired_position_in_token=9,
                reverse_mask=False,
                only_last=True
            )
            tf.ensure_shape(time_sign_mask, conf.INPUT_RANGES["time_sign"])

            tempo_mask = self.get_mask_of_present_values_from_desired_type(
                song, tmp_token_mask,
                desired_type=6,
                desired_type_string="tempo",
                desired_position_in_token=10,
                reverse_mask=False,
                only_last=True
            )
            tf.ensure_shape(tempo_mask,conf.INPUT_RANGES["tempo"])



        elif chosen_type >= 4 and chosen_type <= 6:     # key_sign, time_sign, tempo
            # tf.print("chosen_type 456")
            # If last event is at the beginning of a measure, you can add an event at the same time
            if song[idx, 2] == 0 and song[idx, 3] == 0:  # if beat and position == 0, the event can be at this measure
                min_measure = song[idx, 1]
            else:
                min_measure = song[idx, 1] + 1                   # otherwise it goes to the next measure
            
            # Fine-grain checks
            # Here, there are cases where there is not a LAST key_sign/time_sign (when this is the first 4, 5 or 6). 
            # In these cases we should use the default masks.
            if chosen_type == 4:
                # Cannot put the same key_sign again
                token_is_key_sign = self.mask_tokens_if_type_is_desired_type(song, tmp_token_mask, 4)
                key_sign_occurrences = tf.reduce_sum(token_is_key_sign)

                if key_sign_occurrences == 0:
                    key_sign_mask = tf.cast(tf.ones(conf.INPUT_RANGES["key_sign"]), "bool")
                else:
                    key_sign_mask = self.get_mask_of_present_values_from_desired_type(
                        song, tmp_token_mask,
                        desired_type=4,
                        desired_type_string="key_sign",
                        desired_position_in_token=8,
                        reverse_mask=True, ###
                        only_last=True
                    )
                    tf.ensure_shape(key_sign_mask, conf.INPUT_RANGES["key_sign"])
                    
            elif chosen_type == 5:
                # Cannot put the same time_sign again
                token_is_time_sign = self.mask_tokens_if_type_is_desired_type(song, tmp_token_mask, 5)
                time_sign_occurrences = tf.reduce_sum(token_is_time_sign)

                if time_sign_occurrences == 0:
                    time_sign_mask = tf.cast(tf.ones(conf.INPUT_RANGES["time_sign"]), "bool")
                else:
                    time_sign_mask = self.get_mask_of_present_values_from_desired_type(
                        song, tmp_token_mask,
                        desired_type=5,
                        desired_type_string="time_sign",
                        desired_position_in_token=9,
                        reverse_mask=True, ###
                        only_last=True
                    )
                    tf.ensure_shape(time_sign_mask, conf.INPUT_RANGES["time_sign"])

            elif chosen_type == 6:
                # Cannot put the same tempo again
                token_is_tempo = self.mask_tokens_if_type_is_desired_type(song, tmp_token_mask, 6)
                tempo_occurrences = tf.reduce_sum(token_is_tempo)

                if tempo_occurrences == 0:
                    tempo_mask = tf.cast(tf.ones(conf.INPUT_RANGES["tempo"]), "bool")
                else:
                    tempo_mask = self.get_mask_of_present_values_from_desired_type(
                        song, tmp_token_mask,
                        desired_type=6,
                        desired_type_string="tempo",
                        desired_position_in_token=10,
                        reverse_mask=True, ###
                        only_last=True
                    )
                    tf.ensure_shape(tempo_mask, conf.INPUT_RANGES["tempo"])


        ## ENDING PART ##
        # Put together the masks
        if default_flag:
            # No manual masking required, either "can freely choose this part of the token" (True) or 
            # "can only choose default for this part of the token" (False)
            
            tmp_return = tf.concat([self.default_mask[i] if default_token_parts[i] else self.full_mask[i] 
                for i in range(len(default_token_parts))], axis=-1, name="concat_default")
            
            # tf.print([x.shape for x in tmp_return])
            # tf.print("027-1init", tmp_return.shape)

            tmp_return.set_shape(1391)
            tf.ensure_shape(tmp_return, 1391)
            return tmp_return

        elif forbidden_instruments_flag:
            # Default flag is False and forbidden instruments flag (which means that the chosen type is 1)
            # Only mask the forbidden instruments, all the rest is default
            
            tmp_return = tf.concat(
                [self.default_mask[i] for i in range(5)] + \
                [instruments_mask] + \
                [self.default_mask[i] for i in range(6,len(default_token_parts))], 
                axis=-1, name="concat_forbidden_instruments")
            
            # tf.print([x.shape for x in tmp_return])
            # tf.print("type1 second instrument called final return", tmp_return.shape)
            # tf.print("final return vector", tmp_return)
            
            tmp_return.set_shape(1391)
            tf.ensure_shape(tmp_return, 1391)

            return tmp_return
        
        # elif chosen_type >= 3 and chosen_type <= 6:
        else:
            # General event. What we do depends on which specific event it is, but
            # in general there is always a measure mask.
            
            measure_mask = tf.less_equal(
                tf.range(conf.INPUT_RANGES["measure"]),
                min_measure
            )

            # tf.print("min_measure", min_measure)
            # tf.print("measure_mask", measure_mask)


            tf.ensure_shape(measure_mask, conf.INPUT_RANGES["measure"])
            # We need to do manual masking
            # These are the ones that are default in 4/5/6

            if chosen_type >= 4 and chosen_type <= 6:
                beat_mask        = self.default_mask[1]
                position_mask    = self.default_mask[2] 
                duration_mask    = self.default_mask[3]
                pitch_mask       = self.default_mask[4]
                instruments_mask = self.default_mask[5] 
                velocity_mask    = self.default_mask[6]

                if chosen_type == 4: #respective mask is already defined, only need to define the other 2 ones
                    time_sign_mask = self.default_mask[8]
                    tempo_mask = self.default_mask[9]

                elif chosen_type == 5:
                    key_sign_mask = self.default_mask[7]
                    tempo_mask = self.default_mask[9]

                else:
                    key_sign_mask = self.default_mask[7]
                    time_sign_mask = self.default_mask[8]

            else:
                # key_sign_mask, time_sign_mask and tempo_mask are already defined
                # can always choose any of the below for a note
                duration_mask    = self.full_mask[3]
                pitch_mask       = self.full_mask[4]
                velocity_mask    = self.full_mask[6]

                max_beat = self.get_max_beat_from_time_sign(tf.argmax(time_sign_mask)) # max_beat is the numerator of time_sign_mask
                # allowed beats are only AFTER previous beat and BEFORE max_beat from the numerator of the time_sign
                    
                beat_mask = tf.math.logical_and(
                    tf.greater_equal(
                        tf.range(conf.INPUT_RANGES["beat"]),
                        min_beat
                    ),
                    tf.less(
                        tf.range(conf.INPUT_RANGES["beat"]),
                        max_beat
                    )
                )
                tf.ensure_shape(beat_mask, conf.INPUT_RANGES["beat"])

                position_mask = tf.greater_equal(
                    tf.range(conf.INPUT_RANGES["position"]),
                    min_position
                )
                tf.ensure_shape(position_mask, conf.INPUT_RANGES["position"])
            
            # tf.print("3456 before")
            tmp_return = tf.concat(
                [
                    self.default_mask[0], # ???
                    self.default_mask[1], # bad
                    self.default_mask[2], # bad
                    self.default_mask[3], # bad
                    self.default_mask[4], # ???
                    self.default_mask[5], # bad
                    self.default_mask[6], # bad
                    self.default_mask[7], # bad
                    self.default_mask[8], # ???
                    self.default_mask[9], # bad
                ], axis=-1)
            """ 
            measure_mask =      self.default_mask[0]
            beat_mask =         self.default_mask[1]
            position_mask =     self.default_mask[2]
            duration_mask =     self.default_mask[3]
            pitch_mask =        self.default_mask[4]
            instruments_mask =  self.default_mask[5]
            velocity_mask =     self.default_mask[6]
            key_sign_mask =     self.default_mask[7]
            time_sign_mask =    self.default_mask[8]
            tempo_mask =        self.default_mask[9]
            """
            """ 
            tmp_return = tf.concat([
                measure_mask, 
                beat_mask, 
                position_mask, 
                duration_mask,
                pitch_mask, 
                instruments_mask, 
                velocity_mask, 
                key_sign_mask,
                time_sign_mask, 
                tempo_mask
            ], axis=-1, name="concat_3456")
             """
            # tf.print([x.shape for x in tmp_return])
            # tf.print("3456", tmp_return)

            tmp_return.set_shape(1391)
            tf.ensure_shape(tmp_return, 1391)
            return tmp_return


    def get_mask_for_one_elem_in_batch(self, inputs):
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

        mask = tf.vectorized_map(
            fn=self.get_mask_for_each_token_in_elem,
            elems=(
                tf.range(conf.SEQ_LEN-1),
                chosen_types,
                tf.expand_dims(song_tokens, axis=0),    # expand_dims = 0 allows to pass the song_tokens every time to vectorized_map
                seq_scores,
            )
        )

        # tf.ensure_shape(mask, (conf.SEQ_LEN-1, 1391), name="ensure_shape_mask_batch")

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
                ), dtype="bool")
            , name="map_fn_complex"
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
    
    type_masking_layer = MaskTypeProbabilitiesLayer()
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


##################################
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