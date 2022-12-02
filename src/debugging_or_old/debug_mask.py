import os
import tensorflow as tf
import numpy as np
import sys
import config
# tf.data.experimental.enable_debug_mode()
# np.set_printoptions(threshold=sys.maxsize)
# tf.config.run_functions_eagerly(True)

ROOT_PATH = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
conf = config.Config("single_instruments_type", ROOT_PATH)

# dataset = tf.data.Dataset.load(conf.tf_data7dict_path).batch(conf.BATCH_SIZE).cache().shuffle(conf.SHUFFLE_SIZE).prefetch(conf.PREFETCH_SIZE)

# for batch in dataset.take(1):
#     print(batch[0][0].shape)
#     songs = batch[0][0]
#     print(batch[0][1].shape)

songs = np.load("/home/marcello/github/MusicGeneration/src/inputs.npy")
print(songs.shape)    

out_scores = [np.random.random((conf.BATCH_SIZE, conf.SEQ_LEN-1, PartRange)) for PartRange in conf.INPUT_RANGES.values()] # [BATCH_SIZE * SEQ_LEN-1 * events_elements] * INPUT_RANGES (various)


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
        TensorArrayMask = tf.TensorArray(tf.bool, size=conf.SEQ_LEN-1)

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

            TensorArrayMask = TensorArrayMask.write(i, type_mask)

        return TensorArrayMask.stack()


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
        TensorArrayMask = tf.TensorArray(dtype=tf.bool, size=conf.SEQ_LEN-1)
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
            song = song_tokens
            # song        = song_tokens * (tf.expand_dims([1]*idx + [0]*(conf.SEQ_LEN-1-idx), axis=-1)) # Mask all tokens after index idx
            
            # print("printing song from -5+idx to idx+1")
            # print(song[max(0,idx-10):until_idx])
            ## MAIN BODY ##
            # print("real_type", song[idx+1, 0])
            # print("chosen_type", chosen_type)
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

                # print("scores_measures")
                # print(scores[:conf.INPUT_RANGES["measure"]])
                # print("argmax")
                # print(tf.math.argmax(
                #     scores[:conf.INPUT_RANGES["measure"]], 
                #         output_type=tf.int32))

                if tf.math.argmax(
                    scores[:conf.INPUT_RANGES["measure"]], 
                        output_type=tf.int32) == min_measure:  
                    min_beat = song[idx,2]      # It has to be >= than the last beat when measure is the same

                    # print("scores_beat")
                    # print(scores[
                    #     conf.INPUT_RANGES["measure"] : 
                    #     conf.INPUT_RANGES["measure"] + conf.INPUT_RANGES["beat"]])
                    # print("argmax")
                    # print(tf.math.argmax(
                    #     scores[
                    #     conf.INPUT_RANGES["measure"] : 
                    #     conf.INPUT_RANGES["measure"] + conf.INPUT_RANGES["beat"]], 
                    #         output_type=tf.int32))


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

                # print("last_token", song[idx, :])
                # print("min_measure", min_measure)
                # print("min_beat", min_beat)
                # print("min_position", min_position)

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

            # print("last_token")
            # print(song[idx, :])

            ## ENDING PART ##
            # Put together the masks
            if default_flag:                
                # No manual masking required, either "can freely choose this part of the token" (True) or 
                # "can only choose default for this part of the token" (False)
                TensorArrayMask = TensorArrayMask.write(idx, tf.concat(
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
                
                # print("forbidden_instruments --> instruments mask")
                # print(instruments_mask)
                
                # Only mask the forbidden instruments, all the rest is default
                TensorArrayMask = TensorArrayMask.write(idx, tf.concat(
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
                    ), dtype=tf.dtypes.bool
                )
                
                # print("measure_mask")
                # print(measure_mask)

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
                        # print("forbidden_key_sign")
                        # print(key_sign_mask)
                    else:
                        key_sign_mask = self.full_mask[7]
                elif chosen_type == 5:
                    if forbidden_time_sign != -1: ## forbidden_time_sign can only appear if chosen_type = 5
                        # True in all places but the forbidden time signs
                        time_sign_mask = tf.convert_to_tensor([
                            i != forbidden_time_sign 
                            for i in range(conf.INPUT_RANGES["time_sign"])], 
                            dtype=tf.bool)
                        # print("forbidden_time_sign")
                        # print(time_sign_mask)
                    else:
                        time_sign_mask = self.full_mask[8]
                elif chosen_type == 6:
                    if forbidden_tempo != -1: ## forbidden_tempo can only appear if chosen_type = 6
                        # True in all places but the forbidden tempos
                        tempo_mask = tf.convert_to_tensor([
                            i != forbidden_tempo 
                            for i in range(conf.INPUT_RANGES["tempo"])], 
                            dtype=tf.bool)
                        # print("forbidden_tempo")
                        # print(tempo_mask)
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
                        dtype=tf.dtypes.bool)
                        
                        # print("beat_mask")
                        # print(beat_mask)
                        # print("time_sign")
                        # print(allowed_time_sign)
                    
                    if min_position != -1: # it is ALWAYS != -1 if chosen_type == 3
                            
                        position_mask = tf.cast(tf.concat([
                            tf.repeat([False], min_position), 
                            tf.repeat([True],  conf.INPUT_RANGES["position"]-min_position)],
                            axis=-1), 
                        dtype=tf.dtypes.bool)

                        # print("position_mask")
                        # print(position_mask)

                    instruments_mask = tf.sparse.SparseTensor( # Allowed instruments
                        indices=tf.expand_dims(tf.cast(allowed_instruments, tf.int64), axis=-1),
                        values=tf.ones_like(allowed_instruments),
                        dense_shape=[conf.INPUT_RANGES["instrument"]]
                    )
                    instruments_mask = tf.cast(
                        tf.sparse.to_dense(tf.sparse.reorder(instruments_mask), default_value=0),
                        dtype=tf.dtypes.bool)

                    # print("instrument_mask")
                    # print(instruments_mask)

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
                
                    # print("final note mask")
                    # print([measure_mask, beat_mask, position_mask, duration_mask,
                    #     pitch_mask, instruments_mask, velocity_mask, key_sign_mask,
                    #     time_sign_mask, tempo_mask])
                
                # Write on the mask

                # tf.print(idx)

                # tmp = tf.concat([
                #     measure_mask, beat_mask, position_mask, duration_mask,
                #     pitch_mask, instruments_mask, velocity_mask, key_sign_mask,
                #     time_sign_mask, tempo_mask], axis=-1)

                # tf.print(tmp[250:256])

                TensorArrayMask = TensorArrayMask.write(idx, tf.concat([
                    measure_mask, beat_mask, position_mask, duration_mask,
                    pitch_mask, instruments_mask, velocity_mask, key_sign_mask,
                    time_sign_mask, tempo_mask], axis=-1))

        tf.print("inside map_fn")
        # print(TensorArrayMask.dtype)
        # print(TensorArrayMask.read(0)[:40])
        # Return the whole mask
        # tf.print(idx)
        # final_masks = TensorArrayMask.stack()

        # print(final_masks.shape)
        # print(final_masks.dtype)
        # print(final_masks[0, :40])
        # tf.print(final_masks[0, 250:260])
        return TensorArrayMask.stack()

    def call(self, inputs, training=True):
        '''
        Inputs:
        - songs:                BATCH*(SEQ_LEN-1)*11
        - out_logits:           BATCH*(SEQ_LEN-1)*1391 (all except type)
        - types_probabilities:  BATCH*(SEQ_LEN-1)*8 --> becomes chosen_types through argmax --> BATCH*(SEQ_LEN-1)*1

        passes through map_fn --> get_mask_fro_all_tokens to debatch
        '''
        songs, out_logits, types_probabilities = inputs
        chosen_types  = tf.expand_dims(tf.math.argmax(types_probabilities, axis=2), axis=-1) # TODO: check if SEQ_LEN -1 or -2
        concat_logits = tf.concat(out_logits[1:], axis=-1)                 # Concatenate all logits (except type) into a tensor batch_size x seq_len x 1391
        
        # print(songs.shape)
        # print(chosen_types.shape)
        # print(concat_logits.shape)
        
        masks = tf.map_fn(fn=self.get_mask_for_all_tokens, elems=(         # Iterate function over batch dimension 
                tf.cast(chosen_types, concat_logits.dtype),                # BATCH*(SEQ_LEN-1)*1
                tf.cast(songs,   concat_logits.dtype),                     # BATCH*(SEQ_LEN-1)*11
                concat_logits                                              # BATCH*(SEQ_LEN-1)*1391  # TODO: check if SLICE is needed or we could directly pass the full concat_logits
            ), fn_output_signature=tf.TensorSpec(                          # Total: a BATCH * SEQ_LEN-1 * 1403 tensor
                (conf.SEQ_LEN-1, conf.input_ranges_sum - conf.INPUT_RANGES['type']),
                dtype=tf.bool
            ))
        
        # print("inside call")
        # print(masks.shape)
        # print(masks.dtype)
        # print(masks[0, 0, :40])
        # print(masks[0, 1, :40])

        return masks



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

type_mask           = MaskTypeProbabilitiesLayer()(songs, training=True)

types_probabilities = output_probs_layers[0](out_scores[0], type_mask) # BATCH_SIZE * SEQ_LEN-1 * 8
full_mask           = MaskingActivationLayer()([songs, out_scores, types_probabilities])

# Unpack the final masks into a list of masks
index = 0
masks = []          

for key in conf.INPUT_RANGES:
    if key != 'type':
        masks.append(full_mask[:, :, index:index+conf.INPUT_RANGES[key]])
        index += conf.INPUT_RANGES[key]

max_index = 2
print("full_mask")
print(full_mask.shape)
print(full_mask[0, :10, :10].numpy())
# print(full_mask[0, 256:387, :10].numpy())
# print(full_mask[0, 387:515, :10].numpy())
# print(full_mask[0, 515:651, :10].numpy())
# print(full_mask[0, 651:907, :10].numpy())
# print(full_mask[0, 907:1036, :10].numpy())
# print(full_mask[0, 1036:1164, :10].numpy())
# print(full_mask[0, 1164:1189, :10].numpy())
# print(full_mask[0, 1189:1342, :10].numpy())
# print(full_mask[0, 1342:, :10].numpy())

out_probabilities = [types_probabilities] + [
    output_probs_layers[i](out_scores[i], masks[i-1]) for i in range(1, len(output_probs_layers))
]

print(out_scores[0][0, :max_index])
print(out_probabilities[0][0, :max_index].numpy())
print(out_scores[1][0, :max_index])
print(out_probabilities[1][0, :max_index].numpy())
# print([out_probabilities[0][i, :] for i in len(out_probabilities)])