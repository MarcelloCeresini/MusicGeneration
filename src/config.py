import numpy as np
import transformers
import tensorflow as tf
import os
import math

class Config:
    
    def __init__(self, config_string, root_path):

        # SYSTEM INFO
        self.N_CPUS  = os.cpu_count()
        self.GPUS    = tf.config.experimental.list_physical_devices('GPU')
        # Setup memory growth
        for gpu in self.GPUS:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Setup training streategy for Multi-GPU training
        if len(self.GPUS) > 1:
            self.training_strategy = tf.distribute.MirroredStrategy()
            self.num_devices = self.training_strategy.num_replicas_in_sync
        else:
            self.num_devices = 1
        
        # DATASET SELECTION
        self.config_string = config_string

        if config_string == "complete":
            self.tuple_size = 12
        elif config_string == "single_instruments_type":
            self.tuple_size = 11
        else:
            raise ValueError("Not implemented yet")

        # PATHS DEFINITION
        self.DATA_PATH = os.path.join(root_path, "data/")
        self.dataset_paths = {
            "lmd": os.path.join(self.DATA_PATH, "lmd/"),
            "maestro": os.path.join(self.DATA_PATH, "maestro/"),
            "nes": os.path.join(self.DATA_PATH, "nes/"),
            "hymn": os.path.join(self.DATA_PATH, "hymn/"),
            "folk": os.path.join(self.DATA_PATH, "folk/"),
            "lmd_matched": os.path.join(self.DATA_PATH, "lmd_matched/")
        }
        self.tf_data_path = os.path.join(self.DATA_PATH, "tf_data")
        self.lmdm_tf_data_path = os.path.join(self.DATA_PATH, "lmdm_tf_data")

        # NOTATION DEFINITIONS
        # Tempo
        max_tempo = 256
        min_tempo = 16
        num_tempos = 49
        r = (max_tempo/min_tempo)**(1/(num_tempos-1))
        self.tempos = [min_tempo * r**i for i in range(num_tempos)]
        self.np_tempos = np.asarray(self.tempos)

        # Duration
        np_durations = np.zeros(300)
        i = 0
        note_l = 1/2
        for _ in range(int(256/16)):
            while np_durations[i] < note_l:
                i+=1
                np_durations[i] = np_durations[i-1] + (note_l/32)
            note_l *= 2
        self.np_durations = np_durations[1:129+8]

        # Position
        np_positions = np.zeros(128)
        for i in range(len(np_positions)-1):
            np_positions[i+1] = np_positions[i] + 1/128
        self.np_positions = np_positions

        # Time signature
        self.numerators = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 132]
        self.denominators = [1, 2, 4, 8, 16, 32, 64, 128]
        self.tot_numerators = len(self.numerators)
        self.tot_denominators = len(self.denominators)
        
        # Genres
        # USE THIS FOR THE LARGE DATASET
        self.accepted_subgenres = [
            "rock","pop","dance","country","metal",
            "classical","folk","blues","house","indie", 
            "latin","jazz","funk","rap","punk","r&b", 
            "gospel","electronic"
        ]
        # USE THIS FOR THE SMALL DATASET
        # self.accepted_subgenres = ['folk', 'nes', 'maestro']

        # MODEL CONFIGURATIONS
        # Decoder
        self.SEQ_LEN                        = 6144
        self.TOKEN_DIM                      = 512
        self.GENRE_DIM                      = 512
        self.ATTENTION_BLOCKS               = 6
        self.ATTENTION_HEADS                = 2
        self.DECODER_ACTIVATION_FUNCTION    = "relu"

        # Custom configuration for using GPT2 as a standard transformer decoder
        self.decoder_config = transformers.GPT2Config(
            vocab_size=0, 
            n_positions = self.SEQ_LEN, 
            n_embd = self.TOKEN_DIM, 
            n_layer = self.ATTENTION_BLOCKS, 
            n_head = self.ATTENTION_HEADS, 
            activation_function='relu',
            reorder_and_upcast_attn = True
        )

        # EMBEDDING LAYERS
        self.SINGLE_EMB_SIZE = 64

        # DATASET CONFIGURATIONS
        self.BATCH_SIZE         = 12
        self.GLOBAL_BATCH_SIZE  = self.BATCH_SIZE * self.num_devices
        self.SHUFFLE_SIZE       = 256
        self.PREFETCH_SIZE      = 32

        # TRAINING SETUP
        self.REG_LOSS_SCALE  = 0.001
        self.USE_MASKING     = True

        self.INPUT_RANGES = {
            "type": 8,
            "measure" : 256,
            "beat": 131,
            "position": 128,
            "duration": 136,
            "pitch": 256,
            "instrument": 129,
            "velocity": 128,
            "key_sign": 24+1,
            "time_sign": 153,
            "tempo": 49
        } # tot=1399

        self.input_ranges_sum = sum(self.INPUT_RANGES.values())

        self.embedding_layers = [
            tf.keras.layers.Embedding(self.INPUT_RANGES["type"],        self.SINGLE_EMB_SIZE),
            tf.keras.layers.Embedding(self.INPUT_RANGES["measure"],     self.SINGLE_EMB_SIZE),
            tf.keras.layers.Embedding(self.INPUT_RANGES["beat"],        self.SINGLE_EMB_SIZE),
            tf.keras.layers.Embedding(self.INPUT_RANGES["position"],    self.SINGLE_EMB_SIZE),
            tf.keras.layers.Embedding(self.INPUT_RANGES["duration"],    self.SINGLE_EMB_SIZE),
            tf.keras.layers.Embedding(self.INPUT_RANGES["pitch"],       self.SINGLE_EMB_SIZE),
            tf.keras.layers.Embedding(self.INPUT_RANGES["instrument"],  self.SINGLE_EMB_SIZE),
            tf.keras.layers.Embedding(self.INPUT_RANGES["velocity"],    self.SINGLE_EMB_SIZE),
            tf.keras.layers.Embedding(self.INPUT_RANGES["key_sign"],    self.SINGLE_EMB_SIZE),
            tf.keras.layers.Embedding(self.INPUT_RANGES["time_sign"],   self.SINGLE_EMB_SIZE),
            tf.keras.layers.Embedding(self.INPUT_RANGES["tempo"],       self.SINGLE_EMB_SIZE)
        ]
        
        self.output_dense_layers = [
            tf.keras.layers.Dense(self.INPUT_RANGES["type"]),
            tf.keras.layers.Dense(self.INPUT_RANGES["measure"]),
            tf.keras.layers.Dense(self.INPUT_RANGES["beat"]),
            tf.keras.layers.Dense(self.INPUT_RANGES["position"]),
            tf.keras.layers.Dense(self.INPUT_RANGES["duration"]),
            tf.keras.layers.Dense(self.INPUT_RANGES["pitch"]),
            tf.keras.layers.Dense(self.INPUT_RANGES["instrument"]),
            tf.keras.layers.Dense(self.INPUT_RANGES["velocity"]),
            tf.keras.layers.Dense(self.INPUT_RANGES["key_sign"]),
            tf.keras.layers.Dense(self.INPUT_RANGES["time_sign"]),
            tf.keras.layers.Dense(self.INPUT_RANGES["tempo"])
        ]
    
        self.full_mask = [
            # np.asarray([True]*self.INPUT_RANGES["type"], dtype=np.bool8),
            np.asarray([True]*self.INPUT_RANGES["measure"],dtype=np.bool8),
            np.asarray([True]*self.INPUT_RANGES["beat"],dtype=np.bool8),
            np.asarray([True]*self.INPUT_RANGES["position"],dtype=np.bool8),
            np.asarray([True]*self.INPUT_RANGES["duration"],dtype=np.bool8),
            np.asarray([True]*self.INPUT_RANGES["pitch"],dtype=np.bool8),
            np.asarray([True]*self.INPUT_RANGES["instrument"],dtype=np.bool8),
            np.asarray([True]*self.INPUT_RANGES["velocity"],dtype=np.bool8),
            np.asarray([True]*self.INPUT_RANGES["key_sign"],dtype=np.bool8),
            np.asarray([True]*self.INPUT_RANGES["time_sign"],dtype=np.bool8),
            np.asarray([True]*self.INPUT_RANGES["tempo"], dtype=np.bool8)
        ]

        self.default_mask = [
            # np.asarray([True] + [False]*(self.INPUT_RANGES["type"]-1), dtype=np.bool8),
            np.asarray([True] + [False]*(self.INPUT_RANGES["measure"]-1), dtype=np.bool8),
            np.asarray([True] + [False]*(self.INPUT_RANGES["beat"]-1), dtype=np.bool8),
            np.asarray([True] + [False]*(self.INPUT_RANGES["position"]-1), dtype=np.bool8),
            np.asarray([True] + [False]*(self.INPUT_RANGES["duration"]-1), dtype=np.bool8),
            np.asarray([True] + [False]*(self.INPUT_RANGES["pitch"]-1), dtype=np.bool8),
            np.asarray([True] + [False]*(self.INPUT_RANGES["instrument"]-1), dtype=np.bool8),
            np.asarray([True] + [False]*(self.INPUT_RANGES["velocity"]-1), dtype=np.bool8),
            np.asarray([True] + [False]*(self.INPUT_RANGES["key_sign"]-1), dtype=np.bool8),
            np.asarray([True] + [False]*(self.INPUT_RANGES["time_sign"]-1), dtype=np.bool8),
            np.asarray([True] + [False]*(self.INPUT_RANGES["tempo"]-1), dtype=np.bool8)
        ]

    def get_decoder(self):
        '''
        Instantiate decoder from its corresponding configuration
        '''
        return transformers.TFGPT2Model(self.decoder_config)

    def get_positional_embedding_matrix(self):
        '''
        Obtain the positional encoding matrix for the decoder model.
        From "Attention is all you need", https://arxiv.org/pdf/1706.03762.pdf
        '''
        PE = np.zeros((self.SEQ_LEN, self.TOKEN_DIM))
        for pos in range(self.SEQ_LEN):
            for i in range(int(self.TOKEN_DIM/2)):
                PE[pos,2*i]   = math.sin(pos/(10000**(2*i/self.TOKEN_DIM)))
                PE[pos,2*i+1] = math.cos(pos/(10000**(2*i/self.TOKEN_DIM)))
        return PE