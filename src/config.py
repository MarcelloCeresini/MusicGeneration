import numpy as np
import os

class Config:
    
    def __init__(self, config_string, ROOT_PATH):
        ### config_string for ablation
        self.config_string = config_string

        if config_string == "complete":
            self.tuple_size = 12
        elif config_string == "single_instruments_type":
            self.tuple_size = 11
        else:
            raise ValueError("Not implemented yet")

        # PATHS
        self.DATA_PATH = os.path.join(ROOT_PATH, "data/")
        self.dataset_paths = {
            "lmd": os.path.join(self.DATA_PATH, "lmd/"),
            "maestro": os.path.join(self.DATA_PATH, "maestro/"),
            "nes": os.path.join(self.DATA_PATH, "nes/"),
            "hymn": os.path.join(self.DATA_PATH, "hymn/"),
            "folk": os.path.join(self.DATA_PATH, "folk/"),
            "lmd_matched": os.path.join(self.DATA_PATH, "lmd_matched/")
        }

        self.N_CPUS = os.cpu_count()


        # tempo definition
        max_tempo = 256
        min_tempo = 16
        num_tempos = 49
        r = (max_tempo/min_tempo)**(1/(num_tempos-1))

        self.tempos = [min_tempo * r**i for i in range(num_tempos)]
        self.np_tempos = np.asarray(self.tempos)


        # duration definition
        np_durations = np.zeros(300)
        i = 0
        note_l = 1/2

        for _ in range(int(256/16)):
            while np_durations[i] < note_l:
                i+=1
                np_durations[i] = np_durations[i-1] + (note_l/32)
            note_l *= 2

        self.np_durations = np_durations[1:129+8]


        # position definition
        np_positions = np.zeros(128)
        for i in range(len(np_positions)-1):
            np_positions[i+1] = np_positions[i] + 1/128
        
        self.np_positions = np_positions

        
        # time_signature definition
        self.numerators = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 132]
        self.denominators = [1, 2, 4, 8, 16, 32, 64, 128]
        self.tot_numerators = len(self.numerators)
        self.tot_denominators = len(self.denominators)


        self.tf_data_path = os.path.join(self.DATA_PATH, "tf_data")


        ### MODEL CONFIGURATIONS
        self.SEQ_LEN = 4092
        self.TOKEN_DIM = 512
        self.BATCH_SIZE = 2
        self.SHUFFLE_SIZE = 256
        self.PREFETCH_SIZE = 32

        self.INPUT_RANGES = {
            "type": 8,
            "measure" : 256,
            "beat": 133,
            "position": 128,
            "duration": 136,
            "pitch": 256,
            "instrument": 129,
            "velocity": 128,
            "key_sign": 24, # maybe 25 because a lot of times it's not indicated --> 0 could mean "nothing"
            "time_sign": 153,
            "tempo": 49
        }

        self.EMBEDDING_SIZE = 64
        