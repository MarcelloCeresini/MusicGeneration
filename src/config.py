import numpy as np
import os

class Config:
    
    def __init__(self, config_string, ROOT_PATH):
        ### config_string for ablation
        self.config_string = config_string

        # PATHS
        self.DATA_PATH = os.path.join(ROOT_PATH, "data/")
        self.dataset_paths = {
            "lmd": os.path.join(self.DATA_PATH, "lmd/"),
            "maestro": os.path.join(self.DATA_PATH, "maestro/"),
            "nes": os.path.join(self.DATA_PATH, "nes/"),
            "hymn": os.path.join(self.DATA_PATH, "hymn/"),
            "folk": os.path.join(self.DATA_PATH, "folk/")
        }

        self.N_CPUS = os.cpu_count()
        # tempo definition
        max_tempo = 256
        min_tempo = 16
        num_tempos = 49
        r = (max_tempo/min_tempo)**(1/(num_tempos-1))

        self.tempos = [min_tempo * r**i for i in range(num_tempos)]
        self.np_tempos = np.asarray(self.tempos)
