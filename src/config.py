import numpy as np

class Config:
    def __init__(self, config_string):
        

        # tempo definition
        max_tempo = 256
        min_tempo = 16
        num_tempos = 49
        r = (max_tempo/min_tempo)**(1/(num_tempos-1))

        self.tempos = [min_tempo * r**i for i in range(num_tempos)]
        self.np_tempos = np.asarray(self.tempos)
