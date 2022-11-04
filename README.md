# MusicGeneration

Repository for project Work in A3I

# possible ablation: 
# - position int or float
# - n_instruments or not
# - notes with or without configuration information inside
# - drums?
# - write events 1-2-4-5-6 and then only 3 and 7
# - pass the song 2 times and loss on notes only second time (?)

# style representation with one hot encoding normalized vector + dense (1 layer?) learnable that concatenates with the other embeddings

# inference: 1 step --> unconditional + style --> second step --> input = 1-2-4-5-6 and inference only 3-7

(
    type,
    battuta,
    beat,   # int 0-numeratore 4/4
    position, # discretize in some way
    duration, # [1/64 --> 4] del beat
    pitch, # int
    instrument_type,
    n_instrument, # [0,127] + drums?
    velocity, # [0,127]
    key, # [0,11] and [maj,min] --> [0,22]
    time_sign, # [3/4, 4/4]
    tempo, # qpm 
)

track = [
    (0, 0, 0000),
    (1, 0000, 5, [0-n]),
    (2, 0, ..., 0), # start of events
    (4, 25, 0000, n_key, 0, 0)
    (5, 25, 0000, 0, k_sign, 0)
    (6, 25, 0000, 0, 0, tempo)
    (3, 25, ---, 0, 0, 0)
    (7)
]
