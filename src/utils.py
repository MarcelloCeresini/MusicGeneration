from typing import Tuple
import numpy as np

def key_sign_repr(key_sign: Tuple, measure: int, config = None) -> Tuple:
    
    ### create key_sign map from standard to indexes
    key_sign_index = 0 

    if config == None:
        return (4, measure, 0, 0, 0, 0, 0, 0, 0, key_sign_index, 0, 0)


def time_sign_repr(time_sign: Tuple, measure: int, config = None) -> Tuple:

    ### create time_sign map
    time_sign_index = 0

    if config == None:
        return (5, measure, 0, 0, 0, 0, 0, 0, 0, 0, time_sign_index, 0)


def tempo_repr(tempo: float, measure: int, config = None) -> Tuple:

    tempo_index = np.argmin(np.abs(config.np_tempos - tempo))

    return (6, measure, 0, 0, 0, 0, 0, 0, 0, 0, 0, tempo_index)


def add_notes(final_song, settings, starting_time, starting_measure, time_interval, notes, config):
    
    i = 0
    while i < len(notes):
        note = notes[i]
        if note >= starting_time + time_interval:
            break

        final_song.append((
            3,
            (note[0] - starting_time) // settings["measure"],
            ((note[0] - starting_time) % settings["measure"]) // settings["beat"],
            np.argmin(np.abs(
                config.np_positions -
                (((note[0] - starting_time) % settings["measure"]) % settings["beat"]) / settings["resolution"])),
            note[2],
            note[1],
            

        ))