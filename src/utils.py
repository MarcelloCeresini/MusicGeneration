from typing import Tuple
import numpy as np
import muspy
import os, shutil, tarfile
from tqdm import tqdm
import config



def get_dataset(key: str, config: config.Config) -> muspy.Dataset:

    for dataset_path in (data_path := config.dataset_paths):
        if not os.path.exists(dataset_path):
            os.mkdir(dataset_path)


    if key == "lmd":
        if len(os.listdir(data_path["lmd"])) == 0:
            return muspy.LakhMIDIDataset(data_path["lmd"], download_and_extract=True, convert=True, n_jobs=config.N_CPUS)
        else:
            return muspy.LakhMIDIDataset(data_path["lmd"], use_converted=True)


    if key == "maestro":
        if len(os.listdir(data_path["maestro"])) == 0:
            return muspy.MAESTRODatasetV3(data_path["maestro"], download_and_extract=True, convert=True, n_jobs=config.N_CPUS)
        else:
            return muspy.MAESTRODatasetV3(data_path["maestro"], use_converted=True)


    if key == "nes":
        if len(os.listdir(data_path["nes"])) == 0:
            raise ImportError("Go download the archive at https://drive.google.com/file/d/1tyDEwe0exW4xU1W7ZzUMWoTv8K43jF_5/view and place it into root")
        
        elif len(os.listdir(data_path["nes"])) == 1:
            if not os.listdir(data_path["nes"])[0] == "nesmdb_midi.tar.gz":       
                raise ValueError("Correct archive not found")
            else:

                with tarfile.open(os.path.join(data_path["nes"], "nesmdb_midi.tar.gz")) as file:
                    file.extractall(data_path["nes"])

                # move all MIDIs in parent dir
                for dir_path in ["train", "valid", "test"]:
                    shutil.copytree(
                        full_dir_path := os.path.join(data_path["nes"], "nesmdb_midi", dir_path),
                        midi_dir := os.path.join(data_path["nes"], "nesmdb_midi"),
                        dirs_exist_ok=True
                    )
                    shutil.rmtree(full_dir_path, ignore_errors=True)
                
                if not os.path.exists(conv_path := os.path.join(data_path["nes"], "_converted")):
                    os.mkdir(conv_path)

                for filename in tqdm(os.listdir(midi_dir)):
                    # convert every file into a Music object
                    mus_file = muspy.read_midi(os.path.join(midi_dir, filename))
                    filename = filename.split(".")[0]
                    # then save it as a
                    mus_file.save(
                        os.path.join(conv_path, filename+".json"),
                        kind = "json"        
                    )

                # create this file for compatibility with muspy
                with open(os.path.join(conv_path, ".muspy.success"), "w") as f:
                    pass

                return muspy.FolderDataset(data_path["nes"], use_converted=True)

        else:
            return muspy.FolderDataset(data_path["nes"], use_converted=True)


    if key == "hymn":
        if len(os.listdir(data_path["hymn"])) == 0:
            return muspy.HymnalTuneDataset(data_path["hymn"], download=True, convert=True, n_jobs=config.N_CPUS)
        else:
            return muspy.HymnalTuneDataset(data_path["hymn"], use_converted=True)
            

    if key == "folk":
        if len(os.listdir(data_path["folk"])) == 0:
            return muspy.NottinghamDatabase(data_path["folk"], download_and_extract=True, convert=True, n_jobs=config.N_CPUS)
        else:
            return muspy.NottinghamDatabase(data_path["folk"], use_converted=True)


    else:
        raise ValueError("This dataset is not implemented yet")
    


def key_sign_repr(key_sign: Tuple, measure: int, conf: config.Config) -> Tuple:
    '''Create key_sign map from standard muspy to ours'''
    # TODO: implement it
    key_sign_index = 0 

    if conf.config_string == "complete":
        return (4, measure, 0, 0, 0, 0, 0, 0, 0, key_sign_index, 0, 0)
    elif conf.config_string == "single_instruments_type":
        return (4, measure, 0, 0, 0, 0, 0, 0, key_sign_index, 0, 0)


def time_sign_repr(time_sign: Tuple, measure: int, conf: config.Config) -> Tuple:
    '''Create time_sign map from standard muspy to ours'''
    # TODO: implement it
    time_sign_index = 0

    if conf.config_string == "complete":
        return (5, measure, 0, 0, 0, 0, 0, 0, 0, 0, time_sign_index, 0)
    elif conf.config_string == "single_instruments_type":
        return (5, measure, 0, 0, 0, 0, 0, 0, 0, time_sign_index, 0)


def tempo_repr(tempo: float, measure: int, conf: config.Config) -> Tuple:
    '''Create tempo map from standard muspy to ours'''

    tempo_index = np.argmin(np.abs(config.np_tempos - tempo))

    if conf.config_string == "complete":
        return (6, measure, 0, 0, 0, 0, 0, 0, 0, 0, 0, tempo_index)
    elif conf.config_string == "single_instruments_type":
        return (6, measure, 0, 0, 0, 0, 0, 0, 0, 0, tempo_index)


def add_notes(final_song: list, notes: np.array, settings: dict, t_init: int, measure_init: int, time_interval: int, conf: config.Config):
    '''Central function that translates each note from current note to next "event" into its tuple representation'''
    
    i = 0
    while i < len(notes):
        note = notes[i]
        if note[0] >= t_init + time_interval:
            break
        i+=1

        if conf.config_string == "complete":
            final_song.append((
                3,
                (note[0] - t_init) // settings["measure"] + measure_init,
                ((note[0] - t_init) % settings["measure"]) // settings["beat"],
                np.argmin(np.abs(
                    config.np_positions -
                    (((note[0] - t_init) % settings["measure"]) % settings["beat"]) / settings["resolution"])),
                note[2],
                note[1],
                note[4],
                note[5], #TODO: when implementing complete, here goes n°instrument
                note[3],
                settings["key_sign"],
                settings["time_sign"],
                settings["beat"]
            ))

        elif conf.config_string == "single_instruments_type":
            #TODO: define np_positions (possible positions as fraction of beat)
            # (type, measure, beat, position, duration, pitch, instrument_type, velocity, key_sign, time_sign, tempo)
            final_song.append((
                3,
                (note[0] - t_init) // settings["measure"] + measure_init,
                ((note[0] - t_init) % settings["measure"]) // settings["beat"],
                # position is defined as a fraction of a beat --> find the float [0,1] dividing time for beat, and find the closest fraction
                np.argmin(np.abs(
                    conf.np_positions -
                    (((note[0] - t_init) % settings["measure"]) % settings["beat"]) / settings["resolution"])),
                note[2],
                note[1],
                note[4],
                note[3],
                settings["key_sign"],
                settings["time_sign"],
                settings["beat"]
            ))
    
    if i == len(notes):
        return final_song
    else:
        return (
            final_song,
            notes[i:],
            t_init + time_interval,
            measure_init + (time_interval / settings["measure"])
        )


def transform_representation(song: muspy.music.Music, conf: config.Config, verbose=0):

    '''
    This function accepts as input a song in the muspy format and transforms it in a series of tuples written in the following way

                            Description                                                     Interval                                                Possible values
    (
        type,               # see below -->                                                 [0, 7] -->                                              8
        measure,            # index of the measure inside the song in interval -->          [0, last_measure] -->                                   512?
        beat,               # index of beat inside measure -->                              [0, numerator of time_signature] -->                    ??
        position,           # index with 1/64 beat length granularity -->                   [0, 63/64] -->                                          64
        duration,           # hierarchical structure? -->                                   ??? better to specify after dataset exploration                                  
        pitch,              # height of pitch (128) + drums (another 128) -->               [0, 255] -->                                            256
        instrument_type,    # 128 instrument types + 1 for drums -->                        [0, 128] -->                                            129
        n_instrument(*),    # same instrument twice in the song for multiple voices -->     [0, 7??] -->                                            8
        velocity,           # amplitude of note, strength of play -->                       [0, 127] -->                                            128
        key_sign,           # [0,11] (all possible notes) and [maj,min] -->                 [0, 23] -->                                             24
        time_sign,          # denominator pow(2) in [1,64] and numerator int in [0,128] --> ??? better to specify after dataset exploration
        tempo,              # qpm, geometric progression from 16 to 256 -->                 [0, 48] -->                                             49
    )

    type:
        0: start of song
        1: new instrument
        2: start of events
        3: note
        4: key_signature change event
        5: time_signature change event
        6: tempo change event
        7: end of song


    if type = 0 --> all values 0
    if type = 1 --> only instrument_type must be specified (and n_instrument is 1 bigger than the previous identical instrument defined) (other values are 0)
    if type = 2 --> all values 0

    then, before ANY type = 3, MUST FOLLOW at least one of each:
        type = 4 --> only key_sign (other values are 0 except measure)
        type = 5 --> only time_sign (other values are 0 except measure)
        type = 6 --> only tempo (other values are 0 except measure)

    if type = 3 --> all values are full, and key_sign, time_sign and tempo are identical to the last 4, 5 or 6 respectively
    if type = 7 --> all values 0, end of the representation


    '''
    
    # list of all notes/events
    final_song = []

    events = []

    # record time signatures (the first is the "current" and influences notes immediately following, the others will influence 
    # the song later and are appended into "events", where time_sign, key_sign and tempo changes are all stored)

    for i, key_sign in enumerate(song.key_signatures):
        if i == 0:
            current_key_sign = (key_sign.key, key_sign.major_minor)
            final_song.append(key_sign_repr(current_key_sign, measure = 0, conf = conf))
        else:
            events.append((key_sign.time, "key_sign", key_sign.key, key_sign.major_minor))

    for i, time_sign in enumerate(song.time_signatures):
        if i == 0:
            current_time_sign = (time_sign.numerator, time_sign.denominator)
            final_song.append(time_sign_repr(current_time_sign, measure = 0, conf = conf))
        else:
            events.append((time_sign.time, "time_sign", time_sign.numerator, time_sign.denominator))

    for i, tempo in enumerate(song.beats):
        if i == 0:
            current_tempo = tempo.qpm
            final_song.append(tempo_repr(current_tempo, measure = 0, conf = conf))
        else:
            events.append((tempo.time, "tempo", tempo.qpm))

    # sort events by timestep
    events.sort(key = lambda x: x[0])

    # get all the notes from the song
    if conf.config_string == "complete":
        raise ValueError("Not implemented yet")
        notes = np.zeros((
            np.sum([len(track) for track in song.tracks]), 
            6
        ))
        programs = {}
        ### to know how many times a track is used
        for track in song.tracks:
            if track.program in programs.keys():
                programs[track.program] += 1
            else:
                programs[track.program] = 1
    elif conf.config_string == "single_instruments_type":
        notes = np.zeros((
            np.sum([len(track) for track in song.tracks]), 
            5
        ))


    i = 0
    for track in song.tracks:
        if track.is_drum:
            for note in track.notes:
                notes[i, 0] = note.time
                notes[i, 1] = note.pitch + 128
                notes[i, 2] = note.duration
                notes[i, 3] = note.velocity
                #TODO: implement get_program --> map muspy programs to [0,128]
                notes[i, 4] = utils.get_program(track.program)
                i+=1
        else:
            for note in track.notes:
                notes[i, 0] = note.time
                notes[i, 1] = note.pitch
                notes[i, 2] = note.duration
                notes[i, 3] = note.velocity
                notes[i, 4] = track.program
                i+=1

    # sort them by the 0th column, the time
    notes = notes[notes[:,0].argsort()]

    # number of timesteps per beat --> absolute_time = (60*metrical_time)/(tempo_qpm*resolution)
    resolution = song.resolution

    current_beat_length = resolution

    # n° of timesteps for each measure = resolution * nominator of time_sign (n° of beats in a measure)
    current_measure_length = current_time_sign[0] * resolution
    
    current_settings = {
        "key_sign": current_key_sign,       # tuple with (key, major/minor)
        "time_sign": current_time_sign,     # tuple (nominator, denominator)
        "tempo": current_tempo,             # float
        "resolution": resolution,           # int
        "measure": current_measure_length,  # int
        "beat": current_beat_length         # int
    }


    # remember at which note we stopped changing representation
    current_note_idx = 0

    # also remember the number of the measure
    current_measure_index = 0

    # t = timestep --> differs for each song based on the resolution
    # remember also the timestep at which we stopped
    current_time = 0

    while i < len(events):
        # add notes in between events --> the current configuration is important to define to which measure/beat they belong to
        event = events[i]
        # the dataset is not clean, sometimes it happens that the same event is repeated twice or more --> we want to make changes only when a NEW event occurs
        flag_new_event = False
        new_settings = current_settings
        
        if event[1] == "key_sign":
            if event[2] != current_settings["key_sign"][0] and \
                event[3] != current_settings["key_sign"][1]:

                new_settings["key_sign"] = (event[2], event[3]) # numerator, denominator
                new_settings["measure"] = event[3] * resolution
                new_settings["beat"] = new_settings["measure"] / event[2]

                tmp = key_sign_repr(new_settings["key_sign"])
                flag_new_event = True

        if event[1] == "time_sign":
            if event[2] != current_settings["time_sign"][0] and \
                event[3] != current_settings["time_sign"][1]:
                
                new_settings["time_sign"] = (event[2], event[3]) # note, major/minor
                tmp = time_sign_repr(new_settings["time_sign"])
                flag_new_event = True
        
        if event[1] == "tempo":
            if event[2] != current_settings["tempo"]:
                new_settings["time_sign"] = event[2] # qpm
                tmp = tempo_repr(new_settings["time_sign"])
                flag_new_event = True


        if flag_new_event:
            
            assert (delta_t := (event[0]-t) % current_measure_length) == 0, "The MIDI or the algorithm are wrong, events should happen only at the beginning of measures"

            # if the event happens in the middle of a beat because midi is "wrong" --> move the event to the beginning of THAT measure (not the following one)
            time_interval = current_time - (event[0] - delta_t)

            # shouldn't do anything if there are no notes between current time and t+delta_t
            final_song, notes, current_time, current_measure_index = add_notes( 
                final_song,
                notes,
                current_settings, 
                current_time,
                current_measure_index,
                time_interval,
                conf
            )
            
            current_settings = new_settings
            current_settings["measure"] = current_settings["time_sign"][0] * resolution # beat is constant, onle measure length changes

        final_song.append(tmp)

        i+=1


    final_song, n = add_notes(
        final_song,
        notes,
        current_settings,
        current_time,
        current_measure_index,
        100000, # just to be safe
        conf
    ) # should append every note between current note and note finish
