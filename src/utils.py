from typing import Tuple
import numpy as np
import muspy
import os, shutil, tarfile
from tqdm import tqdm
import tensorflow as tf
import json
from sklearn.metrics import classification_report

import config

def get_dataset_splits(path: str, conf: config.Config) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    whole_dataset = tf.data.Dataset.load(path)
    
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    whole_dataset = whole_dataset.with_options(options)
    
    train_split = int(len(whole_dataset)/10)*8                  # 80%
    val_split = int(len(whole_dataset)/10)*1                    # 10%
    test_split = len(whole_dataset) - val_split - train_split   # 10%

    train_dataset = whole_dataset.take(train_split)
    val_dataset   = whole_dataset.skip(train_split).take(val_split)
    test_dataset  = whole_dataset.skip(train_split+val_split).take(test_split)

    train_dataset = train_dataset.batch(conf.GLOBAL_BATCH_SIZE).\
                                    cache().\
                                    shuffle(conf.SHUFFLE_SIZE).\
                                    prefetch(conf.PREFETCH_SIZE)
    
    val_dataset = val_dataset.batch(conf.GLOBAL_BATCH_SIZE).\
                                    shuffle(conf.SHUFFLE_SIZE).\
                                    prefetch(conf.PREFETCH_SIZE)
    
    test_dataset = test_dataset.batch(conf.GLOBAL_BATCH_SIZE).\
                                    shuffle(conf.SHUFFLE_SIZE).\
                                    prefetch(conf.PREFETCH_SIZE)

    return train_dataset, val_dataset, test_dataset


def get_dataset(key: str, conf: config.Config) -> muspy.Dataset:

    data_path = conf.dataset_paths

    for dataset_path in data_path.values():
        if not os.path.exists(dataset_path):
            os.mkdir(dataset_path)


    if key == "lmd":
        if len(os.listdir(data_path["lmd"])) == 0:
            return muspy.LakhMIDIDataset(data_path["lmd"], download_and_extract=True, convert=True, n_jobs=conf.N_CPUS)
        else:
            return muspy.LakhMIDIDataset(data_path["lmd"], use_converted=True)

    elif key == "lmd_matched":
        path = data_path[key]
        if len(os.listdir(path)) == 0:
            return muspy.LakhMIDIMatchedDataset(path, download_and_extract=True, convert=True, n_jobs=conf.N_CPUS)
        else:
            return muspy.LakhMIDIMatchedDataset(path, use_converted=True)

    elif key == "maestro":
        if len(os.listdir(data_path["maestro"])) == 0:
            return muspy.MAESTRODatasetV3(data_path["maestro"], download_and_extract=True, convert=True, n_jobs=conf.N_CPUS)
        else:
            return muspy.MAESTRODatasetV3(data_path["maestro"], use_converted=True)


    elif key == "nes":
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


    elif key == "hymn":
        if len(os.listdir(data_path["hymn"])) == 0:
            return muspy.HymnalTuneDataset(data_path["hymn"], download=True, convert=True, n_jobs=conf.N_CPUS)
        else:
            return muspy.HymnalTuneDataset(data_path["hymn"], use_converted=True)
            

    elif key == "folk":
        if len(os.listdir(data_path["folk"])) == 0:
            return muspy.NottinghamDatabase(data_path["folk"], download_and_extract=True, convert=True, n_jobs=conf.N_CPUS)
        else:
            return muspy.NottinghamDatabase(data_path["folk"], use_converted=True)


    else:
        raise ValueError("This dataset is not implemented yet")
    

def program_map(track_program: int, is_drum: bool) -> int:
    '''Map muspy program to our program, if is_drum is True, return 128'''
    return 128 if is_drum else track_program


def pitch_map(pitch: int, track_is_drum: bool) -> int:
    '''Map muspy pitch to our pitch, for now it just returns the pitch but it's here for flexible use'''
    return pitch


def key_sign_map(key_sign: Tuple, conf: config.Config) -> int:
    if key_sign[0] == -1:
        return 0
    else:
        if key_sign[1] == "major":
            c = 0
        else:
            c = 1
        return key_sign[0] + 12*c + 1 # 1 is added to shift to account for default


def key_sign_repr(key_sign: Tuple, measure: int, conf: config.Config = None) -> Tuple:
    '''Create key_sign map from standard muspy to ours'''
    key_sign_index = key_sign_map(key_sign, conf)

    if conf.config_string == "complete":
        return (4, measure, 0, 0, 0, 0, 0, 0, 0, key_sign_index, 0, 0)
    elif conf.config_string == "single_instruments_type":
        return (4, measure, 0, 0, 0, 0, 0, 0, key_sign_index, 0, 0)


def time_sign_map(time_sign: Tuple, conf: config.Config = None) -> int:
    if (time_sign[0] in conf.numerators) and (time_sign[1] in conf.denominators):
        return conf.numerators.index(time_sign[0]) + conf.denominators.index(time_sign[1])*conf.tot_numerators
    else:
        return -1


def time_sign_repr(time_sign: Tuple, measure: int, conf: config.Config) -> Tuple:
    '''Create time_sign map from standard muspy to ours'''
    time_sign_index = time_sign_map(time_sign, conf)

    if time_sign_index == -1:
        return False

    if conf.config_string == "complete":
        return (5, measure, 0, 0, 0, 0, 0, 0, 0, 0, time_sign_index, 0)
    elif conf.config_string == "single_instruments_type":
        return (5, measure, 0, 0, 0, 0, 0, 0, 0, time_sign_index, 0)


def tempo_map(tempo: float, conf: config.Config = None) -> int:
    return np.argmin(np.abs(conf.np_tempos - tempo))


def tempo_repr(tempo: float, measure: int, conf: config.Config = None) -> Tuple:
    '''Create tempo map from standard muspy to ours'''
    tempo_index = tempo_map(tempo, conf)

    if conf.config_string == "complete":
        return (6, measure, 0, 0, 0, 0, 0, 0, 0, 0, 0, tempo_index)
    elif conf.config_string == "single_instruments_type":
        return (6, measure, 0, 0, 0, 0, 0, 0, 0, 0, tempo_index)


def add_notes(
        final_song: list, 
        notes: np.array, 
        settings: dict, 
        t_init: int, 
        measure_init: int, 
        time_interval: int, 
        conf: config.Config, 
        debug=False
    ):
    '''
    Central function that translates each note from current note to next "event" into its tuple representation
    
    tuple = (type, measure, beat, position, duration, pitch, instrument_type, velocity, key_sign, time_sign, tempo)
    note = (
      0- time, 
      1- pitch, 
      2- duration, 
      3- velocity, 
      4- program
    )
    '''
    
    i = 0
    interval_finish_time = t_init + time_interval

    while i < len(notes):
        note = notes[i]

        if note[0] >= interval_finish_time:
            break # goes to the second return, all the notes after i (included i), will be returned

        i+=1
        # print("i", i)
        # print("measure", settings["measure"])
        # print("beat",settings["beat"])

        # print("position_value", (((note[0] - t_init) % settings["measure"]) % settings["beat"]) / settings["beat"]) 
        
        # # print("position_idx", 
        # #       np.argmin(np.abs(
        # #             conf.np_positions -
        # #             ((note[0] - t_init) % settings["measure"]) % settings["beat"])
        # #         )
        # # )
        
        # print("t_init", t_init)
        # print("-----")


        if conf.config_string == "complete":
            raise NotImplementedError("Not yet implemented")

        elif conf.config_string == "single_instruments_type":
            final_song.append((
                3,
                (note[0] - t_init) // settings["measure"] + measure_init,
                ((note[0] - t_init) % settings["measure"]) // settings["beat"],
                # position is defined as a fraction of a beat --> find the float [0,1] dividing time for beat, and find the closest fraction
                np.argmin(np.abs(
                    conf.np_positions -
                    (((note[0] - t_init) % settings["measure"]) % settings["beat"]) / settings["beat"]
                )),
                np.argmin(np.abs(conf.np_durations - (note[2]/settings["beat"]))),
                note[1],
                note[4],
                note[3],
                key_sign_map(settings["key_sign"], conf),
                time_sign_map(settings["time_sign"], conf),
                tempo_map(settings["tempo"], conf)
            ))
    
    # if debug:
    #     if i < len(notes):
    #         print(t_init, time_interval, notes[i][0], t_init+time_interval)
    #         print(t_init, time_interval, notes[i+1][0], t_init+time_interval)
    #         print(len(notes))
    #         print(i)
    #         raise ValueError("Something is wrong with the end of the song")

    # else:
    #     if i >= len(notes):
    #         print(t_init,  time_interval)
    #         print(notes[-1][0])
    #         print(i, len(notes))
    #         raise ValueError("???")

    if len(notes) == 0 or i >= len(notes):

        return (
            final_song,
            [], # no more notes, the song is over
            interval_finish_time,
            final_song[-1][1] # the final measure --> cannot be more than 255
        )
    
    else:
        if debug:
            print("settings while adding notes: ", settings)
            print("measure of end of interval: ", measure_init + (time_interval / settings["measure"]))
            print("time_interval: ", time_interval)
        return (
            final_song,
            notes[i:],
            interval_finish_time,
            measure_init + (time_interval / settings["measure"])
        )


def transform_representation(song: muspy.music.Music, conf: config.Config, verbose=0) -> np.ndarray:

    '''
    This function accepts as input a song in the muspy format and transforms it in a series of tuples written in the following way

                            Description                                                     Interval                                                Possible values
    (
        0-type,               # see below                                                     [0, 7]                                                  8
        1-measure,            # index of the measure inside the song in interval              [0, last_measure]                                       512?
        2-beat,               # index of beat inside measure                                  [0, numerator of time_signature]                        132 (max numerator allowed)
        3-position,           # index with 1/64 beat length granularity                       [0, 63/64]                                              64
        4-duration,           # hierarchical structure?                                       [0, ~50]                                                136                                  
        5-pitch,              # height of pitch (128)                                         [0, 127]                                                128
        6-instrument_type,    # 128 instrument types + 1 for drums                            [0, 255]                                                256
        7-velocity,           # amplitude of note, strength of play                           [0, 127]                                                128
        8-key_sign,           # [0,11] (all possible notes) and [maj,min]                     [0, 23]                                                 24
        9-time_sign,          # denominator pow(2) in [1,64] and numerator int in [0,128]     ??? better to specify after dataset exploration
        10-tempo,             # qpm, geometric progression from 16 to 256                     [0, 48]                                                 49
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
    if type = 1 --> only instrument_type must be specified (if "complete" n_instrument is 1 bigger than the previous identical instrument defined ) (other values are 0)
    if type = 2 --> all values 0

    then, before ANY type = 3, MUST FOLLOW at least one of each:
        type = 4 --> only key_sign (other values are 0 except measure)
        type = 5 --> only time_sign (other values are 0 except measure)
        type = 6 --> only tempo (other values are 0 except measure)

    if type = 3 --> all values are full, and key_sign, time_sign and tempo are identical to the last 4, 5 or 6 respectively
    if type = 7 --> all values 0, end of the representation


    Returns:
    - [0]           if song empty
    - [1]           if there are time_signatures not accepted by our discretization
    - [2]           if the song is too long (>256 measures)
    - final_song    if everything works well
    '''
    
    # list of all notes/events
    # REJECT_SONG = False

    final_song = []

    # start of song
    final_song.append(tuple([0]*conf.tuple_size))

    if all([True if len(track.notes)<=0 else False for track in song.tracks]): 
        return [0]

    for track in song.tracks:
        programs = [] # we don't use this, but we could add a feature in the tuple to split different tracks of the same instrument

        track_modified_program = program_map(track.program, track.is_drum)
        assert track_modified_program <= 128, "track_modified_program must be < 128"
        assert track_modified_program >= 0, "track_modified_program must be >= 0"

        if len(track.notes)>0 and (track_modified_program not in set(programs)):
            # different conf_string may change this ####################################
            # add each instrument with (1, 0, 0, 0, 0, 0, program, 0, 0, 0, 0)
            final_song.append(tuple([1]+ [0]*5 +[track_modified_program]+ [0]*4 ))
            if not conf.multiple_tracks_for_same_instrument:
                programs.append(track_modified_program)
            else:
                pass # in this way the same tuple will be added multiple times for the same instrument

    # start of events
    final_song.append(tuple([2]+[0]*(conf.tuple_size-1)))

    events = []

    # record time signatures (the first is the "current" and influences notes immediately following, the others will influence 
    # the song later and are appended into "events", where time_sign, key_sign and tempo changes are all stored)

    
    if len(song.key_signatures) == 0:
        current_key_sign = (-1,-1) # default if there is not even one specified in the song
        final_song.append(key_sign_repr(current_key_sign, measure = 0, conf = conf))

    else:
        for i, key_sign in enumerate(song.key_signatures):
            if i == 0:
                current_key_sign = (key_sign.root, key_sign.mode)
                final_song.append(key_sign_repr(current_key_sign, measure = 0, conf = conf))
            else:
                events.append((key_sign.time, "key_sign", key_sign.root, key_sign.mode))

    if len(song.time_signatures) == 0:
        current_time_sign = (4, 4) # default if there is not even one specified in the song
        final_song.append(time_sign_repr(current_time_sign, measure = 0, conf = conf))

    else:
        for i, time_sign in enumerate(song.time_signatures):
            if i == 0:
                current_time_sign = (time_sign.numerator, time_sign.denominator)
                tmp = time_sign_repr(current_time_sign, measure = 0, conf = conf)
                # we do not accept all the time signatures, all the songs that use different ones are rejected
                if tmp == False:
                    return [1] # special return to count how many songs contain non accepted time_sign
                else:
                    final_song.append(tmp)
            else:
                events.append((time_sign.time, "time_sign", time_sign.numerator, time_sign.denominator))

                
    if len(song.tempos) == 0:
        current_tempo = 120 # same
        final_song.append(tempo_repr(current_tempo, measure = 0, conf = conf))
    
    else:
        for i, tempo in enumerate(song.tempos):
            if i == 0:
                current_tempo = tempo.qpm
                final_song.append(tempo_repr(current_tempo, measure = 0, conf = conf))
            else:
                events.append((tempo.time, "tempo", tempo.qpm))


    # sort events by timestep
    events.sort(key = lambda x: x[0])

    # get all the notes from the song
    if conf.config_string == "complete":
        raise NotImplementedError("Not implemented yet")
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

    resolution = song.resolution

    j = 0
    for track in song.tracks:
        c=0
        for i, note in enumerate(track.notes):
            if note.duration / resolution < min(conf.np_durations):
                pass # we don't accept notes with such low durations to clean the dataset
            else:
                notes[j+c, 0] = note.time
                notes[j+c, 1] = pitch_map(note.pitch, track.is_drum)
                notes[j+c, 2] = note.duration
                notes[j+c, 3] = note.velocity
                notes[j+c, 4] = track.program
                c+=1
        
        j += c

    notes = notes[:j]
    # sort them by the 0th column, the time
    notes = notes[notes[:,0].argsort()]

    if verbose:
        for note in notes:
            print(note)


    # number of timesteps per beat --> absolute_time = (60*metrical_time)/(tempo_qpm*resolution)

    # measure_length = n° of timesteps for each measure = resolution * nominator of time_sign (n° of beats in a measure)
    
    current_settings = {
        "key_sign": current_key_sign,                   # tuple with (key, major/minor)
        "time_sign": current_time_sign,                 # tuple (nominator, denominator)
        "tempo": current_tempo,                         # float
        "resolution": resolution,                       # int
        "measure": current_time_sign[0] * resolution,   # int (THE LENGTH OF THE MEASURE)
        "beat": resolution                              # int
    }

    # remember the number of the measure
    current_measure_index = 0

    # timestep --> differs for each song based on the resolution
    # remember also the timestep at which we stopped
    current_time = 0

    for event in events:
        assert current_time >= 0, "current_time {}".format(current_time)
        # add notes in between events --> the current configuration is important to define to which measure/beat they belong to

        # the dataset is not clean, sometimes it happens that the same event is repeated twice or more --> we want to make changes only when a NEW event occurs
        flag_new_event = False
        new_settings = current_settings.copy()
        
        if event[1] == "key_sign":
            if event[2] != current_settings["key_sign"][0] or \
                event[3] != current_settings["key_sign"][1]:

                new_settings["key_sign"] = (event[2], event[3]) # note, major/minor

                current_event = key_sign_repr(
                    new_settings["key_sign"],
                    (event[0] - current_time) // current_settings["measure"] + current_measure_index,
                    conf
                )

                flag_new_event = True
            else:
                flag_new_event = False

        if event[1] == "time_sign":
            if event[2] != current_settings["time_sign"][0] or \
                event[3] != current_settings["time_sign"][1]:

                new_settings["time_sign"] = (event[2], event[3]) # numerator, denominator
                new_settings["measure"] = event[2] * resolution # measure length changes (only) if numerator changes
                
                current_event = time_sign_repr(
                    new_settings["time_sign"],
                    (event[0] - current_time) // current_settings["measure"] + current_measure_index,
                    conf
                )
                    
                # the time_signature could be rejected, we do not accept all possible ones
                if current_event == False:
                    return [1]
                else:
                    flag_new_event = True
                
            else:
                flag_new_event = False

        if event[1] == "tempo":
            if event[2] != current_settings["tempo"]:

                new_settings["tempo"] = event[2] # qpm

                current_event = tempo_repr(
                    new_settings["tempo"],
                    (event[0] - current_time) // current_settings["measure"] + current_measure_index,
                    conf
                )
                flag_new_event = True
            else:
                flag_new_event = False


        if flag_new_event:
            debug=False
            # if event[1] == "time_sign":
            #     print("current_settings: ", current_settings)
            #     print("new_settings: ", new_settings)
            #     print("time: ", current_time)
            #     print("measure: ", current_measure_index)
            #     print("time_of_event: ", event[0])
            #     debug=True

            if (delta_t := (event[0]-current_time) % current_settings["measure"]) != 0:
                # print(song.metadata)
                # print(events)
                pass

            # if the event happens in the middle of a beat because midi is "wrong" --> move the event to the beginning of THAT measure (not the following one)
            # every note will be positioned with respect to this new timestep (measures, beats and positions are shifted)
            time_interval = (event[0] - delta_t) - current_time
            assert time_interval >= 0, "time interval: {}".format(time_interval)

            # if event[1] == "time_sign":
            #     print("time_interval: ", time_interval)
            #     print("raw_time_interval: ", (event[0] - current_time))
            #     print("delta_t: ", delta_t)

            # shouldn't do anything if there are no notes between current time and t+delta_t
            try:                
                final_song, notes, current_time, current_measure_index = add_notes( 
                    final_song,
                    notes,
                    current_settings, 
                    current_time,
                    current_measure_index,
                    time_interval,
                    conf,
                    debug
                )
                final_song.append(current_event)

            except:
                print(events)
                print(notes[-10:])
                print(song.metadata)

            # if event[1] == "time_sign":
            #     print("time after adding notes: ", current_time)
            #     print("measure after adding notes: ", current_measure_index)
            #     print("--------------------")

            current_settings = new_settings.copy()
            # beat is constant, onle measure length changes --> update it
        
        else:
            pass
    ### sometimes events have timesteps > than last note!

    try:

        final_song, _, _, current_measure_index = add_notes(
            final_song,
            notes,
            current_settings,
            current_time,
            current_measure_index,
            1e15, # time_interval very big so that you catch all the notes TODO: improve it
            conf,
            debug=False
        ) # should append every note between current note and note finish

    except:
        print(final_song)
        print(notes[0])
        print(current_time)
        print(song.metadata)
    
    # we save npy in uint8, so all the parts of the tuples must be <256
    # (beat is <132 that is the biggest numerator accepted in time signatures)
    if current_measure_index > 255: return [2]

    # add end of song
    final_song.append(tuple([7]+[0]*(conf.tuple_size-1)))   
    return np.stack(final_song).astype(dtype=np.uint8)


def map_dataset_to_label(string: str):
    if string == "folk":
        return np.array(0, dtype=np.uint8)
    if string == "nes":
        return np.array(1, dtype=np.uint8)
    if string == "maestro":
        return np.array(2, dtype=np.uint8)
    if string == "lmd_matched":
        return np.array(3, dtype=np.uint8)


def one_hot_encode_labels_nmf(string: str):
    if string == "folk":
        return np.array((1, 0, 0), dtype=np.uint8)
    elif string == "nes":
        return np.array((0, 1, 0), dtype=np.uint8)
    elif string == "maestro":
        return np.array((0, 0, 1), dtype=np.uint8)
    else:
        raise ValueError("Not implemented")


def program_inverse_map(program_own: int) -> Tuple[int, bool]:
    '''
        Returns the instrument/program number given our representation, plus "is_drum" = false
        If it is a drum, returns ALWAYS 0 as a channel, and "is_drum" = True
    '''
    return (int(program_own), False) if program_own < 128 else (0, True) 


def pitch_inverse_map(pitch_own: int) -> int:
    '''Returns the pitch given our representation, now it does nothing but it's here for consistency'''
    return int(pitch_own)


def key_sign_inverse_map(key_sign_own: int) -> Tuple[int, str] or None:
    '''Returns the key signature given our representation'''
    if key_sign_own == 0:
        return None
    else:
        if (key_sign_own-1) < 12:
            mode = "major"
        else:
            mode = "minor"

        return (key_sign_own-1) % 12, mode
    

def time_sign_inverse_map(time_sign_own: int, conf: config.Config) -> Tuple[int, int]:
    '''Returns the time signature given our representation'''
    numerator = conf.numerators[time_sign_own % conf.tot_numerators]
    denominator = conf.denominators[time_sign_own // conf.tot_numerators]
    return (numerator, denominator)


def tempo_inverse_map(tempo_own: int, conf: config.Config) -> int:
    '''Returns the tempo given our representation'''
    return conf.np_tempos[tempo_own]


def anti_tranform_representation(song: np.ndarray, conf: config.Config) -> muspy.music.Music:
    resolution = conf.standard_resolution
    tempos = []
    time_signatures = []
    key_signatures = []
    tracks = []
    tracks_instruments = []
    note_times = {}

    # TODO: add sorting for "time" before the for loop

    current_settings = {
        "time": 0,
        "numerator_time_sign": 4,
        "measure": 0,
    }

    for tuple in song:
        
        # time calculation: settings are changed only if time_sign changes, so everything is related to the last time_sign change
        position = conf.np_positions[tuple[3]]
        beats_in_measure = current_settings["numerator_time_sign"]
        beats_from_current_time = (int(tuple[1])-current_settings["measure"])*beats_in_measure + int(tuple[2]) + position
        time = current_settings["time"] + resolution*beats_from_current_time

        # if tuple[0] == 3 and tuple[6] == 26:
        #     print("position: ", position)
        #     print("beats_in_measure: ", beats_in_measure)
        #     print("beats_from_current_time: ", beats_from_current_time)
        #     print("time: ", time)
        
        if tuple[0] == 0:
            # start of inputs
            pass

        if tuple[0] == 1:
            # new instrument --> create new track
            program, is_drum = program_inverse_map(tuple[6])

            if program not in tracks_instruments:
                tracks.append({
                    "program": int(program),
                    "is_drum": is_drum,
                    "notes": [],
                    "name": str(np.random.randint(1e5)), # TODO: add name
                })
                tracks_instruments.append(program)

        if tuple[0] == 2:
            # start of song
            pass

        if tuple[0] == 3:
            # note
            instrument = tuple[6] # find instrument
            # OSS: this does NOT work if we allow more than one track with the same instrument in our representation
            try:
                idx = tracks_instruments.index(instrument) # find the track with that instrument in the song
            except:
                # Assign a random instrument from the defined ones
                idx = np.random.randint(0, len(tracks_instruments))
                # Comment line below if there are a lot of errors
                #print("Note has instrument {} but the song has only instanciated these ones: {}".format(instrument, tracks_instruments))
                print("Note that does not belong to any defined instrument --> goes to instrument: ", tracks_instruments[idx])


            if time not in note_times.keys():
                note_times[time] = 1
                tracks[idx]["notes"].append({
                        "time": time,
                        "pitch": pitch_inverse_map(tuple[5]),
                        "duration": resolution * conf.np_durations[tuple[4]],
                        "velocity": int(tuple[7]),
                })
            else:
                if note_times[time] < 10:

                    tracks[idx]["notes"].append({
                        "time": time,
                        "pitch": pitch_inverse_map(tuple[5]),
                        "duration": resolution * conf.np_durations[tuple[4]],
                        "velocity": int(tuple[7]),
                    })
                    note_times[time] += 1
                else:
                    pass # Too many notes at the same time, fluidsynth would crash

        if tuple[0] == 4:
            # key signature
            key_signature_tmp = key_sign_inverse_map(tuple[8])
            if key_signature_tmp is not None:
                key_signatures.append({
                    "time": time,
                    "root": int(key_signature_tmp[0]),
                    "mode": key_signature_tmp[1],
                })
            else:
                pass # song CAN have no key signaturescer

        if tuple[0] == 5:
            # time signature
            numerator, denominator = time_sign_inverse_map(tuple[9], conf)
            time_signatures.append({
                "time": time,
                "numerator": numerator,
                "denominator": denominator,
            })

            # print("before")
            # print(current_settings)
            # print("beats_from_current_time {:.0f}".format(beats_from_current_time))
            # print("time {:.2f}".format(time))

            current_settings["time"] = time
            current_settings["numerator_time_sign"] = numerator
            current_settings["measure"] = tuple[1]


        if tuple[0] == 6:
            # tempo
            qpm = tempo_inverse_map(tuple[10], conf)
            tempos.append({
                "time": time,
                "qpm": qpm
            })
        
        if tuple[0] == 7:
            # end of song
            pass

    counter=0
    index = 0
    tot_len = len(tracks)

    while counter<tot_len:
        track = tracks[index]

        if len(track["notes"]) <= 0:
            # print("track {} to be removed".format(track["program"]))
            tracks.pop(index)

        else:
            # print(track["program"], track["is_drum"], track["notes"][:2])
            index += 1
        counter += 1
    

    # print("---------TRACKS THAT GO INTO JSON")
    # for track in tracks:
    #     print(track["program"], len(track["notes"]))

    if len(tempos) == 0:
        standard_qpm = tempo_inverse_map(35, conf) # ~120qpm
        tempos.append({
                "time": 0,
                "qpm": standard_qpm
        })

    if len(time_signatures) == 0:
        time_signatures.append({
                "time": 0,
                "numerator": 4,
                "denominator": 4,
        })

    # create dict
    song = {
        "tracks": tracks,
        "tempos": tempos,
        "time_signatures": time_signatures,
        "key_signatures": key_signatures,
        "resolution": resolution,
    }

    # return song # ONLY FOR DEBUGGING

    if not os.path.isdir(os.path.join(conf.DATA_PATH, "generation")):
        os.mkdir(os.path.join(conf.DATA_PATH, "generation"))
    
    path = os.path.join(conf.DATA_PATH, "generation", "tmp.json")

    with open(path, "w") as f:
        json.dump(song, f)

    converted_muspy_music_object = muspy.load_json(path)
    return converted_muspy_music_object


def metrics_classification_report(gt_genre_vectors, predicted_genre_vectors, conf:config.Config) -> dict:

    bool_gt_array = np.zeros((len(gt_genre_vectors), len(conf.accepted_subgenres)))
    bool_pred_array = np.zeros_like(bool_gt_array)

    for i, (gt_genre, predicted_genre) in tqdm(enumerate(zip(gt_genre_vectors, predicted_genre_vectors))):

        bool_gt_array[i, :] = gt_genre>0

        n_genres = np.sum(bool_gt_array[i, :])
        bool_pred_array[i, :] = (predicted_genre*n_genres) >= conf.genre_classification_threshold

    return classification_report(bool_gt_array, bool_pred_array, target_names=conf.accepted_subgenres, output_dict=True)


# def metric_from_c_matrix(confusion_matrix_one_class, metric):
#     c = confusion_matrix_one_class

#     if metric == "accuracy":
#         return (c[0,0]+c[1,1])/np.sum(c)
#     elif metric == "precision":
#         return c[0,0]/(c[0,0], c[1,0])
#     elif metric == "recall":
#         return c[0,0]/(c[0,0], c[0,1])
#     elif metric == "f1_score":
#         precision   = c[0,0]/(c[0,0], c[1,0])
#         recall      = c[0,0]/(c[0,0], c[0,1])
#         return 2*precision*recall / (precision+recall)
#     else:
#         raise ValueError("{} is not an implemented metric yet".format(metric))


# def metrics(confusion_matrix: np.array):
#     c = confusion_matrix
#     n_labels = c.shape()[0]

#     # TP FN
#     # FP TN
#     accuracy_per_class  = [metric_from_c_matrix(c[i,...], "accuracy")  for i in range(n_labels)]
#     precision_per_class = [metric_from_c_matrix(c[i,...], "precision") for i in range(n_labels)]
#     recall_per_class    = [metric_from_c_matrix(c[i,...], "recall")    for i in range(n_labels)]
#     f1_per_class        = [metric_from_c_matrix(c[i,...], "f1_score")  for i in range(n_labels)]

#     macro_accuracy  = np.mean(accuracy_per_class)
#     macro_precision = np.mean(precision_per_class)
#     macro_recall    = np.mean(recall_per_class)
#     macro_f1        = np.mean(f1_per_class)

#     summed_classes = np.sum(c, axis=0)
#     micro_accuracy  = metric_from_c_matrix(summed_classes, "accuracy")
#     micro_precision = metric_from_c_matrix(summed_classes, "precision")
#     micro_recall    = metric_from_c_matrix(summed_classes, "recall")
#     micro_f1        = metric_from_c_matrix(summed_classes, "f1_score")

#     samples_per_class = np.sum(c, axis=[1, 2])
#     total_samples = np.sum(samples_per_class)

#     weighted_accuracy   = accuracy_per_class*   samples_per_class/total_samples
#     weighted_precision  = precision_per_class*  samples_per_class/total_samples
#     weighted_recall     = recall_per_class*     samples_per_class/total_samples
#     weighted_f1         = f1_per_class*         samples_per_class/total_samples




#     return None # TODO: finish it