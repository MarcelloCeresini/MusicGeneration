from typing import Tuple
import numpy as np
import muspy
import os, shutil, tarfile
from tqdm import tqdm
import tensorflow as tf

import config

def get_dataset_splits(path: str, conf: config.Config) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    whole_dataset = tf.data.Dataset.load(path)
    
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    whole_dataset = whole_dataset.with_options(options)
    
    train_split = int(len(whole_dataset)/10)*8                  # 80%
    val_split = int(len(whole_dataset)/10)*8                    # 10%
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


def add_notes(final_song: list, notes: np.array, settings: dict, t_init: int, measure_init: int, time_interval: int, conf: config.Config, debug=False):
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

    while i < len(notes):
        note = notes[i]
        i+=1

        if note[0] >= t_init + time_interval:
            break # should go into the second return

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
                    ((note[0] - t_init) % settings["measure"]) % settings["beat"])),
                np.argmin(np.abs(conf.np_durations - note[2])),
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
            [],
            t_init + time_interval,
            final_song[-1][1] # the final measure --> cannot be more than 255
        )
    
    else:
        return (
            final_song,
            notes[i:],
            t_init + time_interval,
            measure_init + (time_interval / settings["measure"])
        )


def transform_representation(song: muspy.music.Music, conf: config.Config, verbose=0) -> np.ndarray:

    '''
    This function accepts as input a song in the muspy format and transforms it in a series of tuples written in the following way

                            Description                                                     Interval                                                Possible values
    (
        0-type,               # see below -->                                                 [0, 7] -->                                              8
        1-measure,            # index of the measure inside the song in interval -->          [0, last_measure] -->                                   512?
        2-beat,               # index of beat inside measure -->                              [0, numerator of time_signature] -->                    132 (max numerator allowed)
        3-position,           # index with 1/64 beat length granularity -->                   [0, 63/64] -->                                          64
        4-duration,           # hierarchical structure? -->                                   [0, ~50] -->                                            136                                  
        5-pitch,              # height of pitch (128) + drums (another 128) -->               [0, 255] -->                                            256
        6-instrument_type,    # 128 instrument types dr -->                                   [0, 128] -->                                            129
        7-velocity,           # amplitude of note, strength of play -->                       [0, 127] -->                                            128
        8-key_sign,           # [0,11] (all possible notes) and [maj,min] -->                 [0, 23] -->                                             24
        9-time_sign,          # denominator pow(2) in [1,64] and numerator int in [0,128] --> ??? better to specify after dataset exploration
        10-tempo,              # qpm, geometric progression from 16 to 256 -->                 [0, 48] -->                                             49
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
    REJECT_SONG = False

    final_song = []

    # start of song
    final_song.append(tuple([0]*conf.tuple_size))

    if all([True if len(track.notes)<=0 else False for track in song.tracks]): 
        return [0]

    for track in song.tracks:
        programs = []
        if len(track.notes)>0 and (track.program not in set(programs)):
            # different conf_string may change this ####################################
            # add each instrument with (1, 0, 0, 0, 0, 0, program, 0, 0, 0, 0)
            final_song.append(tuple([1]+ [0]*5 +[track.program]+ [0]*4 ))

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

    j = 0
    for track in song.tracks:
        if track.is_drum:
            for i, note in enumerate(track.notes):
                notes[j+i, 0] = note.time
                notes[j+i, 1] = note.pitch + 128
                notes[j+i, 2] = note.duration
                notes[j+i, 3] = note.velocity
                notes[j+i, 4] = track.program
        else:
            for i, note in enumerate(track.notes):
                notes[j+i, 0] = note.time
                notes[j+i, 1] = note.pitch
                notes[j+i, 2] = note.duration
                notes[j+i, 3] = note.velocity
                notes[j+i, 4] = track.program
        
        j += len(track.notes)

    # sort them by the 0th column, the time
    notes = notes[notes[:,0].argsort()]

    # number of timesteps per beat --> absolute_time = (60*metrical_time)/(tempo_qpm*resolution)
    resolution = song.resolution

    current_beat_length = resolution

    # n?? of timesteps for each measure = resolution * nominator of time_sign (n?? of beats in a measure)
    current_measure_length = current_time_sign[0] * resolution
    
    current_settings = {
        "key_sign": current_key_sign,       # tuple with (key, major/minor)
        "time_sign": current_time_sign,     # tuple (nominator, denominator)
        "tempo": current_tempo,             # float
        "resolution": resolution,           # int
        "measure": current_measure_length,  # int
        "beat": current_beat_length         # int
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
        
        if event[1] == "tempo":
            if event[2] != current_settings["tempo"]:

                new_settings["tempo"] = event[2] # qpm

                current_event = tempo_repr(
                    new_settings["tempo"],
                    (event[0] - current_time) // current_settings["measure"] + current_measure_index,
                    conf
                )
                flag_new_event = True


        if flag_new_event:
            
            if (delta_t := (event[0]-current_time) % current_measure_length) != 0:
                # print(song.metadata)
                # print(events)
                pass

            # if the event happens in the middle of a beat because midi is "wrong" --> move the event to the beginning of THAT measure (not the following one)
            # every note will be positioned with respect to this new timestep (measures, beats and positions are shifted)
            time_interval = (event[0] - delta_t) - current_time
            assert time_interval >= 0, "time interval: {}".format(time_interval)

            # shouldn't do anything if there are no notes between current time and t+delta_t
            try:                
                final_song, notes, current_time, current_measure_index = add_notes( 
                    final_song,
                    notes,
                    current_settings, 
                    current_time,
                    current_measure_index,
                    time_interval,
                    conf
                )

            except:
                print(events)
                print(notes[-10:])
                print(song.metadata)


            current_settings = new_settings.copy()
            # beat is constant, onle measure length changes --> update it
            final_song.append(current_event)
            
    ### sometimes events have timesteps > than last note!

    try:

        final_song, _, _, current_measure_index = add_notes(
            final_song,
            notes,
            current_settings,
            current_time,
            current_measure_index,
            1e15, # just to be safe TODO improve it
            conf,
            debug=True
        ) # should append every note between current note and note finish

    except:
        print(final_song)
        print(notes[0])
        print(current_time)
        print(song.metadata)
    
    # we save npy in uint8, so all the parts of the tuples must be <258
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

