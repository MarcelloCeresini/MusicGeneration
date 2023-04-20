import os
import muspy
import tensorflow as tf
import sys

import config
import utils

config_string = "single_instruments_type"
ROOT_PATH = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
print(ROOT_PATH)

#####
BACKEND_TO_CREATE_MIDI = "pretty_midi" # "mido" or "pretty_midi"
#####

conf = config.Config(config_string, ROOT_PATH)

lmd_matched = utils.get_dataset("lmd_matched", conf)
song = lmd_matched[0]
print(type(song))

if not (soundfont_path:=muspy.get_musescore_soundfont_dir()):
    muspy.download_musescore_soundfont(overwrite=True)
else:
    print(soundfont_path)

print(sys.path)
sys.path.append(str(soundfont_path))
print(sys.path)

muspy.outputs.write_midi(
    path=os.path.join(ROOT_PATH, "data", "audio", "original_"+BACKEND_TO_CREATE_MIDI+".midi"),
    music=song,
    backend=BACKEND_TO_CREATE_MIDI,
)

# here we should transform and anti-transform the song
# and then compare the original and the transformed one

# muspy.outputs.write_midi(
#     path=os.path.join(ROOT_PATH, "data", "audio", "transformed_"+BACKEND_TO_CREATE_MIDI+".midi"),
#     music=transformed_song,
#     backend=BACKEND_TO_CREATE_MIDI,
# )