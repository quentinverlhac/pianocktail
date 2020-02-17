import argparse
import os

from preprocessing import dump_all_songs
import config

path_list = []
for outer_path in os.listdir(config.EMOTIFY_DATA_PATH) :
  if os.path.isdir(os.path.join(config.EMOTIFY_DATA_PATH,outer_path)) :
    for inner_path in os.listdir(os.path.join(config.EMOTIFY_DATA_PATH,outer_path)) :
      path_list.append(os.path.join(config.EMOTIFY_DATA_PATH,outer_path,inner_path))

dump_all_songs(path_list,config.EMOTIFY_DUMP_PATH)
