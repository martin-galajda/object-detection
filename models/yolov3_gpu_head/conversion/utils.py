import configparser
import io
from collections import defaultdict
from models.yolov3_gpu_head.constants import PathConstants
from typing import Dict


def load_classes(path_to_classes_file=PathConstants.YOLOV3_LABELS_FILE_PATH) -> Dict[int, str]:
    """
    Load mapping from class index to human readable class by specified file path.

    :param path_to_classes_file:
        file path to file containing labels separated by newline
    :return:
        dictionary containing mapping from number produced by the CNN to human readable class
    """
    class_idx_to_class_name = {}
    with open(path_to_classes_file, 'r') as f:
        for idx, class_name in enumerate(f):
            class_idx_to_class_name[idx] = str(class_name).strip('\n').lower()

    return class_idx_to_class_name


def unique_config_sections(config_file: str):
    """
    Convert all config sections to have unique names.

    Adds unique suffixes to config sections for compatibility with configparser.
    """
    section_counters = defaultdict(int)
    output_stream = io.StringIO()
    with open(config_file) as fin:
        for line in fin:
            if line.startswith('['):
                section = line.strip().strip('[]')
                _section = section + '_' + str(section_counters[section])
                section_counters[section] += 1
                line = line.replace(section, _section)
            output_stream.write(line)
    output_stream.seek(0)
    return output_stream


def parse_darknet_config(path_to_config_file: str):
    unique_config_file = unique_config_sections(path_to_config_file)
    cfg_parser = configparser.ConfigParser()
    cfg_parser.read_file(unique_config_file)

    for section in cfg_parser.sections():
        print('Parsing section {}'.format(section))
        print(dict(cfg_parser[section]))

    return cfg_parser
