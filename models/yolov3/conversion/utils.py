import configparser
import io
from collections import defaultdict


def load_classes(path_to_classes_file = './resources/yolov3/openimages.names'):
    class_idx_to_class_name = {}
    with open(path_to_classes_file, 'r') as f:
        for idx, class_name in enumerate(f):
            class_idx_to_class_name[idx] = class_name

    return class_idx_to_class_name


def unique_config_sections(config_file):
    """Convert all config sections to have unique names.

    Adds unique suffixes to config sections for compability with configparser.
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


def parse_darknet_config(path_to_config_file):
    unique_config_file = unique_config_sections(path_to_config_file)
    cfg_parser = configparser.ConfigParser()
    cfg_parser.read_file(unique_config_file)

    for section in cfg_parser.sections():
        print('Parsing section {}'.format(section))
        print(dict(cfg_parser[section]))

    return cfg_parser
