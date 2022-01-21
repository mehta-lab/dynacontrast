from utils.patch_VAE import pool_datasets
import argparse
from utils.config_reader import YamlReader

def main():
    arguments = parse_args()
    config = YamlReader()
    config.read_config(arguments.config)
    pool_datasets(config)


def parse_args():
    """
    Parse command line arguments for CLI.

    :return: namespace containing the arguments passed.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-c', '--config',
        type=str,
        required=True,
        help='path to yaml configuration file'
    )
    return parser.parse_args()


if __name__ == '__main__':
    main()

