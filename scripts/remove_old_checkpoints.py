import argparse
from utils.remove_old_checkpoints import remove_old_checkpoints

def main(args):
  checkpoint_dir = args.checkpoint_dir

  remove_old_checkpoints(checkpoint_dir)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Retrain pretrained vision network for openimages.')

  parser.add_argument('--checkpoint_dir', type=str, choices=['./checkpoints/inceptionV3/2000'], required=True,
                      help='Directory containing checkpoints for models to remove.')
  args = parser.parse_args()

  main(args)
