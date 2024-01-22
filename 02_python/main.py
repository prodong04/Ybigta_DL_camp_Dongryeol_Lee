import argparse
import logging
from utils.command_handler import CommandHandler
from utils.command_parser import CommandParser
# TODO 1-1: Use argparse to parse the command line arguments (verbose and log_file).
# TODO 1-2: Set up logging and initialize the logger object.
parser = argparse.ArgumentParser()
parser.add_argument('--verbose', action='store_true', help='Enable verbose mode')
parser.add_argument('--log_path', type=str, help='Specify log file path', default='file_explorer.log')
args = parser.parse_args()

log_format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s'
logging.basicConfig(filename=args.log_path, encoding='utf-8', level=logging.INFO, format=log_format)
logger = logging.getLogger(__name__)
command_parser = CommandParser(args.verbose)
handler = CommandHandler(command_parser)

logger.info('Program started')
while True:

    command = input(">> ")
    logger.info(f"input command: {command}")
    handler.execute(command)