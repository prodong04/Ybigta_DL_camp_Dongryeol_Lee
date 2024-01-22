# utils/command_parser.py
import logging
from typing import Dict, Any
logger = logging.getLogger(__name__)
#basicConfig는 최상단 레벨에서만 사용하면 된다. 
"""
TODO 10-1: Add a logger object.
Log command name, options, and arguments.

Logging format:
f"Command name: {command_name}"
f"Options: {options}"
f"Positional args: {positional_args}"
"""

class CommandParser:
    """
    A class for parsing commands and extracting command name, options, and arguments.

    Args:
        verbose (bool, optional): If True, print additional information during parsing. Defaults to False.

    Methods:
        parse_command(input_command: str) -> Dict[str, Any]:
            Parses the input command and returns a dictionary containing the command name, options, and arguments.

    """

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
    
    def parse_command(self, input_command: str) -> Dict[str, Any]:
        # Split the input command into tokens
        tokens = input_command.split()

        # Extract the command name (the first token)
        command_name = tokens[0]

        # Initialize lists for options and arguments
        options = []
        args = []

        # Iterate over the tokens starting from the second one
        for token in tokens[1:]:
            # Check if the token is an option (starts with '-')
            if token.startswith('-'):
                options.append(token)
            else:
                # If not an option, consider it as an argument
                args.append(token)
        logger.info(f"Command name: {command_name}")
        logger.info(f"Options: {options}")
        logger.info(f"Positional args: {args}")
        return {
            'command_name': command_name,
            'options': options,
            'args': args
        }

