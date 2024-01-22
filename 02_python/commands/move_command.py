from .base_command import BaseCommand
import os
import shutil
from typing import List
import logging
logger = logging.getLogger(__name__)

class MoveCommand(BaseCommand):
    def __init__(self, options: List[str], args: List[str]) -> None:
        """
        Initialize the MoveCommand object.

        Args:
            options (List[str]): List of command options.
            args (List[str]): List of command arguments.
        """
        super().__init__(options, args)

        # Override the attributes inherited from BaseCommand
        self.description = 'Move a file or directory to another location'
        self.usage = 'Usage: mv [source] [destination]'

        # TODO 5-1: Initialize any additional attributes you may need.
        self.source_path = args[0] if args else None
        self.destination_path = args[1] if len(args) > 1 else None

    def execute(self) -> None:
        """
        Execute the move command.
        Supported options:
            -i: Prompt the user before overwriting an existing file.
            -v: Enable verbose mode (print detailed information)
        """
        # Parse options
        prompt_before_overwrite = '-i' in self.options
        verbose_mode = '-v' in self.options

        # Check if source and destination arguments are provided
        if len(self.args) < 2:
            print("Error: Both source and destination arguments are required.")
            logger.error("arguments are not enough")
            self.show_usage()
            return

        source_path = self.args[0]
        destination_path = self.args[1]
        # Check if the source file or directory exists
        if not os.path.exists(source_path):
            print(f"Error: Source '{source_path}' does not exist.")
            logger.error(f"Error: Source '{source_path}' does not exist.")
            return

        # Check if the destination directory exists
        #destination_directory = os.path.dirname(destination_path)
        #맥북 환경에서 path가 앞 경로가 안붙어서 나와서, path 그대로 가져다 썼습니다. ㅂ
        if not os.path.exists(destination_path):
            print(f"Error: Destination directory '{destination_path}' does not exist.")
            logger.error(f"Error: Destination directory '{destination_path}' does not exist.")
            return

        # Check if the destination file or directory already exists
        if os.path.exists(destination_path):
            if prompt_before_overwrite:
                user_input = input(f"File '{destination_path}' already exists. Overwrite? (y/n): ")
                if user_input.lower() != 'y':
                    print("Move operation canceled.")
                    logger.error("Move operation canceled.")
                    return
            else:
                print(f"Warning: File '{destination_path}' already exists. Overwriting...")
                logger.warning(f"Warning: File '{destination_path}' already exists. Overwriting...")
        # Move the file or directory
        try:
            shutil.move(source_path, destination_path)
            if verbose_mode:
                print(f"mv: Moving '{source_path}' to '{destination_path}'.")
                logger.info(f"mv: Moving '{source_path}' to '{destination_path}'.")
        except Exception as e:
            print(f"Error: Failed to move '{source_path}' to '{destination_path}'.")
            print(f"Reason: {e}")
            logger.error(f"Error: Failed to move '{source_path}' to '{destination_path}'.")

    
    def file_exists(self, directory: str, file_name: str) -> bool:
        """
        Check if a file exists in a directory.
        Feel free to use this method in your execute() method.

        Args:
            directory (str): The directory to check.
            file_name (str): The name of the file.

        Returns:
            bool: True if the file exists, False otherwise.
        """
        file_path = os.path.join(directory, file_name)
        return os.path.exists(file_path)
