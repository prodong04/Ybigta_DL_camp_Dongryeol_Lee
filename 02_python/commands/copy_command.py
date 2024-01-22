from .base_command import BaseCommand
import os
import shutil
import logging
from typing import List

logger = logging.getLogger(__name__)

class CopyCommand(BaseCommand):
    def __init__(self, options: List[str], args: List[str]) -> None:
        """
        Initialize the CopyCommand object.

        Args:
            options (List[str]): List of command options.
            args (List[str]): List of command arguments.
        """
        super().__init__(options, args)

        # Override the attributes inherited from BaseCommand
        self.description = 'Copy a file or directory to another location'
        self.usage = 'Usage: cp [source] [destination]'

        # TODO 6-1: Initialize any additional attributes you may need.
        # Refer to list_command.py, grep_command.py to implement this.
        # ...
        self.source_path = self.args[0] if len(self.args) > 0 else None
        self.destination_path = self.args[1] if len(self.args) > 1 else None

    def execute(self) -> None:
        """
        Execute the copy command.
        Supported options:
            -i: Prompt the user before overwriting an existing file.
            -v: Enable verbose mode (print detailed information)
        
        TODO 6-2: Implement the functionality to copy a file or directory to another location.
        You may need to handle exceptions and print relevant error messages.
        You may use the file_exists() method to check if the destination file already exists.
        """
        # Your code here
        # Check if the correct number of arguments is provided
        if len(self.args) < 2:
            print("Usage: cp [source] [destination]")
            logger.error("arguments are not enough")
            return
        
        if not os.path.exists(self.source_path):
            print(f"cp: cannot copy '{self.source_path}': No such file or directory")
            logger.error(f"cp: cannot copy '{self.source_path}': No such file or directory")
            return
        
        if self.file_exists(self.destination_path, self.source_path):
            if '-v' in self.options:
                print("cp: copying '{self.source_path}' to '{self.destination_path}'")
                # Add more verbose information if needed
                logger.info(f"cp: copying '{self.source_path}' to '{self.destination_path}'")
            if '-i' in self.options:
                overwrite = input(f"cp: overwrite '{self.destination_path}'? (y/n): ")
                logger.info(f"cp: overwrite to '{self.destination_path}'")
                if overwrite.lower() != 'y':
                    print("cp: operation canceled")
                    logger.error("cp: operation canceled")
                    return
        

        try:
            # Perform the copy operation
            shutil.copy(self.source_path, self.destination_path)
            print(f"Copied '{self.source_path}' to '{self.destination_path}'")
            logger.info(f"Copied '{self.source_path}' to '{self.destination_path}'")

            

        except Exception as e:
            print(f"cp: an error occurred during the copy operation: {e}")
            logger.error(f"cp: an error occurred during the copy operation: {e}")

        
        

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
