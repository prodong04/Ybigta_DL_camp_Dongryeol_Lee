from commands.base_command import BaseCommand
import os
import shutil
from typing import List
import logging
logger = logging.getLogger(__name__)
class ChangeDirectoryCommand(BaseCommand):
    def __init__(self, options: List[str], args: List[str]) -> None:
        """
        Initialize the ChangeDirectoryCommand object.

        Args:
            options (List[str]): List of command options.
            args (List[str]): List of command arguments.
        """
        super().__init__(options, args)

        # Override the attributes inherited from BaseCommand
        self.description = 'Change the current working directory'
        self.usage = 'Usage: cd [options] [directory]'

        # TODO 7-1: Initialize any additional attributes you may need.
        # Refer to list_command.py, grep_command.py to implement this.
        # ...
        self.togo = self.args[0] if self.args else None

    def execute(self) -> None:
        """
        Execute the cd command.
        Supported options:
            -v: Enable verbose mode (print detailed information)
        
        TODO 7-2: Implement the functionality to change the current working directory.
        You may need to handle exceptions and print relevant error messages.
        """
        # Your code here
        try:
            # 현재 작업 디렉터리 변경
            if self.args:
                os.chdir(self.togo)
                print(f"Succefully changed to '{os.getcwd()}'")
                logger.info(f"Succefully changed to '{os.getcwd()}'")
            else:
                print("사용법: cd [directory]")

            if '-v' in self.options:
                print(f"cd: changing directory to {self.togo}")
                # 필요한 경우 더 많은 자세한 정보를 추가하세요

        except Exception as e:
            if '-v' in self.options:
                print(f"cd: Fail changing directory to {self.togo}")
            print(f"cd: cannot change directory to {self.togo}: {e}")
            logger.error(f"cd: cannot change directory to {self.togo}: {e}")