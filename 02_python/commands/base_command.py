# commands/base_command.py
import os
from typing import List

"""
TODO 3-1: The BaseCommand class has a show_usage method implemented, but the execute method is not 
implemented and is passed on to the child class. Think about why this difference is made.

Answer (You may write your answer in either Korean or English):


TODO 3-2: The update_current_path method of the BaseCommand class is slightly different from other methods. 
It has a @classmethod decorator and takes a cls argument instead of self. In Python, this is called a 
class method, and think about why it was implemented as a class method instead of a normal method.

Answer (You may write your answer in either Korean or English):

"""
class BaseCommand:
    """
    Base class for all commands. Each command should inherit from this class and 
    override the execute() method.
    
    For example, the MoveCommand class overrides the execute() method to implement 
    the mv command.

    Attributes:
        current_path (str): The current path. Usefull for commands like ls, cd, etc.
    """

    current_path = os.getcwd()

    @classmethod
    def update_current_path(cls, new_path: str):
        """
        Update the current path.
        You need to understand how class methods work.

        Args:
            new_path (str): The new path. (Must be an relative path)
        """
        BaseCommand.current_path = os.path.join(BaseCommand.current_path, new_path)
        '''
        클래스 메서드는 클래스 단위에 접근할 때 사용됩니다. 
        일반 정적 메서드는 인스턴스 단위에서 사용이 가능한데, 
        클래스 메서드는 클래스의 변수에도 접근이 가능합니다. 
        따라서, current_path를 바꾸기 위해선 클래스 단위로 접근이 가능한
        클래스 메서드를 사용해야 합니다.
        '''

    def __init__(self, options: List[str], args: List[str]) -> None:
        """
        Initialize a new instance of BaseCommand.

        Args:
            options (List[str]): The command options (e.g. -v, -i, etc.)
            args (List[str]): The command arguments (e.g. file names, directory names, etc.)
        """
        self.options = options
        self.args = args
        self.description = 'Helpful description of the command'
        self.usage = 'Usage: command [options] [arguments]'

    def show_usage(self) -> None:
        """
        Show the command usage.
        """
        print(self.description)
        print(self.usage)

    def execute(self) -> None:
        """
        Execute the command. This method should be overridden by each subclass.
        """
        raise NotImplementedError
        '''
        execute() 메소드를 구현하지 않은 이유는, 이 base command 클래스를 
        상속 받아서 구현하는 move copy 등등 각각의 클래스에서 execute()
        구현 방법이 모두 다르기 때문이며, 이를 위해 base command 클래스에서 추상 메소드로 남겨 놓고 넘어가야 합니다. 
        '''