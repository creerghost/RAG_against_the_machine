import ast  # noqa: F401
import os  # noqa: F401
from datetime import datetime  # noqa: F401


def simple_function():
    """This is a very short function."""
    print("Hello world")
    return True


class MassiveClassToTestFallback:
    """
    This class is specifically designed to be larger than the max_chunk_size
    in our test, forcing the PythonChunker to dive into its body and chunk
    these methods individually!
    """

    def first_method(self):
        return "I am the first method of the massive class"

    def second_method(self):
        return "I am the second method, and I should be my own chunk!"

    def third_method(self):
        # just some random logic to add bulk
        if True:
            return False
        return True
