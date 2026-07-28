import sys


class Application:
    def __init__(self):
        pass

    def run(self):
        try:
            print("hi")
        except Exception as e:
            print(f"Error: {e}")
            sys.quit(1)
