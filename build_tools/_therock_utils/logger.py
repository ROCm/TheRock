import sys 

class TheRockLogger: 
    def __init__(self):
        pass

    @staticmethod
    def log(*args, **kwargs):
        print(*args, **kwargs)
        sys.stdout.flush()
