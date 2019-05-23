import pickle

class ProgramNN:
    """
     Base class representing an algorithm for finding the nearest
     in-grammar neighbour of a real student program submission.
    """

    def findNearestNeighbours(self, studentProgram, **kwargs):
        raise NotImplementedError('Method findNearestNeighbor needs to be implemented by subclass')
