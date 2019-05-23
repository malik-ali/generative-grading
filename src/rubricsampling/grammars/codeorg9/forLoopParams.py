import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
import numpy as np
from base import Decision, ReusableDecision


class ForLoopParams(ReusableDecision):
    def registerChoices(self):
        self.addChoice(self.getKey(), {
            'standard': 100,
            'pixel': 10,
        })

    def updateRubric(self):
        choice = self.getChoice(self.getKey())
        rubric = {
            'standard': [],
            'pixel': ['For loop: not looping by sides'],
        }
        labelList = rubric[choice]
        for label in labelList:
            self.turnOnRubric(label)
    
    def renderCode(self):
        choice = self.getChoice(self.getKey())
        self.incrementState('for_loop_params_count')

        if choice == 'standard':
            start_value_count = self.getState('start_value_count')
            end_value_count = self.getState('end_value_count')
            increment_count = self.getState('increment_count')

            template = '''
            {StartValue},{EndValue},{Increment}
            '''
            templateVars = {
                'StartValue': self.expand(
                    'StartValue',
                    {'id': 'start_value_{}'.format(start_value_count)},
                ),
                'EndValue': self.expand(
                    'EndValue',
                    {'id': 'end_value_{}'.format(end_value_count)},
                ),
                'Increment': self.expand(
                    'Increment',
                    {'id': 'increment_{}'.format(increment_count)},
                ),
            }
        elif choice == 'pixel':
            pixel_start_value_count = self.getState('pixel_start_value_count')
            pixel_end_value_count = self.getState('pixel_end_value_count')
            pixel_increment_count = self.getState('pixel_increment_count')
            
            template = '''
            {PixelStartValue},{PixelEndValue},{PixelIncrement}
            '''
            templateVars = {
                'PixelStartValue': self.expand(
                    'PixelStartValue',
                    {'id': 'pixel_start_value_{}'.format(pixel_start_value_count)},
                ),
                'PixelEndValue': self.expand(
                    'PixelEndValue',
                    {'id': 'pixel_end_value_{}'.format(pixel_end_value_count)},
                ),
                'PixelIncrement': self.expand(
                    'PixelIncrement',
                    {'id': 'pixel_increment_{}'.format(pixel_increment_count)},
                ),
            }
        else:
            raise Exception('how did you get here?')
        
        return gu.format(template, templateVars)


class StartValue(ReusableDecision):
    def registerChoices(self):
        self.addChoice(self.getKey(), {
            '3': 100,
            '2': 10,
            '1': 20,
            'random': 10,
        })

    def updateRubric(self):
        choice = self.getChoice(self.getKey())
        rubric = {
            '3': [],
            '2': ['For loop: wrong start'],
            '1': ['For loop: wrong start'],
            'random': ['For loop: wrong start'],
        }
        labelList = rubric[choice]
        for label in labelList:
            self.turnOnRubric(label)
    
    def renderCode(self):
        choice = self.getChoice(self.getKey())
        self.incrementState('start_value_count')

        if choice == 'random':
            return str(int(np.random.randint(4, 10)))
        else:
            return choice


class EndValue(ReusableDecision):
    def registerChoices(self):
        self.addChoice(self.getKey(), {
            '8': 3,
            '9': 100,
            '10': 30,
            '4': 20,
            '1': 20,
            'random': 5,
        })

    def updateRubric(self):
        choice = self.getChoice(self.getKey())
        rubric = {
            '8': ['For loop: wrong end'],
            '9': ['For loop: correct end'],
            '10': ['For loop: correct end'],
            '4': ['For loop: wrong end'],
            '1': ['For loop: wrong end'],
            'random': ['For loop: wrong end'],
        }
        labelList = rubric[choice]
        for label in labelList:
            self.turnOnRubric(label)

    def renderCode(self):
        choice = self.getChoice(self.getKey())
        self.incrementState('end_value_count')

        if choice == 'random':
            return str(int(np.random.randint(1, 20)))
        else:
            return choice


class Increment(ReusableDecision):
    def registerChoices(self):
        self.addChoice(self.getKey(), {
            '1': 10,
            '2': 100,
            '3': 10,
            'random': 100,
        })

    def updateRubric(self):
        choice = self.getChoice(self.getKey())
        rubric = {
            '1': ['For loop: wrong delta'],
            '2': [],
            '3': ['For loop: wrong delta'],
            'random': ['For loop: wrong delta'],
        }
        labelList = rubric[choice]
        for label in labelList:
            self.turnOnRubric(label)

    def renderCode(self):
        choice = self.getChoice(self.getKey())
        self.incrementState('increment_count')

        if choice == 'random':
            return str(int(np.random.randint(4, 20)))
        else:
            return choice


class PixelStartValue(ReusableDecision):
    def registerChoices(self):
        self.addChoice(self.getKey(), {
            '30': 10,
            '1': 10,
            'random': 10,
        })

    def renderCode(self):
        choice = self.getChoice(self.getKey())
        self.incrementState('pixel_start_value_count')

        if choice == 'random':
            return str(int(np.random.randint(1, 360)))
        else:
            return choice


class PixelEndValue(ReusableDecision):
    def registerChoices(self):
        self.addChoice(self.getKey(), {
            '100': 30,
            'random': 10,
        })

    def renderCode(self):
        choice = self.getChoice(self.getKey())
        self.incrementState('pixel_end_value_count')

        if choice == 'random':
            return str(int(np.random.randint(25, 360)))
        else:
            return choice


class PixelIncrement(ReusableDecision):
    def registerChoices(self):
        self.addChoice(self.getKey(), {
            '10': 40,
            '25': 40,
            'random': 5,
        })

    def renderCode(self):
        choice = self.getChoice(self.getKey())
        self.incrementState('pixel_increment_count')

        if choice == 'random':
            return str(int(np.random.randint(25, 360)))
        else:
            return choice
