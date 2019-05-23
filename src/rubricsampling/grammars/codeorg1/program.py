import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
from base import Decision


MAX_DEPTH_DICT = {
    'RandomStrategy': 20,
    'SingleBody': 10,
    'SingleClockwiseBody': 10,
    'RepeatedBody': 10,
    'Turn': 10,
    'TurnAmount': 10,
    'Move': 10,
    'MoveAmount': 10,
    'ClockwiseTurn': 10,
    'RepeatNum': 10,
}


class Program(Decision):

    def preregisterDecisionIds(self):
        # these decisions are used in different ways
        decisions =  {
            'RandomStrategy': [
                'random_strategy_{}'.format(i) for i in range(
                    MAX_DEPTH_DICT['RandomStrategy'])
            ],
            'SingleBody': [
                'single_body_{}'.format(i) for i in range(
                    MAX_DEPTH_DICT['SingleBody']
                )
            ],
            'SingleClockwiseBody': [
                'single_clockwise_body_{}'.format(i) for i in range(
                    MAX_DEPTH_DICT['SingleClockwiseBody']
                )
            ],
            'RepeatedBody': [
                'repeated_body_{}'.format(i) for i in range(
                    MAX_DEPTH_DICT['RepeatedBody']
                )
            ],
            'RepeatNum': [
                'repeat_num_{}'.format(i) for i in range(
                    MAX_DEPTH_DICT['RepeatNum']
                )
            ],
            'Turn': [
                'turn_{}'.format(i)  for i in range(
                    MAX_DEPTH_DICT['Turn']
                )
            ],
            'TurnAmount': [
                'turn_amount_{}'.format(i)  for i in range(
                    MAX_DEPTH_DICT['TurnAmount']
                )
            ],
            'Move': [
                'move_{}'.format(i)  for i in range(
                    MAX_DEPTH_DICT['Move']
                )
            ],
            'MoveAmount': [
                'move_amount_{}'.format(i)  for i in range(
                    MAX_DEPTH_DICT['MoveAmount']
                )
            ],
            'ClockwiseTurn': [
                'clockwise_turn_{}'.format(i)  for i in range(
                    MAX_DEPTH_DICT['ClockwiseTurn']
                )
            ],
        }
        return decisions

    def renderCode(self):
        # many random decisions can appear in many places so
        # we must keep track of counts
        self.addState('random_strategy_count', 0)
        self.addState('single_body_count', 0)
        self.addState('single_clockwise_body_count', 0)
        self.addState('repeated_body_count', 0)
        self.addState('repeat_num_count', 0)
        self.addState('turn_count', 0)
        self.addState('move_count', 0)
        self.addState('turn_amount_count', 0)
        self.addState('move_amount_count', 0)
        self.addState('clockwise_turn_count', 0)

        templateVars = {
            'Strategy': self.expand('Strategy'),
        }
        template = '''
        Program {Strategy}
        '''
        
        return gu.format(template, templateVars)
