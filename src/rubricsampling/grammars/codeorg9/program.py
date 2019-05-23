import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
from base import Decision


MAX_DEPTH_DICT = {
    'ForLoopParams': 10,
    'SingleShape': 10,
    'Random': 10,
    'RepeatNum': 10,
    'StartValue': 10,
    'EndValue': 10,
    'Increment': 10,
    'PixelStartValue': 10,
    'PixelEndValue': 10,
    'PixelIncrement': 10,
    'Body': 10,
    'SingleBody': 10,
    'Move': 10,
    'Turn': 10,
    'MoveAmount': 10,
    'TurnAmount': 10,
    'ExtraLoop': 10,
    'ExtraCommand': 10,
}


class Program(Decision):

    def preregisterDecisionIds(self):
        decisions = {
            'ForLoopParams': [
                'for_loop_params_{}'.format(i) for i in range(
                    MAX_DEPTH_DICT['ForLoopParams']
                )
            ],
            'SingleShape': [
                'single_shape_{}'.format(i) for i in range(
                    MAX_DEPTH_DICT['SingleShape']
                )
            ],
            'Random': [
                'random_{}'.format(i) for i in range(
                    MAX_DEPTH_DICT['Random']
                )
            ],
            'RepeatNum': [
                'repeat_num_{}'.format(i) for i in range(
                    MAX_DEPTH_DICT['RepeatNum']
                )
            ],
            'StartValue': [
                'start_value_{}'.format(i) for i in range(
                    MAX_DEPTH_DICT['StartValue']
                )
            ],
            'EndValue': [
                'end_value_{}'.format(i) for i in range(
                    MAX_DEPTH_DICT['EndValue']
                )
            ],
            'Increment': [
                'increment_{}'.format(i) for i in range(
                    MAX_DEPTH_DICT['Increment']
                )
            ],
            'PixelStartValue': [
                'pixel_start_value_{}'.format(i) for i in range(
                    MAX_DEPTH_DICT['PixelStartValue']
                )
            ],
            'PixelEndValue': [
                'pixel_end_value_{}'.format(i) for i in range(
                    MAX_DEPTH_DICT['PixelEndValue']
                )
            ],
            'PixelIncrement': [
                'pixel_increment_{}'.format(i) for i in range(
                    MAX_DEPTH_DICT['PixelIncrement']
                )
            ],
            'Body': [
                'body_{}'.format(i) for i in range(
                    MAX_DEPTH_DICT['Body']
                )
            ],
            'SingleBody': [
                'single_body_{}'.format(i) for i in range(
                    MAX_DEPTH_DICT['SingleBody']
                )
            ],
            'Move': [
                'move_{}'.format(i) for i in range(
                    MAX_DEPTH_DICT['Move']
                )
            ],
            'Turn': [
                'turn_{}'.format(i) for i in range(
                    MAX_DEPTH_DICT['Turn']
                )
            ],
            'MoveAmount': [
                'move_amount_{}'.format(i) for i in range(
                    MAX_DEPTH_DICT['MoveAmount']
                )
            ],
            'TurnAmount': [
                'turn_amount_{}'.format(i) for i in range(
                    MAX_DEPTH_DICT['TurnAmount']
                )
            ],
            'ExtraLoop': [
                'extra_loop_{}'.format(i) for i in range(
                    MAX_DEPTH_DICT['ExtraLoop']
                )
            ],
            'ExtraCommand': [
                'extra_command_{}'.format(i) for i in range(
                    MAX_DEPTH_DICT['ExtraCommand']
                )
            ],
        }
        return decisions

    def renderCode(self):
        # many random decisions can appear multiple times in a 
        # trajectory. We thus need to keep track of counts.
        self.addState('for_loop_params_count', 0)
        self.addState('single_shape_count', 0)
        self.addState('random_count', 0)
        self.addState('repeat_num_count', 0)
        self.addState('start_value_count', 0)
        self.addState('end_value_count', 0)
        self.addState('increment_count', 0)
        self.addState('pixel_start_value_count', 0)
        self.addState('pixel_end_value_count', 0)
        self.addState('pixel_increment_count', 0)
        self.addState('body_count', 0)
        self.addState('single_body_count', 0)
        self.addState('move_count', 0)
        self.addState('turn_count', 0)
        self.addState('move_amount_count', 0)
        self.addState('turn_amount_count', 0)
        self.addState('extra_loop_count', 0)
        self.addState('extra_command_count', 0)

        templateVars = {
            'MultipleShapes':self.expand('MultipleShapes'),
        }
        template = '''
        Program {MultipleShapes}
        '''
        return gu.format(template, templateVars)
