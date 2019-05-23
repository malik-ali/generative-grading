import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
import numpy as np
from base import Decision, ReusableDecision


class Body(ReusableDecision):
    def registerChoices(self):
        self.addChoice(self.getKey(), {
            'correct': 100,
            'armsLength': 5,
            'noTurn': 5,
            'noMove': 5,
            'forwardBodyIncorrect': 5,
            'reverseBodyIncorrect': 5,
        })

    def updateRubric(self):
        choice = self.getChoice(self.getKey())
        rubric = {
            'correct': ['Single body (correct)'],
            'armsLength': ['Single shape: armslength'],
            'noTurn': ['Turn: no turn'],
            'noMove': ['Move: no move'],
            'forwardBodyIncorrect': ['Single shape: body incorrect'],
            'reverseBodyIncorrect': ['Single shape: body incorrect'],
        }
        labelList = rubric[choice]
        for label in labelList:
            self.turnOnRubric(label)
    
    def renderCode(self):
        _id = int(self.getKey().split('_')[-1])
        choice = self.getChoice(self.getKey())
        
        self.incrementState('body_count')

        if choice == 'correct':
            count = self.getState('single_body_count')
            template = '''
            {SingleBody}
            '''
            templateVars = {
                'SingleBody': self.expand(
                    'SingleBody',
                    {'id': 'single_body_{}'.format(count)},
                ),
            }
        elif choice == 'armsLength':
            single_body_count = self.getState('single_body_count')
            body_count = self.getState('body_count')

            singleBodyExpansion = self.expand(
                'SingleBody',
                {'id': 'single_body_{}'.format(single_body_count)},
            )

            bodyExpansion = self.expand(
                'Body',
                {'id': 'body_{}'.format(_id+1)},
            )

            template = '''
            {SingleBody}
            {Body}
            '''
            templateVars = {
                'SingleBody': singleBodyExpansion,
                'Body': bodyExpansion,
            }
        elif choice == 'noTurn':
            move_count = self.getState('move_count')
            template = '''
            {Move}\n
            '''
            templateVars = {
                'Move': self.expand(
                    'Move',
                    {'id': 'move_{}'.format(move_count)},
                ),
            }
        elif choice == 'noMove':
            turn_count = self.getState('turn_count')
            template = '''
            {Turn}\n
            '''
            templateVars = {
                'Turn': self.expand(
                    'Turn',
                    {'id': 'turn_{}'.format(turn_count)},
                ),
            }
        elif choice == 'forwardBodyIncorrect':
            # assume extraCommand and singleBody cannot lead to each other
            single_body_count = self.getState('single_body_count')
            extra_command_count = self.getState('extra_command_count')

            template = '''
            {SingleBody}
            {ExtraCommand}
            '''
            templateVars = {
                'SingleBody': self.expand(
                    'SingleBody',
                    {'id': 'single_body_{}'.format(single_body_count)},
                ),
                'ExtraCommand': self.expand(
                    'ExtraCommand',
                    {'id': 'extra_command_{}'.format(extra_command_count)},
                ),
            }
        elif choice == 'reverseBodyIncorrect':
            # assume extraCommand and singleBody cannot lead to each other
            single_body_count = self.getState('single_body_count')
            extra_command_count = self.getState('extra_command_count')

            template = '''
            {ExtraCommand}
            {SingleBody}
            '''
            templateVars = {
                'ExtraCommand': self.expand(
                    'ExtraCommand',
                    {'id': 'extra_command_{}'.format(extra_command_count)},
                ),
                'SingleBody': self.expand(
                    'SingleBody',
                    {'id': 'single_body_{}'.format(single_body_count)},
                ),
            }
        else:
            raise Exception('how did you get here?')
        
        return gu.format(template, templateVars)
