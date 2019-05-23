import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
import numpy as np
from base import Decision, ReusableDecision


class ExtraCommand(ReusableDecision):
    def registerChoices(self):
        self.addChoice(self.getKey(), {
            'turn': 100,
            'move': 100,
        })
    
    def renderCode(self):
        choice = self.getChoice(self.getKey())
        self.incrementState('extra_command_count')

        if choice == 'turn':
            count = self.getState('turn_count')
            template = '''
            {Turn}\n
            '''
            templateVars = {
                'Turn': self.expand(
                    'Turn', 
                    {'id': 'turn_{}'.format(count)},
                ),
            }
        elif choice == 'move':
            count = self.getState('move_count')
            template = '''
            {Move}\n
            '''
            templateVars = {
                'Move': self.expand(
                    'Move',
                    {'id': 'move_{}'.format(count)},
                ),
            }
        else:
            raise Exception('how did you get here?')
        
        return gu.format(template, templateVars)
