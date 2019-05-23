import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
import numpy as np
from base import Decision, ReusableDecision


class ExtraLoop(ReusableDecision):
    def registerChoices(self):
        self.addChoice(self.getKey(), {
            'forSingle': 100,
            'repeatSingle': 100,
            'forDouble': 1,
            'repeatDouble': 1,
        })

    def updateRubric(self):
        choice = self.getChoice(self.getKey())
        rubric = {
            'forSingle': ['Unnecessary loop structure'],
            'repeatSingle': ['Unnecessary loop structure'],
            'forDouble': ['Unnecessary loop structure'],
            'repeatDouble': ['Unnecessary loop structure'],
        }
        labelList = rubric[choice]
        for label in labelList:
            self.turnOnRubric(label)
    
    def renderCode(self):
        choice = self.getChoice(self.getKey())
        self.incrementState('extra_loop_count')

        if choice == 'forSingle':
            for_loop_params_count = self.getState('for_loop_params_count')
            body_count = self.getState('body_count')

            template = '''
            For({ForLoopParams}) {{\n
                {Body}
            }}\n
            '''
            templateVars = {
                'ForLoopParams': self.expand(
                    'ForLoopParams',
                    {'id': 'for_loop_params_{}'.format(for_loop_params_count)},
                ),
                'Body': self.expand(
                    'Body',
                    {'id': 'body_{}'.format(body_count)},
                ),
            }
        elif choice == 'repeatSingle':
            repeat_num_count = self.getState('repeat_num_count')
            body_count = self.getState('body_count')

            template = '''
            Repeat({RepeatNum}) {{\n
                {Body}
            }}\n
            '''
            templateVars = {
                'RepeatNum': self.expand(
                    'RepeatNum',
                    {'id': 'repeat_num_{}'.format(repeat_num_count)},
                ),
                'Body': self.expand(
                    'Body',
                    {'id': 'body_{}'.format(body_count)},
                ),
            }
        elif choice == 'forDouble':
            for_loop_params_count = self.getState('for_loop_params_count')
            forLoopParamsExpansion = self.expand(
                'ForLoopParams',
                {'id': 'for_loop_params_{}'.format(for_loop_params_count)},
            )
            body_count = self.getState('body_count')
            body0Expansion = self.expand(
                'Body',
                {'id': 'body_{}'.format(body_count)},
            )
            extra_loop_count = self.getState('extra_loop_count')
            extraLoopExpansion = self.expand(
                'ExtraLoop',
                {'id': 'extra_loop_{}'.format(extra_loop_count)},
            )
            body_count = self.getState('body_count')
            body1Expansion = self.expand(
                'Body',
                {'id': 'body_{}'.format(body_count)},
            )

            template = '''
            For({ForLoopParams}) {{\n
                {Body0}
                {ExtraLoop}
                {Body1}
            }}\n
            '''
            templateVars = {
                'ForLoopParams': forLoopParamsExpansion,
                'Body0': body0Expansion,
                'ExtraLoop': extraLoopExpansion,
                'Body1': body1Expansion,
            }
        elif choice == 'repeatDouble':
            repeat_num_count = self.getState('repeat_num_count')
            repeatNumExpansion = self.expand(
                'RepeatNum',
                {'id': 'repeat_num_{}'.format(repeat_num_count)},
            )
            body_count = self.getState('body_count')
            body0Expansion = self.expand(
                'Body',
                {'id': 'body_{}'.format(body_count)},
            )
            extra_loop_count = self.getState('extra_loop_count')
            extraLoopExpansion = self.expand(
                'ExtraLoop',
                {'id': 'extra_loop_{}'.format(extra_loop_count)},
            )
            body_count = self.getState('body_count')
            body1Expansion = self.expand(
                'Body',
                {'id': 'body_{}'.format(body_count)},
            )

            template = '''
            Repeat({RepeatNum}) {{\n
                {Body0}
                {ExtraLoop}
                {Body1}
            }}\n
            '''
            templateVars = {
                'RepeatNum': repeatNumExpansion,
                'Body0': body0Expansion,
                'ExtraLoop': extraLoopExpansion,
                'Body1': body1Expansion,
            }
        else:
            raise Exception('how did you get here?')
        
        return gu.format(template, templateVars)
