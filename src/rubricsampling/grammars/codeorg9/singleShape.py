import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
from base import Decision, ReusableDecision


class SingleShape(ReusableDecision):
    def registerChoices(self):
        self.addChoice(self.getKey(), {
            'correct' : 100,
            'missingRepeat' : 10,
            'moveNestingIssue' : 10,
            'turnNestingIssue' : 10,
            'forwardArmsLength' : 10,
            'reverseArmsLength' : 10,
            'doubleArmsLength' : 2,
        })

    def updateRubric(self):
        choice = self.getChoice(self.getKey())
        rubric = {
            'correct' : ['Correct inner loop structure'],
            'missingRepeat' : ['Single shape: missing repeat'],
            'moveNestingIssue' : ['Single shape: nesting issue'],
            'turnNestingIssue' : ['Single shape: nesting issue'],
            'forwardArmsLength' : ['Single shape: armslength'],
            'reverseArmsLength' : ['Single shape: armslength'],
            'doubleArmsLength' : ['Single shape: armslength'],
        }
        labelList = rubric[choice]
        for label in labelList:
            self.turnOnRubric(label)
    
    def renderCode(self):
        choice = self.getChoice(self.getKey())
        self.incrementState('single_shape_count')

        if choice == 'correct':
            body_count = self.getState('body_count')
            repeat_num_count = self.getState('repeat_num_count')

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
        elif choice == 'missingRepeat':
            body_count = self.getState('body_count')
            template = '''
            {Body}
            '''
            templateVars = {
                'Body': self.expand(
                    'Body',
                    {'id': 'body_{}'.format(body_count)},
                ),
            }
        elif choice == 'moveNestingIssue':
            move_count = self.getState('move_count')
            repeat_num_count = self.getState('repeat_num_count')
            turn_count = self.getState('turn_count')

            template = '''
            {Move}
            Repeat({RepeatNum}) {{\n
                {Turn}
            }}\n
            '''
            templateVars = {
                'Move': self.expand(
                    'Move',
                    {'id': 'move_{}'.format(move_count)},
                ),
                'RepeatNum': self.expand(
                    'RepeatNum',
                    {'id': 'repeat_num_{}'.format(repeat_num_count)},
                ),
                'Turn': self.expand(
                    'Turn',
                    {'id': 'turn_{}'.format(turn_count)},
                ),
            }
        elif choice == 'turnNestingIssue':
            repeat_num_count = self.getState('repeat_num_count')
            move_count = self.getState('move_count')
            turn_count = self.getState('turn_count')

            template = '''
            Repeat({RepeatNum}) {{\n
                {Move}
            }}\n
            {Turn}
            '''
            templateVars = {
                'RepeatNum': self.expand(
                    'RepeatNum',
                    {'id': 'repeat_num_{}'.format(repeat_num_count)},
                ),
                'Move': self.expand(
                    'Move',
                    {'id': 'move_{}'.format(move_count)},
                ),
                'Turn': self.expand(
                    'Turn',
                    {'id': 'turn_{}'.format(turn_count)},
                ),
            }
        elif choice == 'forwardArmsLength':
            body_count = self.getState('body_count')
            body0Expansion = self.expand(
                'Body',
                {'id': 'body_{}'.format(body_count)},
            )
            repeat_num_count = self.getState('repeat_num_count')
            repeatNumExpansion = self.expand(
                'RepeatNum',
                {'id': 'repeat_num_{}'.format(repeat_num_count)},
            )
            body_count = self.getState('body_count')
            body1Expansion = self.expand(
                'Body',
                {'id': 'body_{}'.format(body_count)},
            )

            template = '''
            {Body0}
            Repeat({RepeatNum}) {{\n
                {Body1}
            }}\n
            '''
            templateVars = {
                'Body0': body0Expansion,
                'RepeatNum': repeatNumExpansion,
                'Body1': body1Expansion,
            }
        elif choice == 'reverseArmsLength':
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
            body_count = self.getState('body_count')
            body1Expansion = self.expand(
                'Body',
                {'id': 'body_{}'.format(body_count)},
            )

            template = '''
            Repeat({RepeatNum}) {{\n
                {Body0}
            }}\n
            {Body1}
            '''
            templateVars = {
                'RepeatNum': repeatNumExpansion,
                'Body0': body0Expansion,
                'Body1': body1Expansion,
            }
        elif choice == 'doubleArmsLength':
            repeat_num_count = self.getState('repeat_num_count')
            repeatNum0Expansion = self.expand(
                'RepeatNum',
                {'id': 'repeat_num_{}'.format(repeat_num_count)},
            )
            body_count = self.getState('body_count')
            body0Expansion = self.expand(
                'Body',
                {'id': 'body_{}'.format(body_count)},
            )
            repeat_num_count = self.getState('repeat_num_count')
            repeatNum1Expansion = self.expand(
                'RepeatNum',
                {'id': 'repeat_num_{}'.format(repeat_num_count)},
            )
            body_count = self.getState('body_count')
            body1Expansion = self.expand(
                'Body',
                {'id': 'body_{}'.format(body_count)},
            )

            template = '''
            Repeat({RepeatNum0}) {{\n
                {Body0}
            }}\n
            Repeat({RepeatNum1}) {{\n
                {Body1}
            }}\n
            '''
            templateVars = {
                'RepeatNum0': repeatNum0Expansion,
                'Body0': body0Expansion,
                'RepeatNum1': repeatNum1Expansion,
                'Body1': body1Expansion,
            }
        else:
            raise Exception('how did you get here?')
        
        return gu.format(template, templateVars)
