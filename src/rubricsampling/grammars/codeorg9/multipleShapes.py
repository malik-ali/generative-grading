import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
from base import Decision


class MultipleShapes(Decision):
    def registerChoices(self):
        self.addChoice('multipleShapesChoices', {
            'correct' : 70,
            'useRepeat' : 5,
            'forwardArmsLength' : 5,
            'reverseArmsLength' : 5,
            'randomArmsLength' : 5,
            'doubleArmsLength' : 2,
            'noLoop' : 10,
            'doubleShape' : 5,
        })

    def updateRubric(self):
        choice = self.getChoice('multipleShapesChoices')
        rubric = {
            'correct' : ['Correct loop structure'],
            'useRepeat' : ['For loop: repeat instead of for'],
            'forwardArmsLength' : ['For loop: armslength'],
            'reverseArmsLength' : ['For loop: armslength'],
            'randomArmsLength' : ['For loop: armslength'],
            'doubleArmsLength' : ['For loop: armslength'],
            'noLoop' : ['For loop: no loop'],
            'doubleShape' : ['For loop: armslength'],
        }
        labelList = rubric[choice]
        for label in labelList:
            self.turnOnRubric(label)
    
    def renderCode(self):
        choice = self.getChoice('multipleShapesChoices')

        if choice == 'correct':
            for_loop_params_count = self.getState('for_loop_params_count')
            forLoopParamsExpansion = self.expand(
                'ForLoopParams',
                {'id': 'for_loop_params_{}'.format(for_loop_params_count)},
            )
            single_shape_count = self.getState('single_shape_count')
            singleShapeExpansion = self.expand(
                'SingleShape',
                {'id': 'single_shape_{}'.format(single_shape_count)},
            )

            template = '''
            For({ForLoopParams}) {{\n
                {SingleShape}
            }}\n
            '''
            templateVars = {
                'ForLoopParams': forLoopParamsExpansion,
                'SingleShape': singleShapeExpansion,
            }
        elif choice == 'useRepeat':
            repeat_num_count = self.getState('repeat_num_count')
            repeatNumExpansion = self.expand(
                'RepeatNum',
                {'id': 'repeat_num_{}'.format(repeat_num_count)},
            )
            single_shape_count = self.getState('single_shape_count')
            singleShapeExpansion = self.expand(
                'SingleShape',
                {'id': 'single_shape_{}'.format(single_shape_count)},
            )

            template = '''
            Repeat({RepeatNum}) {{\n
                {SingleShape}
            }}\n
            '''
            templateVars = {
                'RepeatNum': repeatNumExpansion,
                'SingleShape': singleShapeExpansion,
            }
        elif choice == 'forwardArmsLength':
            single_shape_count = self.getState('single_shape_count')
            singleShape0Expansion = self.expand(
                'SingleShape',
                {'id': 'single_shape_{}'.format(single_shape_count)},
            )
            for_loop_params_count = self.getState('for_loop_params_count')
            forLoopParamsExpansion = self.expand(
                'ForLoopParams',
                {'id': 'for_loop_params_{}'.format(for_loop_params_count)},
            )
            single_shape_count = self.getState('single_shape_count')
            singleShape1Expansion = self.expand(
                'SingleShape',
                {'id': 'single_shape_{}'.format(single_shape_count)},
            )
            template = '''
            {SingleShape0}
            For({ForLoopParams}) {{\n
                {SingleShape1}
            }}\n
            '''
            templateVars = {
                'SingleShape0': singleShape0Expansion,
                'ForLoopParams': forLoopParamsExpansion,
                'SingleShape1': singleShape1Expansion,
            }
        elif choice == 'reverseArmsLength':
            for_loop_params_count = self.getState('for_loop_params_count')
            forLoopParamsExpansion = self.expand(
                'ForLoopParams',
                {'id': 'for_loop_params_{}'.format(for_loop_params_count)},
            )
            single_shape_count = self.getState('single_shape_count')
            singleShape0Expansion = self.expand(
                'SingleShape',
                {'id': 'single_shape_{}'.format(single_shape_count)},
            )
            single_shape_count = self.getState('single_shape_count')
            singleShape1Expansion = self.expand(
                'SingleShape',
                {'id': 'single_shape_{}'.format(single_shape_count)},
            )
            template = '''
            For({ForLoopParams}) {{\n
                {SingleShape0}
            }}\n
            {SingleShape1}
            '''
            templateVars = {
                'ForLoopParams': forLoopParamsExpansion,
                'SingleShape0': singleShape0Expansion,
                'SingleShape1': singleShape1Expansion,
            }
        elif choice == 'randomArmsLength':
            for_loop_params_count = self.getState('for_loop_params_count')
            forLoopParamsExpansion = self.expand(
                'ForLoopParams',
                {'id': 'for_loop_params_{}'.format(for_loop_params_count)},
            )
            single_shape_count = self.getState('single_shape_count')
            singleShapeExpansion = self.expand(
                'SingleShape',
                {'id': 'single_shape_{}'.format(single_shape_count)},
            )
            random_count = self.getState('random_count')
            randomExpansion = self.expand(
                'Random',
                {'id': 'random_{}'.format(random_count)},
            )
            template = '''
            For({ForLoopParams}) {{\n
                {SingleShape}
            }}\n
            {Random}
            '''
            templateVars = {
                'ForLoopParams': forLoopParamsExpansion,
                'SingleShape': singleShapeExpansion,
                'Random': randomExpansion,
            }
        elif choice == 'doubleArmsLength':
            for_loop_params_count = self.getState('for_loop_params_count')
            forLoopParams0Expansion = self.expand(
                'ForLoopParams',
                {'id': 'for_loop_params_{}'.format(for_loop_params_count)},
            )
            single_shape_count = self.getState('single_shape_count')
            SingleShape0Expansion = self.expand(
                'SingleShape',
                {'id': 'single_shape_{}'.format(single_shape_count)},
            )
            for_loop_params_count = self.getState('for_loop_params_count')
            forLoopParams1Expansion = self.expand(
                'ForLoopParams',
                {'id': 'for_loop_params_{}'.format(for_loop_params_count)},
            )
            single_shape_count = self.getState('single_shape_count')
            SingleShape1Expansion = self.expand(
                'SingleShape',
                {'id': 'single_shape_{}'.format(single_shape_count)},
            )

            template = '''
            For({ForLoopParams0}) {{\n
                {SingleShape0}
            }}\n
            For({ForLoopParams1}) {{\n
                {SingleShape1}
            }}\n
            '''
            templateVars = {
                'ForLoopParams0': forLoopParams0Expansion,
                'SingleShape0': SingleShape0Expansion,
                'ForLoopParams1': forLoopParams1Expansion,
                'SingleShape1': SingleShape1Expansion,
            }
        elif choice == 'noLoop':
            single_shape_count = self.getState('single_shape_count')
            template = '''
            {SingleShape}
            '''
            templateVars = {
                'SingleShape': self.expand(
                    'SingleShape',
                    {'id': 'single_shape_{}'.format(single_shape_count)},
                ),
            }
        elif choice == 'doubleShape':
            single_shape_count = self.getState('single_shape_count')
            SingleShape0Expansion = self.expand(
                'SingleShape',
                {'id': 'single_shape_{}'.format(single_shape_count)},
            )
            single_shape_count = self.getState('single_shape_count')
            SingleShape1Expansion = self.expand(
                'SingleShape',
                {'id': 'single_shape_{}'.format(single_shape_count)},
            )

            template = '''
            {SingleShape0}
            {SingleShape1}
            '''
            templateVars = {
                'SingleShape0': SingleShape0Expansion,
                'SingleShape1': SingleShape1Expansion,
            }
        else:
            raise Exception('how did you get here?')
        
        return gu.format(template, templateVars)

