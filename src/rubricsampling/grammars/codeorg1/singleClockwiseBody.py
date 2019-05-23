import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
from base import Decision, ReusableDecision


class SingleClockwiseBody(ReusableDecision):
    def registerChoices(self):
        choice_name = self.getKey()
        self.addChoice(choice_name, {
            'correctBodyOrder' : 100,
            'incorrectBodyOrder' : 40,
            'missingPrePost' : 15,
            'missingMove': 30,
            'missingTurn': 30,
            'extraRandom': 5,
            'random': 5
        })

    def updateRubric(self):
        choice_name = self.getKey()
        choice = self.getChoice(choice_name)
        rubric = {
            'correctBodyOrder' : [],
            'incorrectBodyOrder' : ['Body order is incorrect (turn/move)'],
            'missingPrePost' : ['Does not get pre/post condition'],
            'missingMove': [],
            'missingTurn': [],
            'extraRandom': [],
            'random': [],
        }
        labelList = rubric[choice]
        for label in labelList:
            self.turnOnRubric(label)
    
    def renderCode(self):
        _id = int(self.getKey().split('_')[-1])
        choice = self.getChoice(self.getKey())
        self.incrementState('single_clockwise_body_count')

        if choice == 'correctBodyOrder':
            move_cnt = self.getState('move_count')
            turn_cnt = self.getState('clockwise_turn_count')

            template = '''
            {Move}
            {ClockwiseTurn}
            '''
            templateVars = {   
                'Move': self.expand(
                    'Move', {'id': 'move_{}'.format(move_cnt)}),
                'ClockwiseTurn': self.expand(
                    'ClockwiseTurn', {'id': 'clockwise_turn_{}'.format(turn_cnt)}),
            }
        elif choice == 'incorrectBodyOrder':
            move_cnt = self.getState('move_count')
            turn_cnt = self.getState('clockwise_turn_count')
            
            template = '''
            {ClockwiseTurn}
            {Move}
            '''
            templateVars = {  
                'Move': self.expand(
                    'Move', {'id': 'move_{}'.format(move_cnt)}),
                'ClockwiseTurn': self.expand(
                    'ClockwiseTurn', {'id': 'clockwise_turn_{}'.format(turn_cnt)}), 
            }
        elif choice == 'missingPrePost':
            move_cnt = self.getState('move_count')
            moveExpansion = self.expand(
                'Move', {'id': 'move_{}'.format(move_cnt)})
            turn_cnt = self.getState('clockwise_turn_count')
            clockwiseTurn0Expansion = self.expand(
                'ClockwiseTurn', {'id': 'clockwise_turn_{}'.format(turn_cnt)})
            turn_cnt = self.getState('clockwise_turn_count')
            clockwiseTurn1Expansion = self.expand(
                'ClockwiseTurn', {'id': 'clockwise_turn_{}'.format(turn_cnt)})

            template = '''
            {ClockwiseTurn0}
            {Move}
            {ClockwiseTurn1}
            '''
            templateVars = {  
                'Move': moveExpansion,
                'ClockwiseTurn0': clockwiseTurn0Expansion, 
                'ClockwiseTurn1': clockwiseTurn1Expansion, 
            }
        elif choice == 'missingMove':
            move_cnt = self.getState('move_count')
            template = '''
            {Move}
            '''
            templateVars = {   
                'Move': self.expand(
                    'Move', {'id': 'move_{}'.format(move_cnt)}),
            }
        elif choice == 'missingTurn':
            turn_cnt = self.getState('clockwise_turn_count')
            template = '''
            {ClockwiseTurn}
            '''
            templateVars = {   
                'ClockwiseTurn': self.expand(
                    'ClockwiseTurn', {'id': 'clockwise_turn_{}'.format(turn_cnt)}),
            }
        elif choice == 'extraRandom':
            singleClockwiseBodyExpansion = self.expand(
                'SingleClockwiseBody',
                {'id': 'single_clockwise_body_{}'.format(_id+1)},
            )
            random_cnt = self.getState('random_strategy_count')
            randomStrategyExpansion = self.expand(
                'RandomStrategy',
                {'id': 'random_strategy_{}'.format(random_cnt)},
            )
            template = '''
            {SingleClockwiseBody}
            {RandomStrategy}
            '''
            templateVars = {   
                'SingleClockwiseBody': singleClockwiseBodyExpansion,
                'RandomStrategy': randomStrategyExpansion,
            }
        elif choice == 'random':
            random_cnt = self.getState('random_strategy_count')
            template = '''
            {RandomStrategy}
            '''
            templateVars = {   
                'RandomStrategy': self.expand(
                    'RandomStrategy',
                    {'id': 'random_strategy_{}'.format(random_cnt)}
                ),
            }
        else:
            raise Exception('how did you get here?')
        
        return gu.format(template, templateVars)

