import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
from base import Decision, ReusableDecision


class SingleBody(ReusableDecision):
    def registerChoices(self):
        choice_name = self.getKey()
        self.addChoice(choice_name, {
            'correctBodyOrder' : 100,
            'incorrectBodyOrder' : 50,
            'missingPrePost' : 30,
            'missingMove': 30,
            'missingTurn': 30,
            'extraRandom': 15,
            'random': 5,
        })

    def updateRubric(self):
        choice_name = self.getKey()
        choice = self.getChoice(choice_name)
        rubric = {
            'correctBodyOrder': [],
            'incorrectBodyOrder': ['Body order is incorrect (turn/move)'],
            'missingPrePost': ['Does not get pre/post condition',
                               'Body order is incorrect (turn/move)'],
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
        self.incrementState('single_body_count')

        if choice == 'correctBodyOrder':
            move_cnt = self.getState('move_count')
            turn_cnt = self.getState('turn_count')
            
            template = '''
            {Move}
            {Turn}
            '''
            # Move and Turn are independent, so we don't need
            # to do sequential expansions
            templateVars = {   
                'Move': self.expand(
                    'Move', {'id': 'move_{}'.format(move_cnt)}),
                'Turn': self.expand(
                    'Turn', {'id': 'turn_{}'.format(turn_cnt)}),
            }
        elif choice == 'incorrectBodyOrder':
            move_cnt = self.getState('move_count')
            turn_cnt = self.getState('turn_count')

            template = '''
            {Turn}
            {Move}
            '''
            templateVars = {  
                'Move': self.expand(
                    'Move', {'id': 'move_{}'.format(move_cnt)}),
                'Turn': self.expand(
                    'Turn', {'id': 'turn_{}'.format(turn_cnt)}), 
            }
        elif choice == 'missingPrePost':
            move_cnt = self.getState('move_count')
            moveExpansion = self.expand(
                'Move', {'id': 'move_{}'.format(move_cnt)})
            turn_cnt = self.getState('turn_count')
            turn0Expansion = self.expand(
                'Turn', {'id': 'turn_{}'.format(turn_cnt)})
            turn_cnt = self.getState('turn_count')
            turn1Expansion = self.expand(
                'Turn', {'id': 'turn_{}'.format(turn_cnt)})

            template = '''
            {Turn0}
            {Move}
            {Turn1}
            '''
            templateVars = {  
                'Move': moveExpansion,
                'Turn0': turn0Expansion, 
                'Turn1': turn1Expansion, 
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
            turn_cnt = self.getState('turn_count')
            template = '''
            {Turn}
            '''
            templateVars = {   
                'Turn': self.expand(
                    'Turn', {'id': 'turn_{}'.format(turn_cnt)}),
            }
        elif choice == 'extraRandom':
            singleBodyExpansion = self.expand(
                'SingleBody', {'id': 'single_body_{}'.format(_id+1)})
            random_cnt = self.getState('random_strategy_count')
            randomStrategyExpansion = self.expand(
                'RandomStrategy',
                {'id': 'random_strategy_{}'.format(random_cnt)})

            template = '''
            {SingleBody}
            {RandomStrategy}
            '''
            templateVars = {   
                'SingleBody': singleBodyExpansion,
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
                    {'id': 'random_strategy_{}'.format(random_cnt)}),
            }
        else:
            raise Exception('how did you get here?')
        
        return gu.format(template, templateVars)

