import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
from base import Decision, ReusableDecision


class Strategy(Decision):

    def registerChoices(self):
        self.addChoice('strategyChoices', {
            'standard' : 100,
            'clockwise' : 2,
            'extraRandom' : 3,
            'random': 4,
        })

    def updateRubric(self):
        choice = self.getChoice('strategyChoices')
        rubric = {
            'standard' : ['Standard Strategy'],
            'clockwise' : [],
            'extraRandom' : ['Standard Strategy'],
            'random' : [],
        }
        labelList = rubric[choice]
        for label in labelList:
            self.turnOnRubric(label)

    def renderCode(self):
        choice = self.getChoice('strategyChoices')

        if choice == 'standard':
            template = '''
            {StandardStrategy}
            '''
            templateVars = {
                'StandardStrategy': self.expand('StandardStrategy'),
            }
        elif choice == 'clockwise':
            template = '''
            {ClockwiseStrategy}
            '''
            templateVars = {
                'ClockwiseStrategy': self.expand('ClockwiseStrategy'),
            }
        elif choice == 'extraRandom':
            standardStrategyExpansion = self.expand('StandardStrategy')
            count = self.getState('random_strategy_count')
            randomStrategyExpansion = self.expand(
                'RandomStrategy',
                {'id': 'random_strategy_{}'.format(count)},
            )
            template = '''
            {StandardStrategy}
            {RandomStrategy}
            '''
            templateVars = {
                'StandardStrategy': standardStrategyExpansion,
                'RandomStrategy': randomStrategyExpansion,
            }
        elif choice == 'random':
            count = self.getState('random_strategy_count')
            template = '''
            {RandomStrategy}
            '''
            templateVars = {
                'RandomStrategy': self.expand(
                    # we know for sure this is the first random
                    'RandomStrategy',
                    {'id': 'random_strategy_{}'.format(count)},
                ),
            }
        else:
            raise Exception('how did you get here?')
    
        return gu.format(template, templateVars)


class StandardStrategy(Decision): 

    def registerChoices(self):
        self.addChoice('standardStrategyChoices', {
            'noRepeat': 20,
            'looped': 70,
            'repeatBody': 10,
        })

    def updateRubric(self):
        choice = self.getChoice('standardStrategyChoices')
        rubric = {
            'noRepeat': ["Doesn't use a repeat"],
            'looped': [],
            'repeatBody': ["Doesn't use a repeat", 'Repetition of bodies'],
        }
        labelList = rubric[choice]
        for label in labelList:
            self.turnOnRubric(label)

    def renderCode(self):
        choice = self.getChoice('standardStrategyChoices')
        if choice == 'noRepeat':
            single_body_count = self.getState('single_body_count')
            
            template = '''
            {SingleBody}
            '''
            templateVars = {
                'SingleBody': self.expand(
                    'SingleBody', 
                    {'id': 'single_body_{}'.format(single_body_count)},
                ),
            }
        elif choice == 'looped':
            single_body_count = self.getState('single_body_count')
            repeat_num_count = self.getState('repeat_num_count')

            template = '''
            Repeat({RepeatNum}) {{ \n
                {SingleBody}
            }} \n
            '''
            templateVars = {
                'RepeatNum': self.expand(
                    'RepeatNum', 
                    {'id': 'repeat_num_{}'.format(repeat_num_count)},
                ),
                'SingleBody': self.expand(
                    'SingleBody',
                    {'id': 'single_body_{}'.format(single_body_count)},
                ),
            }
        elif choice == 'repeatBody':
            single_body_count = self.getState('single_body_count')
            singleBody0Expansion = self.expand(
                'SingleBody',
                {'id': 'single_body_{}'.format(single_body_count)},
            )
            single_body_count = self.getState('single_body_count')
            singleBody1Expansion = self.expand(
                'SingleBody',
                {'id': 'single_body_{}'.format(single_body_count)},
            )
            repeated_body_count = self.getState('repeated_body_count')
            repeatedBodyExpansion = self.expand(
                'RepeatedBody',
                {'id': 'repeated_body_{}'.format(repeated_body_count)},
            )

            template = '''
            {SingleBody0}
            {SingleBody1}
            {RepeatedBody}  \n
            '''
            templateVars = {
                'SingleBody0': singleBody0Expansion,
                'SingleBody1': singleBody1Expansion,
                'RepeatedBody': repeatedBodyExpansion,
            }
        else:
            raise Exception('how did you get here?')

        return gu.format(template, templateVars)


class ClockwiseStrategy(Decision):

    def registerChoices(self):
        self.addChoice('clockwiseStrategyChoices', {
            'noRepeat': 20,
            'looped': 70,
            'repeatBody': 10,
        })

    def updateRubric(self):
        choice = self.getChoice('clockwiseStrategyChoices')
        rubric = {
            'noRepeat': [],
            'looped': [],
            'repeatBody': [],
        }
        labelList = rubric[choice]
        for label in labelList:
            self.turnOnRubric(label)

    def renderCode(self):
        choice = self.getChoice('clockwiseStrategyChoices')
        if choice == 'noRepeat':
            count = self.getState('single_clockwise_body_count')

            template = '''
            {ClockwiseTurnStart}
            {SingleClockwiseBody}
            '''
            templateVars = {
                'ClockwiseTurnStart': self.expand('ClockwiseTurnStart'),
                'SingleClockwiseBody': self.expand(
                    'SingleClockwiseBody',
                    {'id': 'single_clockwise_body_{}'.format(count)},
                ),
            }
        elif choice == 'looped':
            repeat_num_count = self.getState('repeat_num_count')
            single_clockwise_body_count = self.getState('single_clockwise_body_count')

            template = '''
            {ClockwiseTurnStart} \n
            Repeat({RepeatNum}) {{ \n
            {SingleClockwiseBody}
            }} \n
            '''
            templateVars = {
                'ClockwiseTurnStart': self.expand('ClockwiseTurnStart'),
                'RepeatNum': self.expand(
                    'RepeatNum', 
                    {'id': 'repeat_num_{}'.format(repeat_num_count)},
                ),
                'SingleClockwiseBody': self.expand(
                    'SingleClockwiseBody',
                    {'id': 'single_clockwise_body_{}'.format(single_clockwise_body_count)},
                ),
            }
        elif choice == 'repeatBody':
            single_clockwise_body_count = self.getState('single_clockwise_body_count')
            singleClockwiseBodyExpansion = self.expand(
                'SingleClockwiseBody',
                {'id': 'single_clockwise_body_{}'.format(single_clockwise_body_count)},
            )
            random_strategy_count = self.getState('random_strategy_count')
            randomStrategyExpansion = self.expand(
                'RandomStrategy',
                {'id': 'random_strategy_{}'.format(random_strategy_count)},
            )

            template = '''
            {ClockwiseTurnStart}
            {SingleClockwiseBody}
            {RandomStrategy}
            '''
            templateVars = {
                'ClockwiseTurnStart': self.expand('ClockwiseTurnStart'),
                'SingleClockwiseBody': singleClockwiseBodyExpansion,
                'RandomStrategy': randomStrategyExpansion,
            }
        else:
            raise Exception('how did you get here?')

        return gu.format(template, templateVars)


class RandomStrategy(ReusableDecision):
    
    def registerChoices(self):
        choice_name = self.getKey()
        self.addChoice(choice_name, {
            'move': 25,
            'turnLeft': 20,
            'turnRight': 20,
            'repeat': 10,
            'none': 50,
        })

    def renderCode(self):
        _id = int(self.getKey().split('_')[-1])
        choice = self.getChoice(self.getKey())
        # record that we have used one more random
        self.incrementState('random_strategy_count')

        if choice == 'move':
            move_amount_count = self.getState('move_amount_count')

            template = '''
            Move({MoveAmount}) \n
            {RandomStrategy}
            '''
       
            # NOTE: we assume MoveAmount cannot lead back to RandomStrategy
            moveAmountExpansion = self.expand(
                'MoveAmount',
                {'id': 'move_amount_{}'.format(move_amount_count)},
            )
            # for example, MoveAmount could lead back to RandomStrategy
            # in which case, _id+1 is not the right number anymore
            randomStrategyExpansion = self.expand(
                'RandomStrategy',
                {'id': 'random_strategy_{}'.format(_id+1)})
        
            templateVars = {
                'MoveAmount': moveAmountExpansion,
                'RandomStrategy': randomStrategyExpansion, 
            }
        elif choice == 'turnLeft':
            turn_amount_count = self.getState('turn_amount_count')

            template = '''
            TurnLeft({TurnAmount}) \n
            {RandomStrategy}
            '''

            # Assume this cannot cycle back
            turnAmountExpansion = self.expand(
                'TurnAmount',
                {'id': 'turn_amount_{}'.format(turn_amount_count)},
            )
            randomStrategyExpansion = self.expand(
                'RandomStrategy',
                {'id': 'random_strategy_{}'.format(_id+1)})

            templateVars = {
                'TurnAmount': turnAmountExpansion,
                'RandomStrategy': randomStrategyExpansion,
            }
        elif choice == 'turnRight':
            turn_amount_count = self.getState('turn_amount_count')

            template = '''
            TurnRight({TurnAmount}) \n
            {RandomStrategy}
            '''

            # Assume this cannot cycle back
            turnAmountExpansion = self.expand(
                'TurnAmount',
                {'id': 'turn_amount_{}'.format(turn_amount_count)},
            )
            randomStrategyExpansion = self.expand(
                'RandomStrategy',
                {'id': 'random_strategy_{}'.format(_id+1)})

            templateVars = {
                'TurnAmount': turnAmountExpansion,
                'RandomStrategy': randomStrategyExpansion
            }
        elif choice == 'repeat':
            # NOTE: here we cannot assume that _id+1 and _id+2 will work
            # as in, the first RandomStrategy could expand into multiple
            # random strategies..., this is where we must use State.
            randomStrategy0Expansion = self.expand(
                'RandomStrategy',
                {'id': 'random_strategy_{}'.format(_id+1)})
            # this is where _id+2 would be wrong
            random_strategy_count = self.getState('random_strategy_count')  # already incremented 
            randomStrategy1Expansion = self.expand(
                'RandomStrategy',
                {'id': 'random_strategy_{}'.format(random_strategy_count)})
            
            # incrementing is handled in the RepeatNum class
            repeat_num_count = self.getState('repeat_num_count')

            repeatNumExpansion = self.expand(
                'RepeatNum',
                {'id': 'repeat_num_{}'.format(repeat_num_count)})

            template = '''
            Repeat({RepeatNum}) {{ \n
            {RandomStrategy0}
            }} \n
            {RandomStrategy1}
            '''
            templateVars = {
                'RepeatNum': repeatNumExpansion,
                'RandomStrategy0': randomStrategy0Expansion,
                'RandomStrategy1': randomStrategy1Expansion,
            }
        elif choice == 'none':
            template = ""
            templateVars = {}
        else:
            raise Exception('how did you get here?')

        return gu.format(template, templateVars)
