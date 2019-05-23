import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
from base import Decision, ReusableDecision


class RepeatedBody(ReusableDecision):
    def registerChoices(self):
        self.addChoice(self.getKey(), {
            'repeat' : 20,
            'none' : 80,
        })
    
    def renderCode(self):
        _id = int(self.getKey().split('_')[-1])
        choice = self.getChoice(self.getKey())
        self.incrementState('repeated_body_count')

        if choice == 'repeat':
            single_body_count = self.getState('single_body_count')

            template = '''
            {SingleBody}
            {RepeatedBody}  \n
            '''
            templateVars = {
                'SingleBody': self.expand(
                    'SingleBody',
                    {'id': 'single_body_{}'.format(single_body_count)}), 
                'RepeatedBody': self.expand(
                    'RepeatedBody',
                    {'id': 'repeated_body_{}'.format(_id+1)}), 
            }
        elif choice == 'none':
            template = ""
            templateVars = {}
        else:
            raise Exception('how did you get here?')
        
        return gu.format(template, templateVars)

