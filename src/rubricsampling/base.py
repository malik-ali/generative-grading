import numpy as np
import random



class Decision:
    ROOT_RV_NAME = 'ROOT_RV'
    ROOT_RV_VAL  = 'default'

    def __init__(self, engine):
        self._engine = engine
        self.params = {}
        self._rvs = {}

    ##############################################
    #              Public Methods                #  
    ##############################################

    def getClassName(self):
        return type(self).__name__

    def expand(self, nonterminal, params = {}):
        ret = self._engine.render(nonterminal, params)
        ret = ret.replace('{', '{{')
        ret = ret.replace('}', '}}')
        return ret


    '''
    Choices 
    Choices are random variables. You can declare them,
    And the engine will chose their values for you. This is
    the only part of your program that can have randomness
    '''
    def addChoice(self, choice_name, values):
        """
         - choice_name: name of a RV
         - values: possible values of RV. Either a list, in which case it should be picked uniformly
            otherwise it is a dictionary with weights
        """
        self._rvs[choice_name] = values
        val = self._engine._pick_rv(choice_name, values)
        self._engine.addGlobalChoice(choice_name, val)


    def getChoice(self, key):
        return self._engine.choices[key]

    def hasChoice(self, key):
        return key in self._engine.choices



    '''
    State 
    State is global data that can be used to help future decisions
    know about previous decisions
    '''
    def addState(self, key, value):
        self._engine.state[key] = value

    def getState(self, key):
        return self._engine.state[key]

    def hasState(self, key):
        return key in self._engine.state

    def setState(self, key, value):
        assert key in self._engine.state
        self._engine.state[key] = value

    def addOrSetState(self, key, value):
        if self.hasState(key):
            self.setState(key, value)
        else:
            self.addState(key, value)

    def incrementState(self, key, step=1):
        value = self.getState(key)
        self.setState(key, value + step)


    '''
    Rubric 
    Rubrics are the labels you would need to calculate a grade
    (though they are not grades themselves). They should be a 
    function of the random variables
    '''
    # can only make a boolean true, never turns it false
    # we grade down. This should mean points off
    def turnOnRubric(self, key):
        self._engine.rubric[key] = True



    ##############################################
    #               To Overload                  #  
    ##############################################

    # Overload
    def updateRubric(self):
        pass

    # Overload
    def registerChoices(self):
        pass

    def preregisterDecisionIds(self):
        '''
         Must return a dictionary from ReusableDecision name to
         the set of valid ids for that decision.
        '''
        pass 

    # Overload
    def renderCode(self):
        """
        Renders the code after choices have been made, returning
        a template string with both code and format specifiers
        directing the enginine on what to render in that place.

        Usage notes:
            - Put nonterminals in format specifier for the engine to render.

            - Make sure to escape code curly braces with {{ and }}

            - To generate the same instance of a nonterminal in two places,
              use the same name inmultiple places.

            - To generate separate instance of a nonterminal in two places,
              append the nonterminal name with numbers _1, _2 etc.

        """
        className = type(self).__name__
        raise NotImplementedError('Method renderCode needs to be implemented by ' + className)

    ##############################################
    #            Private Methods                 #  
    ##############################################

    # the engine can initialize current params
    def _setParams(self, params):
        self.params = params

    def _getRandomVariables(self):
        return self._rvs



class ReusableDecision(Decision):

    def __init__(self, engine):
        super(ReusableDecision, self).__init__(engine)

    def getCount(self):
        className = type(self).__name__
        return self.getState(className + '_count')
    def getDecisionName(self):
        className = type(self).__name__
        return className + '_' + str(self.getCount())
    def incrementCount(self):
        className = type(self).__name__
        self.setState(className + '_count', self.getCount() + 1)
       
    def addChoice(self, choice_name, values, is_prefix=False):
        """
         Primarily just uses super classes implementation. Key difference is that
         it ensures choice_name is in the declared set of validIds for this class.
        """
        

        if choice_name not in self.validIds:
            # check if a prefix of this choice_name is a validId.
            # This is super not robust to catching errors but we need it for multiple
            # RV choices in a reusable class. See Diameter
            is_prefix_valid = is_prefix and any(choice_name.startswith(valId) for valId in self.validIds)
            if not is_prefix_valid:
                raise ValueError(f'Choice name {choice_name} not in validIds of reusable decision. Valid ids=[{self.validIds}]')

        super(ReusableDecision, self).addChoice(choice_name, values)
        

    def registerValidIds(self, validIds):
        '''
        Sets the validIds for this reusable decisions. The engine should
        first get all preregestered ids from all classes and then set the
        validIds relevant to this class. 
        
        This will be a set of ids (strings) that can be used as a param
        to append to the name of this reusable component.
        '''
        self.validIds = validIds
        if not isinstance(self.validIds, (set, list)) or len(self.validIds) < 1:
            raise ValueError('Must set self.validIds to be a set/list containing at least one valid id.')        
  
    def _getValidIds(self):
        return self.validIds

    def getKey(self):
        return self.params['id']

