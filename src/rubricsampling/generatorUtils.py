from numpy.random import choice
import random

def bernoulli(p):
    return random.random() < p

def mapChoice(weightMap) :
    elems = []
    weights = []
    for key in weightMap:
        elems.append(key)
        weights.append(weightMap[key])
    index = listChoice(weights)
    return elems[index]

def listChoice(prob) :
    candidates = range(len(prob))

    # normalize the weights to sum to 1
    dist = []
    total = sum(prob)
    for weight in prob:
        p = float(weight) / total
        dist.append(p)

    # use the build in numpy package
    return choice(candidates, p=dist)

def fixWhitespace(program):
    lines = program.split('\n')
    result = ''

    indent = 0 
    for i in range(len(lines)):
        line = lines[i]
        if line == '' or line.isspace(): continue

        stripped = line.strip()

        # update the indent
        if stripped[0] == '}':
            indent -= 1

        # make the new line
        result += getIndent(indent)
        result += stripped

        # don't add whitespace to the last line
        if i != len(lines) - 1: result += '\n'

        # update the indent
        if stripped[-1] == '{':
            indent += 1

    return result

def format(template, templateVars):
    startTemp = '<START_BRACKET>'
    endTemp = '<END_BRACKET>'
    template = template.replace('{{', startTemp)
    template = template.replace('}}', endTemp)

    result = template.format(**templateVars)

    result = result.replace(startTemp, '{{')
    result = result.replace(endTemp, '}}')
    return result

def getIndent(n):
    space = ''
    for i in range(n):
        space += '  '
    return space

def calcGoodTuringP(countMap):
    if len(countMap) == 0: return 1.0
    n1 = 0
    for key in countMap:
        if countMap[key] == 1:
            n1 += 1
    return float(n1) / len(countMap)
