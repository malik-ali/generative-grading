import os
import re
import copy
import pickle
import javalang
import numpy as np
from glob import glob
from tqdm import tqdm

def anonimizeVariables(code):
    counter = 0
    while True:
        code, changed = anonimizeNextVar(code, counter)
        counter += 1
        if not changed: break
    return code

def anonimizeMethods(code):
    counter = 0
    while True:
        code, changed = anonimizeNextMethod(code, counter)
        counter += 1
        if not changed: break
    return code

# this doesn't work in the case where the student has
# a bug like println("i") <-- i isn't getting anonimized
def anonimizeNextVar(code, counter):
    scanner = TokenScanner(code)
    while scanner.hasMoreTokens():
        tk = scanner.nextToken()
        if tk == 'int' or tk == 'double':
            oldName = scanner.nextToken()
            if oldName.startswith('var'): continue
            if oldName == 'START': continue
            newName = 'var' + str(counter)
            code = tokenReplace(code, oldName, newName)
            return code, True

    return code, False

def anonimizeNextMethod(code, counter):
    scanner = TokenScanner(code)
    while scanner.hasMoreTokens():
        tk = scanner.nextToken()
        nextTk = scanner.peek()
        if tk == 'private' and nextTk == 'void':
            _ = scanner.nextToken()
            oldName = scanner.nextToken()
            if oldName.startswith('method'): continue
            newName = 'method' + str(counter)
            code = tokenReplace(code, oldName, newName)
            return code, True
    return code, False

def tokenReplace(code, oldName, newName):
    ret = ''
    scanner = TokenScanner(code)
    while scanner.hasMoreTokens():
        tk = scanner.nextToken()
        if tk == oldName:
            ret += newName
        else:
            ret += tk
        ret += ' '
    return ret
    
def anonimizeNextIdentifier(code, counter):
    scanner = TokenScanner(code)
    while scanner.hasMoreTokens():
        tk = scanner.nextTokenRaw()
        tkType = TokenScanner.tokenType(tk)
        if tkType == 'Identifier':
            oldName = tk.value
            if oldName.startswith('id_'): continue
            newName = 'id_' + str(counter)
            code = code.replace(oldName, newName)
            return code, True
    return code, False

def normalizeWhitespace(string):
    scanner = TokenScanner(string)
    ret = ''
    forLoopSemis = 0
    while scanner.hasMoreTokens():
        tk = scanner.nextToken()
        nextTk = scanner.peek()

        if tk == 'for': forLoopSemis = 2

        if tk == ';' and forLoopSemis > 0:
            forLoopSemis -= 1
            postWhite = ' '
        else:
            postWhite = getNextWhitespace(tk, nextTk)

        ret += tk + postWhite
    ret = fixWhitespace(ret)
    return ret

def getNextWhitespace(tk, nextTk):
    if tk == '(': return ''
    if tk == '{': return '\n'
    if tk == '}': return '\n'
    if tk == ';': return '\n'
    if nextTk == ';': return ''
    if nextTk == '(': return ''
    if nextTk == ')': return ''
    if nextTk == '++': return ''
    if nextTk == '--': return ''
    return ' '

def removeComments(string):
    string = re.sub(re.compile("/\*.*?\*/",re.DOTALL ) ,"" ,string) # remove all occurance streamed comments (/*COMMENT */) from string
    string = re.sub(re.compile("//.*?\n" ) ,"\n" ,string) # remove all occurance singleline comments (//COMMENT\n ) from string
    return string

def removeNoops(string):
    string = re.sub(re.compile("pause(.*?);",re.DOTALL ) ,"" ,string)
    string = re.sub(re.compile("import.*?;",re.DOTALL ) ,"" ,string)
    string = re.sub(re.compile("setFont(.*?);",re.DOTALL ) ,"" ,string)
    string = re.sub(re.compile("private static final long serialVersionUID.*?;", re.DOTALL), "", string)
    return string

def normalizeBinaryOppWhitespace(string):
    return string

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

def getIndent(n):
    space = ''
    for i in range(n):
        space += '  '
    return space

class TokenScanner():

    @staticmethod
    def tokenType(tk):
        return tk.__class__.__name__

    def __init__(self, code):
        self.tokens = list(javalang.tokenizer.tokenize(code))
        self.tokens.reverse()

    def nextToken(self):
        return self.tokens.pop().value

    def nextTokenRaw(self):
        return self.tokens.pop()

    def saveToken(self, token):
        self.tokens.append(token)

    def peek(self):
        if not self.hasMoreTokens():
            return None
        return self.tokens[-1].value

    def hasMoreTokens(self):
        return len(self.tokens) > 0

    def verifyToken(self, expected):
        n = self.nextToken()
        assert(n.value == expected)


if __name__ == '__main__':
    import argparse
    arg_parser = argparse.ArgumentParser()
    # we pass the directory bc we want to auto-find all the count files
    arg_parser.add_argument(
        'data_dir',
        default='None',
        help='The path to directory storing counts pickle.')
    arg_parser.add_argument(
        'problem',
        help='liftoff|drawCircles|citizenship13|posTagging')
    args = arg_parser.parse_args()

    count_paths = glob(os.path.join(args.data_dir, 'counts*.pkl'))
    for counts_path in count_paths:
        print('Anonymizing {}'.format(os.path.basename(counts_path)))
    
        with open(counts_path, 'rb') as fp:
            counts = pickle.load(fp)

        # we need to store it this way bc its not a 1-to-1 mapping
        raw_to_anonymized = {}
        pbar = tqdm(total=len(counts))
        for code in counts:
            raw_code = copy.deepcopy(code)
            if args.problem in ['liftoff', 'drawCircles']:
                code = removeComments(code)
                code = removeNoops(code)
                code = normalizeBinaryOppWhitespace(code)
                code = anonimizeMethods(code)
                code = anonimizeVariables(code)
                code = normalizeWhitespace(code)	
                raw_to_anonymized[raw_code] = code
            else:  # code.org
                raw_to_anonymized[raw_code] = raw_code
            pbar.update()
        pbar.close()

        anon_mapping_path, ext = os.path.splitext(counts_path)
        anon_mapping_path = anon_mapping_path.replace('counts', 'anon_mapping') + ext

        with open(anon_mapping_path, 'wb') as fp:
            pickle.dump(raw_to_anonymized, fp)
