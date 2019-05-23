# problem 1

CODEORG1_IX_TO_LABEL = {
    0: u'Standard Strategy',
    1: u'Does not get nesting',
    2: u'Does not get pre/post condition',
    3: u"Doesn't use a repeat",
    4: u'Repetition of bodies',
    5: u"Doesn't loop three times",
    6: u'Left/Right confusion',
    7: u'Does not know equilateral is 60',
    8: u'Does not invert angle',
    9: u'Default turn',
    10: u'Random move amount',
    11: u'Default move',
    12: u'Body order is incorrect (turn/move)',
}
CODEORG1_LABEL_TO_IX = {v: k for k, v in CODEORG1_IX_TO_LABEL.items()}
CODEORG1_N_LABELS = len(CODEORG1_IX_TO_LABEL.keys())
CODEORG1_LOOP_LABELS_IX = [1, 2, 3, 4, 5, 12]       # which labels account for looping
CODEORG1_GEOMETRY_LABELS_IX = [6, 7, 8, 9, 10, 11]  # which labels account for geometry

# problem 9

CODEORG9_IX_TO_LABEL = {
    0 : 'Move: wrong multiple',
    1 : 'Move: constant',
    2 : 'Move: wrong opp',
    3 : 'Move: missing opp',
    4 : 'Move: correct',
    5 : 'Move: forward/backward confusion',
    6 : 'Move: no move',
    7 : 'Turn: constant',
    8 : 'Turn: wrong multiple',
    9 : 'Turn: wrong opp',
    10 : 'Turn: missing opp',
    11 : 'Turn: no turn',
    12 : 'Turn: left/right confusion',
    13 : 'Single shape: wrong iter #',
    14 : 'Single shape: body incorrect',
    15 : 'Single shape: wrong MT order',
    16 : 'Single shape: missing repeat',
    17 : 'Single shape: nesting issue',
    18 : 'Single shape: armslength',
    19 : 'For loop: wrong start',
    20 : 'For loop: wrong end',
    21 : 'For loop: wrong delta',
    22 : 'For loop: not looping by sides',
    23 : 'For loop: no loop',
    24 : 'For loop: armslength',
    25 : 'For loop: repeat instead of for',
}
CODEORG9_LABEL_TO_IX = {v: k for k, v in CODEORG9_IX_TO_LABEL.items()}
CODEORG9_N_LABELS = len(CODEORG9_IX_TO_LABEL.keys())
CODEORG9_LOOP_LABELS_IX = [13, 14, 15, 16, 17, 18,19, 20, 21, 22, 23, 24, 25]
CODEORG9_GEOMETRY_LABELS_IX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
