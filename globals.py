import os

BASE_PATH = os.path.dirname(os.path.abspath(__file__))


PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
START_TOKEN = '<sos>'
END_TOKEN = '<eos>'

HEAD = 0
BODY = 1
TAIL = 2

# bad idea to do... but hack now
# see line 184 of process_data.py
PAD_IDX = 0
UNK_IDX = 1
START_IDX = 2
END_IDX = 3
