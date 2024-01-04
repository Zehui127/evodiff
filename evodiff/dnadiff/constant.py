DNA = 'ATGC'

DNA_GAP = 'N'
MASK = '#'  # Useful for masked language model training
MSA_PAD = '!'
STOP = '*'
START = '@'

SPECIALS = DNA_GAP + STOP  + MASK + START

DNA_ALPHABET = DNA + SPECIALS + MSA_PAD
DNA_MSA_AAS = DNA + DNA_GAP
