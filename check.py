from text.symbols import symbols
from text import text_to_sequence

s = "{o l a sp o s}"          # usamos 'sp' SIN @ en el texto
ids = text_to_sequence(s, ["english_cleaners"])
print([symbols[i] for i in ids])
