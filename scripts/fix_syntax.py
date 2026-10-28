with open('src/yield_predictor.py', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('print("\n  2. FIELD-CROPS', 'print("\\n  2. FIELD-CROPS')
text = text.replace('print("\n  3. TEMPORAL', 'print("\\n  3. TEMPORAL')

with open('src/yield_predictor.py', 'w', encoding='utf-8') as f:
    f.write(text)
print('Fixed syntax successfully')
