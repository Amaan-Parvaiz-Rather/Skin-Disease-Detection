import re

with open(r'd:\Skindiseases\ProjectFlask_internship_Assignment\static\css\style.css', 'r', encoding='utf-8') as f:
    text = f.read()

# Remove comments
text_clean = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
# Remove strings
text_clean = re.sub(r'".*?"', '', text_clean)
text_clean = re.sub(r"'.*?'", '', text_clean)

# Check curly braces
lines = text_clean.split('\n')
stack = []
for i, line in enumerate(lines):
    for j, char in enumerate(line):
        if char == '{':
            stack.append((i+1, j+1))
        elif char == '}':
            if stack:
                stack.pop()
            else:
                print(f'Extra }} at line {i+1}:{j+1}')

if stack:
    print('Unclosed { at:', stack)
else:
    print('Curly braces matched in clean text')

