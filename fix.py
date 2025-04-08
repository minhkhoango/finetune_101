import json

data = None
with open('law_rules.json', encoding='utf-8') as f:
    data = json.load(f)

trimmed_data = []
for item in data:
    rule_name = item['rule_name']
    output = ""

    if 'section' in item and item['section']:
        output += item['section'] + '\n'
    
    if 'subsections' in item and item['subsections']:
        count = 0

        for subsection in item['subsections']:

            output += subsection + '\n'

            if 'letter_subsections' not in item or count == len(item['letter_subsections']):
                continue
            
            while True:
                l_sub = item['letter_subsections']
                output += l_sub[count] + '\n'
                count = count + 1
                if count == len(l_sub) or l_sub[count][0] == 'a':
                    break
    
    trimmed_data.append({
        'input': rule_name,
        'output': output,
    })

print(trimmed_data[0])

with open('trimmed_data.json', 'w', encoding='utf-8') as f:
    json.dump(trimmed_data, f, ensure_ascii=False, indent=4)