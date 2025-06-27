with open('advanced_backtesting.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find _train_model method and add debugging
for i, line in enumerate(lines):
    if 'def _train_model(' in line:
        # Add debug after the docstring
        j = i + 1
        while j < len(lines) and '"""' not in lines[j]:
            j += 1
        if j < len(lines) - 1:
            j += 1  # Move past the closing """
            indent = '        '  # 8 spaces
            lines.insert(j, f'{indent}print(f"DEBUG _train_model: train_start={{train_start}}, train_end={{train_end}}")\n')
            lines.insert(j+1, f'{indent}print(f"DEBUG _train_model: all_data has {{len(all_data)}} symbols")\n')
    
    # Add debug in the data preparation loop
    if 'for symbol, df in all_data.items():' in line and i > 200:  # Make sure we're in _train_model
        indent = '            '  # 12 spaces
        lines.insert(i+1, f'{indent}print(f"DEBUG: Processing {{symbol}} with {{len(df)}} rows")\n')
    
    # Add debug after train_data[symbol] = train_df
    if 'train_data[symbol] = train_df' in line:
        indent = '                '  # 16 spaces
        lines.insert(i, f'{indent}print(f"DEBUG: Adding {{symbol}} to train_data with {{len(train_df)}} rows")\n')

with open('advanced_backtesting.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print('Added debugging to _train_model')
