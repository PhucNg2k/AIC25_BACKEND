
answer_raw = "He said "Hello""

answer = answer_raw.strip()
# Collapse any runs of quotes to a single quote before normalization (avoid over-escaping)
answer = re.sub(r'"{2,}', '"', answer)
# Normalize accidental surrounding or unmatched quotes from user input
if answer.startswith('"') and answer.endswith('"') and len(answer) >= 2:
    answer = answer[1:-1]
elif answer.endswith('"') and not answer.startswith('"'):
    answer = answer[:-1]
elif answer.startswith('"') and not answer.endswith('"'):
    answer = answer[1:]

# Keep content as-is (no per-token re-wrapping). We'll quote once for the whole answer field.
# Trim again after removing surrounding quotes to drop trailing/leading spaces
answer = answer.strip()
# Escape internal quotes per CSV rules
answer_escaped = answer.replace('"', '""')

print(answer_escaped)