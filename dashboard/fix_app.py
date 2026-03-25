import re, os

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
print("Fixing:", path)

with open(path, "r", encoding="utf-8", errors="replace") as f:
    lines = f.readlines()

fixed = 0
new_lines = []
for i, line in enumerate(lines):
    if "param_input" in line and "AOV" in line:
        new_line = re.sub(r"12_?000", "50_000", line)
        new_line = re.sub(r'"AOV \([^"]*\)"', '"AOV (Rs.)"', new_line)
        if new_line != line:
            print("Line", i+1, "BEFORE:", line.rstrip())
            print("Line", i+1, "AFTER: ", new_line.rstrip())
            fixed += 1
        new_lines.append(new_line)
    else:
        new_lines.append(line)

with open(path, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print("Done.", fixed, "line(s) fixed.")
print("Now run:  streamlit run app.py")
