import os
import pathlib
import re

root = pathlib.Path("src")
count = 0
files_modified = []

# Pattern pour trouver les print("...") ou print(f"...") avec un seul argument texte
# On ignore les print() sans arguments, ou les print(df) etc.
# Ce regex trouve `print("quelque chose")` ou `print(f"quelque chose")` ou `print('quelque chose')`
pattern = re.compile(r'^(\s*)print\s*\(\s*([fF]?["\'][^"\']*["\'])\s*\)\s*(#.*)?$')

for pyfile in root.rglob("*.py"):
    if pyfile.name == "logger.py":
        continue

    text = pyfile.read_text(encoding="utf-8")
    if "print" not in text:
        continue

    lines = text.split("\n")
    new_lines = []
    removed = 0
    needs_logger = False

    for line in lines:
        match = pattern.match(line)
        if match:
            indent = match.group(1)
            content = match.group(2)
            comment = match.group(3) or ""
            # On remplace par logger.info
            new_line = f"{indent}logger.info({content}){comment}"
            new_lines.append(new_line)
            removed += 1
            needs_logger = True
        else:
            new_lines.append(line)

    if removed > 0:
        new_text = "\n".join(new_lines)

        # Ajouter l'import si besoin
        if needs_logger and "from src.utils.logger import logger" not in new_text:
            # Chercher le bon endroit pour l'import (après les imports standards)
            # Simplification : on le met juste après les docstrings ou en haut
            import_line = "from src.utils.logger import logger"

            # Find the last import line to append after it, or just put it at line 0 if no imports
            # For a quick script, just adding it at the top of imports is fine.
            lines = new_text.split("\n")
            insert_idx = 0
            for i, l in enumerate(lines):
                if l.startswith("import ") or l.startswith("from "):
                    insert_idx = i
                    break

            lines.insert(insert_idx, import_line)
            new_text = "\n".join(lines)

        pyfile.write_text(new_text, encoding="utf-8")
        count += removed
        files_modified.append(f"{pyfile.relative_to(root)}: {removed} print(s)")

print(f"Total: {count} print() remplaces par logger.info() dans {len(files_modified)} fichiers")
for f in files_modified:
    print(f"  {f}")
