from pathlib import Path

def loop_dirs(path:Path) -> list:
    full_path = []
    for c in path.iterdir():
        if c.is_dir():
            full_path.extend(loop_dirs(c))
        else:
            full_path.append(c)
    return full_path

paths = loop_dirs(Path("../2025"))

for path in paths:
    print(path)
