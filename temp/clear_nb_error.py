import json

path = r"E:\codes\projects\gsplatlab\pygsplat\test\glviewer.ipynb"
with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell.get("id") == "splat-render-gl":
        cell["execution_count"] = None
        cell["outputs"] = []

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")
print("cleared")