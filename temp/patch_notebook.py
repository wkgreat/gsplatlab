import json

path = r"E:\codes\projects\gsplatlab\pygsplat\test\glviewer.ipynb"
with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "splat-render-compare",
    "metadata": {},
    "outputs": [],
    "source": [
        "from PIL import Image as PILImage\n",
        "\n",
        "gl = np.asarray(PILImage.open(\"output/glviewer.png\").convert(\"RGB\"), dtype=np.float32)\n",
        "cu = np.asarray(PILImage.open(\"output/cuviewer.png\").convert(\"RGB\"), dtype=np.float32)\n",
        "diff = np.abs(gl - cu)\n",
        "print(\"mean abs diff:\", float(diff.mean()), \"max:\", float(diff.max()))\n",
        "assert diff.mean() < 2.0",
    ],
}

index = next(
    i for i, c in enumerate(nb["cells"]) if c.get("id") == "splat-render-viewer-title"
)
nb["cells"].insert(index, cell)

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")
print("cells:", [c.get("id") for c in nb["cells"]])