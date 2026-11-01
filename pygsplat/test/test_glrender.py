from pygsplat import SplatNumpyData, SplatRenderOpenGL

if __name__ == "__main__":
    import os

    width, height = 500, 500

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataName = "Hummel_figurine_Little_Violinist.ply"
    dataPath = os.path.join(script_dir, f"../../data/{dataName}")

    data = SplatNumpyData.from_ply(dataPath)
    renderer = SplatRenderOpenGL(width, height)
    renderer.set_splat_data(data)
    outputImagePath = os.path.join(script_dir, "output/test_glviewer.png")
    renderer.render_to_image(outputImagePath, flipY=True)