# Photometric Stereo

## Environment

* MATLAB 2022b



## Analysis and Conclusion

### Results

|        | Bear                                 | Buddha                                 | Cat                                 | Pot                                 |
| ------ | ------------------------------------ | -------------------------------------- | ----------------------------------- | ----------------------------------- |
| Albedo | ![](results/fine/bearPNG_Albedo.png) | ![](results\fine\buddhaPNG_Albedo.png) | ![](results/fine/catPNG_Albedo.png) | ![](results/fine/potPNG_Albedo.png) |
| Normal | ![](results/fine/bearPNG_Normal.png) | ![](results/fine/buddhaPNG_Normal.png) | ![](results/fine/catPNG_Normal.png) | ![](results/fine/potPNG_Normal.png) |
| Render | ![](results/fine/bearPNG_Render.png) | ![](results/fine/buddhaPNG_Render.png) | ![](results/fine/catPNG_Render.png) | ![](results/fine/potPNG_Render.png) |

### Comparison with non-highlight & shadow handling

|                                           | Bear                                 | Buddha                                 | Cat                                 | Pot                                 |
| ----------------------------------------- | ------------------------------------ | -------------------------------------- | ----------------------------------- | ----------------------------------- |
| **Albedo w/ highlight & shadow handling** | ![](results/fine/bearPNG_Albedo.png) | ![](results\fine\buddhaPNG_Albedo.png) | ![](results/fine/catPNG_Albedo.png) | ![](results/fine/potPNG_Albedo.png) |
| Albedo w/o highlight & shadow handling    | ![](results/wo/bearPNG_Albedo.png)   | ![](results\wo\buddhaPNG_Albedo.png)   | ![](results/wo/catPNG_Albedo.png)   | ![](results/wo/potPNG_Albedo.png)   |
| **Render w/ highlight & shadow handling** | ![](results/fine/bearPNG_Render.png) | ![](results/fine/buddhaPNG_Render.png) | ![](results/fine/catPNG_Render.png) | ![](results/fine/potPNG_Render.png) |
| Render w/o highlight & shadow handling    | ![](results/wo/bearPNG_Render.png)   | ![](results/wo/buddhaPNG_Render.png)   | ![](results/wo/catPNG_Render.png)   | ![](results/wo/potPNG_Render.png)   |

Obviously, after processing, the highlight area becomes less shiny and more consistent with Lambert's model.

