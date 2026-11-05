using Numpy;

namespace Zad3;

public static class Sobel
{
    public static NDarray ApplySobel(NDarray chunk)
    {
        var padWidth = chunk.ndim == 3
            ? new[,] { { 1, 1 }, { 1, 1 }, { 0, 0 } }
            : new[,] { { 1, 1 }, { 1, 1 } };
        var p = chunk.astype(np.float64).pad(np.array(padWidth), "reflect");

        var sobelX = p["0:-2,2:"] + p["1:-1,2:"] * 2 + p["2:,2:"]
                     - p["0:-2,0:-2"] - p["1:-1,0:-2"] * 2 - p["2:,0:-2"];
        var sobelY = p["2:,0:-2"] + p["2:,1:-1"] * 2 + p["2:,2:"]
                     - p["0:-2,0:-2"] - p["0:-2,1:-1"] * 2 - p["0:-2,2:"];

        var magnitude = (sobelX * sobelX + sobelY * sobelY).sqrt();
        return magnitude.clip((NDarray)0.0, (NDarray)255.0).astype(np.uint8);
    }
}