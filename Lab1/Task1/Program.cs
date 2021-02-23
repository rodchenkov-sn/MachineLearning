using Microsoft.ML;
using ScottPlot;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;

namespace Task1
{
    internal class Program
    {
        private static void DrawGraph(DataModel model, string graphPath)
        {
            var testFractions = Enumerable.Range(1, 99).Select(x => x / 100.0).ToArray();
            var testAccuracy = new List<double>();
            var trainAccuracy = new List<double>();

            foreach (var fraction in testFractions)
            {
                var result = model.RunModel(fraction);
                testAccuracy.Add(result.TestAccuracy);
                trainAccuracy.Add(result.TrainAccuracy);
            }

            var plot = new ScottPlot.Plot();
            plot.Add(new PlottableScatter(testFractions, testAccuracy.ToArray())
            {
                color = Color.Blue,
                label = "test",
                lineStyle = LineStyle.Solid,
                markerShape = MarkerShape.none
            });
            plot.Add(new PlottableScatter(testFractions, trainAccuracy.ToArray())
            {
                color = Color.Red,
                label = "train",
                lineStyle = LineStyle.Solid,
                markerShape = MarkerShape.none
            });
            plot.Legend();
            plot.XLabel("test fraction");
            plot.YLabel("accuracy");
            plot.SaveFig(graphPath);
        }

        private static void Main()
        {
            var context = new MLContext();

            var ticTacToeModel = new TicTacToeModel(context, "./data/TicTacToe.csv");
            DrawGraph(ticTacToeModel, "../../../graphs/TicTacToe.png");

            var spamModel = new SpamModel(context, "./data/Spam.csv");
            DrawGraph(spamModel, "../../../graphs/Spam.png");
        }
    }
}
