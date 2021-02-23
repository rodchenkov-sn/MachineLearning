using Microsoft.ML;
using Microsoft.ML.Data;

namespace Task1
{
    internal class TicTacToeModel : DataModel
    {
        private readonly MLContext _context;
        private readonly IDataView _data;

        public TicTacToeModel(MLContext context, string dataPath)
        {
            _context = context;
            _data = _context.Data.LoadFromTextFile<TicTacToeData>(dataPath, separatorChar: ',');
        }

        protected override IEstimator<ITransformer> MakePipeline()
        {
            var dataProcess = _context
                .Transforms.Conversion.MapValueToKey("Label", "Label")
                .Append(_context.Transforms.Categorical.OneHotEncoding(new[]
                {
                    new InputOutputColumnPair("Marks", "Marks")
                }))
                .Append(_context.Transforms.Concatenate("Features", "Marks"))
                .Append(_context.Transforms.NormalizeMinMax("Features", "Features"))
                .AppendCacheCheckpoint(_context)
                ;

            var trainer = _context
                .MulticlassClassification.Trainers.NaiveBayes()
                .Append(_context.Transforms.Conversion.MapKeyToValue("PredictedLabel"))
                ;

            var pipeline = dataProcess.Append(trainer);
            return pipeline;
        }

        protected override MLContext GetContext()
        {
            return _context;
        }

        protected override IDataView GetDataView()
        {
            return _data;
        }

        private class TicTacToeData
        {
            [LoadColumn(0, 8), VectorType(9)]
            public string[] Marks { get; set; }

            [LoadColumn(9)]
            public string Label { get; set; }
        }

    }
}
