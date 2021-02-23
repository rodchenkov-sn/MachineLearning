using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Task1
{
    internal class SpamModel : DataModel
    {
        private readonly MLContext _context;
        private readonly IDataView _data;

        public SpamModel(MLContext context, string dataPath)
        {
            _context = context;
            _data = _context.Data.LoadFromTextFile<SpamData>(dataPath, ',', true);
        }

        protected override IEstimator<ITransformer> MakePipeline()
        {
            var features = _data.Schema
                .Select(col => col.Name)
                .Where(name => name != "Label")
                .ToArray()
                ;

            var dataProcess = _context
                .Transforms.Conversion.MapValueToKey("Label", "Label")
                .Append(_context.Transforms.Conversion.ConvertType("CapitalLong", outputKind: DataKind.Single))
                .Append(_context.Transforms.Conversion.ConvertType("CapitalTotal", outputKind: DataKind.Single))
                .Append(_context.Transforms.Concatenate("Features", features))
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

        private class SpamData
        {
            [LoadColumn(1, 55), VectorType(55)]
            public float[] Keywords;

            [LoadColumn(56)]
            public int CapitalLong;

            [LoadColumn(57)]
            public int CapitalTotal;

            [LoadColumn(58)]
            public string Label;
        }
    }
}
