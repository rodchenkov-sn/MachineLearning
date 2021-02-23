using Microsoft.ML;

namespace Task1
{
    internal abstract class DataModel
    {
        public RunResult RunModel(double testFraction)
        {
            var split = GetContext().Data.TrainTestSplit(GetDataView(), testFraction);

            var pipeline = MakePipeline();

            var model = pipeline.Fit(split.TrainSet);

            var testPredictions = model.Transform(split.TestSet);
            var testMetrics = GetContext().MulticlassClassification.Evaluate(testPredictions);

            var trainPredictions = model.Transform(split.TrainSet);
            var trainMetrics = GetContext().MulticlassClassification.Evaluate(trainPredictions);

            return new RunResult
            {
                TestAccuracy = testMetrics.MicroAccuracy,
                TrainAccuracy = trainMetrics.MicroAccuracy
            };
        }

        protected abstract MLContext GetContext();
        protected abstract IDataView GetDataView();
        protected abstract IEstimator<ITransformer> MakePipeline();
    }
}
