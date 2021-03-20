using Microsoft.ML.Data;

namespace NeuroSorterLibrary
{
    internal class SortedFilePrediction
    {
        [ColumnName("PredictedLabel")]
        public string Label;

        public float[] Score;
    }
}
