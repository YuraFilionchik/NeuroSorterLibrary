using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace NeuroSorterLibrary
{
    /// <summary>
    /// This class use for sorting inputs files using leaned Model
    /// </summary>
    public class Sorter
    {

        private readonly string _modelPath;
        private readonly MLContext _mlContext;
        private readonly PredictionEngine<SortedFile, SortedFilePrediction> _predEngine;
        private readonly ITransformer _trainedModel;
        private List<SortedFile> _sortedFiles;
        private FullPrediction[] _fullPredictions;
        /// <summary>
        /// Sorter file based on learned model
        /// </summary>
        /// <param name="modelPath">Full filename of leaned model</param>
        /// <param name="inputs">List of unlabeled filenames</param>
        public Sorter(string modelPath)
        {
            _modelPath = modelPath;
            //_inputFiles = ConvertToSortedFiles(inputs);

            _mlContext = new MLContext();

            // Load model from file.
            _trainedModel = _mlContext.Model.Load(_modelPath, out var modelInputSchema);

            // Create prediction engine related to the loaded trained model.
            _predEngine = _mlContext.Model.CreatePredictionEngine<SortedFile, SortedFilePrediction>(_trainedModel);

        }
        #region functions   
        /// <summary>
        /// make List<SortedFile> from filename
        /// </summary>
        /// <param name="inputs">file names with full path</param>
        /// <returns></returns>
        private List<SortedFile> ConvertToSortedFiles(IEnumerable<string> inputs)
        {
            List<SortedFile> sortedfiles = new List<SortedFile>();
            if (inputs == null || inputs.Count() <= 0) throw new Exception("Function ConvertToSortedFiles: wrong inputs");
            try
            {
                foreach (string file in inputs)
                {
                    sortedfiles.Add(new SortedFile { FileName= file.Split('\\').Last() });
                }
                return sortedfiles;
            }
            catch (Exception ex)
            {
                throw new Exception("Function ConvertToSortedFiles", ex.InnerException);
               
            }
        }


        #endregion

        private FullPrediction[] GetBestThreePredictions(SortedFilePrediction prediction)
        {
            float[] scores = prediction.Score;
            int size = scores.Length;
            int index0, index1, index2 = 0;

            VBuffer<ReadOnlyMemory<char>> slotNames = default;
            _predEngine.OutputSchema[nameof(SortedFilePrediction.Score)].GetSlotNames(ref slotNames);

            GetIndexesOfTopThreeScores(scores, size, out index0, out index1, out index2);

            _fullPredictions = new FullPrediction[]
                {
                    new FullPrediction(slotNames.GetItemOrDefault(index0).ToString(),scores[index0],index0),
                    new FullPrediction(slotNames.GetItemOrDefault(index1).ToString(),scores[index1],index1),
                    new FullPrediction(slotNames.GetItemOrDefault(index2).ToString(),scores[index2],index2)
                };

            return _fullPredictions;
        }

        private void GetIndexesOfTopThreeScores(float[] scores, int n, out int index0, out int index1, out int index2)
        {
            int i;
            float first, second, third;
            index0 = index1 = index2 = 0;
            if (n < 3)
            {
                Console.WriteLine("Invalid Input");
                return;
            }
            third = first = second = 000;
            for (i = 0; i < n; i++)
            {
                // If current element is  
                // smaller than first 
                if (scores[i] > first)
                {
                    third = second;
                    second = first;
                    first = scores[i];
                }
                // If arr[i] is in between first 
                // and second then update second 
                else if (scores[i] > second)
                {
                    third = second;
                    second = scores[i];
                }

                else if (scores[i] > third)
                    third = scores[i];
            }
            var scoresList = scores.ToList();
            index0 = scoresList.IndexOf(first);
            index1 = scoresList.IndexOf(second);
            index2 = scoresList.IndexOf(third);
        }

        /// <summary>
        /// Predict 3 best variants 
        /// Предсказывает 3 лучших варианта
        /// </summary>
        /// <param name="fullFileName">filename with full path</param>
        /// <returns></returns>
        public FullPrediction[] PredictOne(string fullFileName)
        {
            var prediction = _predEngine.Predict(new SortedFile { FileName = fullFileName.Split('\\').Last() });

            var fullPredictions = GetBestThreePredictions(prediction);

            return fullPredictions;
        }
        public List<SortedFile> PredictAll(IEnumerable<string> fileNames)
        {
            _sortedFiles= ConvertToSortedFiles(fileNames);
            for (int i = 0; i < _sortedFiles.Count; i++)
            {
                SortedFile file = _sortedFiles[i];
                var prediction = _predEngine.Predict(file);
                var fullpredictions = GetBestThreePredictions(prediction);
                if (fullpredictions[0].Score >= 0.3)
                    file.Label = prediction.Label;
                else file.Label = "none";
            }

            return _sortedFiles;
        }
       
    }
}

