using System;
using System.IO;
// Requires following NuGet packages
// NuGet package -> Microsoft.Extensions.Configuration
// NuGet package -> Microsoft.Extensions.Configuration.Json
using Microsoft.ML;
using System.Collections.Generic;
using System.Linq;

namespace NeuroSorterLibrary
{
    public class BuilderModel
    {
        private static  string _modelRelativePath = @"Data\FilesSorterModel.zip"; //learned model
        private static string _dataSetPath = @"Data\DataSet.csv"; 
        private static  string DataSetPath; //info about sorted files
        private static string ModelPath;
        private static  List<SortedFile> InputFiles=new List<SortedFile>(); //list of input files
        private static List<SortedFile> SortedFiles=new List<SortedFile>(); //list of Sorted files
        public static Exception Errors=new Exception("No errors");
        public enum MyTrainerStrategy : int { SdcaMultiClassTrainer = 1, OVAAveragedPerceptronTrainer = 2 };

        //public static IConfiguration Configuration { get; set; }

        /// <summary>
        /// Creating Dataset from sorted files for learning model, build and train model, save it to file 
        /// </summary>
        /// <param name="sortedDirectories">Directories with sorted files</param>
        /// <param name="unsortedFilesDirectory">input directory with all unsorted files</param>
        /// <param name="rebuildModel">rebuild model if already exist</param>
       
        public static void BuildModel(IEnumerable<string> sortedDirectories, string unsortedFilesDirectory,
             MyTrainerStrategy trainerStrategy = MyTrainerStrategy.OVAAveragedPerceptronTrainer, bool rebuildModel = false)
        {
            try
            {
                SetupConfiguration();
                if (File.Exists(ModelPath) && !rebuildModel) return;

                GetInputFiles(unsortedFilesDirectory);

                //Prepair DataSet 
                BuildDataSet(sortedDirectories);//for learning

                BuildAndTrainModel(DataSetPath, ModelPath, trainerStrategy);

            }
            catch (Exception ex)
            {
                Errors = ex;
            }
           
        }
        /// <summary>
        /// Build and  train new model
        /// </summary>
        /// <param name="sortedDirectories"></param>
        /// <param name="unsortedFiles"></param>
        /// <param name="trainerStrategy"></param>
        /// <param name="rebuildModel"></param>
        public static void BuildModel(IEnumerable<string> sortedDirectories, IEnumerable<string> unsortedFiles,
             MyTrainerStrategy trainerStrategy = MyTrainerStrategy.OVAAveragedPerceptronTrainer, bool rebuildModel = false)
        {
            try
            {
                SetupConfiguration();
                if (File.Exists(ModelPath) && !rebuildModel) return ;

                GetInputFiles(unsortedFiles);

                //Prepair DataSet 
                BuildDataSet(sortedDirectories);//for learning

                BuildAndTrainModel(DataSetPath, ModelPath, trainerStrategy);

            }
            catch (Exception ex)
            {
                Errors = ex;
            }
        }

        public static string GetModelPath()
        {
            return GetAbsolutePath(_modelRelativePath);
        }
        public static string GetDataSetPath()
        {
            return GetAbsolutePath(_dataSetPath);
        }

        /// <summary>
        /// Build dataset from files which sorted by directories
        /// </summary>
        /// <param name="sortedDirs">List of directories with sorted files</param>
        private static void BuildDataSet(IEnumerable<string> sortedDirectories)
        {
            try
            {

                if (sortedDirectories.Count() == 0) throw new Exception("Empty list of sorted directories");
            List<SortedFile> AllSortedFiles = new List<SortedFile>();
            //Dataset - file csv "label;filename;"
            foreach (string directory in sortedDirectories)
            {
                    if (!Directory.Exists(directory)) continue;
                    string label = directory;
                var files = Directory.GetFiles(directory);
                foreach (var file in files)
                {
                    AllSortedFiles.Add(new SortedFile { Label = directory, FileName = file.Split('\\').Last() });
                }
            }
                if (AllSortedFiles.Count <= 0) throw new Exception("Sorted files are absent in all directories");
            string text = "Label;FileName"+Environment.NewLine;
            foreach (SortedFile file in AllSortedFiles)
            {
                text += file.Label + ";" + file.FileName + Environment.NewLine;
            }
            File.WriteAllText(DataSetPath, text);
            }
            catch (Exception ex)
            {
                throw new Exception("BuildDataset: "+ex.Message,ex.InnerException);
            }
            
        }
        /// <summary>
        /// Get List<SortedFile> from directory
        /// </summary>
        /// <param name="inputDir"></param>
        private static void GetInputFiles(string inputDir)
        {
            if (!Directory.Exists(inputDir)) throw new Exception("Directory with unsorted files not found: "+inputDir);
            var files = Directory.GetFiles(inputDir);
            if (files.Length == 0) throw new Exception("Files not found in "+inputDir);
            foreach (var file in files)
            {
                InputFiles.Add(new SortedFile { FileName = file.Split('\\').Last() });
            }
        }
        private static void GetInputFiles(IEnumerable <string> inputFiles)
        {
            var files = inputFiles;
            if (files.Count() == 0) throw new Exception("Files not found");
            foreach (var file in files)
            {
                var fname = file.Split('\\').Last();
               if (!InputFiles.Any(x=>x.FileName==fname)) InputFiles.Add(new SortedFile { FileName = fname });
            }
        }

        private static void SetupConfiguration()
        {
            //IConfigurationBuilder builder = new ConfigurationBuilder()
            //                            .SetBasePath(Directory.GetCurrentDirectory())
            //                            .AddJsonFile("appsettings.json");

            //Configuration = builder.Build();
            ModelPath = GetAbsolutePath(_modelRelativePath);
            DataSetPath = GetAbsolutePath(_dataSetPath);
        }
       
        private  static void BuildAndTrainModel(string DataSetLocation, string ModelPath, MyTrainerStrategy selectedStrategy)
        {
            // Create MLContext to be shared across the model creation workflow objects 
            // Set a random seed for repeatable/deterministic results across multiple trainings.
            var mlContext = new MLContext(seed: 1);

            // STEP 1: Common data loading configuration
            var trainingDataView = mlContext.Data.LoadFromTextFile<SortedFile>(DataSetLocation, hasHeader: true, separatorChar: ';', allowSparse: false);

            // STEP 2: Common data process configuration with pipeline data transformations
            var dataProcessPipeline = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "Label", inputColumnName: nameof(SortedFile.Label))
                            .Append(mlContext.Transforms.Text.FeaturizeText(outputColumnName: "FileNameFeaturized", inputColumnName: nameof(SortedFile.FileName)))
                            .Append(mlContext.Transforms.Concatenate(outputColumnName: "Features", "FileNameFeaturized"))
                            .AppendCacheCheckpoint(mlContext);
            // Use in-memory cache for small/medium datasets to lower training time. 
            // Do NOT use it (remove .AppendCacheCheckpoint()) when handling very large datasets.

            // STEP 3: Create the selected training algorithm/trainer
            IEstimator<ITransformer> trainer = null;
            switch (selectedStrategy)
            {
                case MyTrainerStrategy.SdcaMultiClassTrainer:
                    trainer = mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features");
                    break;
                case MyTrainerStrategy.OVAAveragedPerceptronTrainer:
                    {
                        // Create a binary classification trainer.
                        var averagedPerceptronBinaryTrainer = mlContext.BinaryClassification.Trainers.AveragedPerceptron("Label", "Features", numberOfIterations: 10);
                        // Compose an OVA (One-Versus-All) trainer with the BinaryTrainer.
                        // In this strategy, a binary classification algorithm is used to train one classifier for each class, "
                        // which distinguishes that class from all other classes. Prediction is then performed by running these binary classifiers, "
                        // and choosing the prediction with the highest confidence score.
                        trainer = mlContext.MulticlassClassification.Trainers.OneVersusAll(averagedPerceptronBinaryTrainer);

                        break;
                    }
                default:
                    break;
            }

            //Set the trainer/algorithm and map label to value (original readable state)
            var trainingPipeline = dataProcessPipeline.Append(trainer)
                    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            // STEP 4: Cross-Validate with single dataset (since we don't have two datasets, one for training and for evaluate)
            // in order to evaluate and get the model's accuracy metrics

            var crossValidationResults = mlContext.MulticlassClassification.CrossValidate(data: trainingDataView, estimator: trainingPipeline, numberOfFolds: 6, labelColumnName: "Label");

            // STEP 5: Train the model fitting to the DataSet
            var trainedModel = trainingPipeline.Fit(trainingDataView);

            // (OPTIONAL) Try/test a single prediction with the "just-trained model" (Before saving the model)
           // var inputFile = new SortedFile() { FileName = "ВКС селектор.doc" };
            // Create prediction engine related to the loaded trained model
           // var predEngine = mlContext.Model.CreatePredictionEngine<SortedFile, SortedFilePrediction>(trainedModel);
            //Score
           // var prediction = predEngine.Predict(inputFile);
           // Console.WriteLine($"=============== Single Prediction just-trained-model - Result: {prediction.Label} ===============");
            //

            // STEP 6: Save/persist the trained model to a .ZIP file
           // Console.WriteLine("=============== Saving the model to a file ===============");
            mlContext.Model.Save(trainedModel, trainingDataView.Schema, ModelPath);

        }

        private static string GetAbsolutePath(string relativePath)
        {
            var _dataRoot = new FileInfo(typeof(BuilderModel).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }
    }
}
