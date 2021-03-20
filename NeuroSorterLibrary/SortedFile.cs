using Microsoft.ML.Data;

namespace NeuroSorterLibrary
{
    /// <summary>
    /// Label - присвоенная метка или папка для файла FileName
    /// </summary>
    public class SortedFile
    {
        [LoadColumn(0)]
        public string Label;
        [LoadColumn(1)]
        public string FileName;
    }
}
