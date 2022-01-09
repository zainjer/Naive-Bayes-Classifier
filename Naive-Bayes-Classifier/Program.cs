using System;
using System.Collections.Generic;
using System.IO;

namespace Naive_Bayes_Classifier
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Naive Bayes\n");
            var filePath = "corpus.txt";
            var predictionVarCount = 3;
            var classCount = 2;
            var totalDataItems = 40;
            var data = LoadData(filePath, totalDataItems, predictionVarCount + 1, ',');
            Console.WriteLine("Training data:");
            for (int i = 0; i < totalDataItems; ++i)
            {
                Console.Write("[" + i + "] ");
                for (int j = 0; j < predictionVarCount + 1; ++j)
                {
                    Console.Write(data[i][j] + " ");
                }

                Console.WriteLine();
            }

            Console.WriteLine(". . . \n");
            var jointCts = MatrixInt(predictionVarCount, classCount);
            var yCts = new int[classCount];
            var X = new string[] {"Cyan", "Small", "Twisted"};
            Console.WriteLine("Item to classify: ");
            for (int i = 0; i < predictionVarCount; ++i)
            {
                Console.Write(X[i] + " ");
            }
            Console.WriteLine();
            // Compute joint counts and y counts
            for (int i = 0; i < totalDataItems; ++i)
            {
                var y = int.Parse(data[i][predictionVarCount]);
                ++yCts[y];
                for (var j = 0; j < predictionVarCount; ++j)
                {
                    if (data[i][j] == X[j])
                    {
                        ++jointCts[j][y];
                    }
                }
            }

            //LapLace Smoothing
            for (var i = 0; i < predictionVarCount; ++i)
            {
                for (var j = 0; j < classCount; ++j)
                {
                    ++jointCts[i][j];
                }
            }

            Console.WriteLine("Joint counts: ");
            for (var i = 0; i < predictionVarCount; ++i)
            {
                for (var j = 0; j < classCount; ++j)
                {
                    Console.Write(jointCts[i][j] + " ");
                }

                Console.WriteLine("");
            }

            Console.WriteLine("\nClass counts: ");
            for (var k = 0; k < classCount; ++k)
                Console.Write(yCts[k] + " ");
            Console.WriteLine("\n");
            // Compute evidence terms
            var eTerms = new double[classCount];
            for (var k = 0; k < classCount; ++k)
            {
                var v = 1.0;
                for (var j = 0; j < predictionVarCount; ++j)
                {
                    v *= (double) (jointCts[j][k]) / (yCts[k] + predictionVarCount);
                }

                v *= (double) (yCts[k]) / totalDataItems;
                eTerms[k] = v;
            }

            Console.WriteLine("Evidence terms:");
            for (var k = 0; k < classCount; ++k)
            {
                Console.Write(eTerms[k].ToString("F4") + " ");
            }

            Console.WriteLine();

            var evidence = 0.0;
            for (int k = 0; k < classCount; ++k)
            {
                evidence += eTerms[k];
            }

            var probs = new double[classCount];
            for (var k = 0; k < classCount; ++k)
            {
                probs[k] = eTerms[k] / evidence;
            }

            Console.WriteLine("Probabilities: ");
            for (var k = 0; k < classCount; ++k)
            {
                Console.Write(probs[k].ToString("F4") + " ");
            }

            Console.WriteLine();
            var pc = ArgMax(probs);
            Console.WriteLine("Predicted class: ");
            Console.WriteLine(pc);
        }

        static List<List<string>> MatrixString(int rows, int cols)
        {
            var matrix = new List<List<string>>();
            for (var i = 0; i < rows; ++i)
            {
                var list = new List<string>() { };
                for (var j = 0; j < cols; j++)
                {
                    list.Add(string.Empty);
                }

                matrix.Add(list);
            }

            return matrix;
        }

        static List<List<int>> MatrixInt(int rows, int cols)
        {
            var matrix = new List<List<int>>();
            for (var i = 0; i < rows; ++i)
            {
                var list = new List<int>() { };
                for (var j = 0; j < cols; j++)
                {
                    list.Add(default);
                }

                matrix.Add(list);
            }

            return matrix;
        }

        static List<List<string>> LoadData(string filePath, int rows,
            int columns, char splitter)
        {
            var result = MatrixString(rows, columns);
            var allLines = File.ReadAllLines(filePath);
            var tokens = Array.Empty<string>();
            var i = 0;
            foreach (var line in allLines)
            {
                tokens = line.Split(splitter);
                for (int j = 0; j < columns; ++j)
                    result[i][j] = tokens[j];
                ++i;
            }

            return result;
        }

        static int ArgMax(double[] vector)
        {
            int result = 0;
            double maxV = vector[0];
            for (int i = 0; i < vector.Length; ++i)
            {
                if (vector[i] > maxV)
                {
                    maxV = vector[i];
                    result = i;
                }
            }

            return result;
        }
    }
}