using System;
using System.Collections.Generic;
using System.Windows.Forms;
using System.Drawing;
using PluginContracts;
using OutputHelperLib;
using System.ComponentModel;
using System.Text;
using System.IO;
using edu.stanford.nlp.ling;
using edu.stanford.nlp.neural.rnn;
using edu.stanford.nlp.pipeline;
using edu.stanford.nlp.sentiment;
using edu.stanford.nlp.trees;
using edu.stanford.nlp.util;
using java.util;
using System.Linq;


namespace CoreNLPSentiment
{
    public class CoreNLPSentiment : Plugin
    {


        public string[] InputType { get; } = { "String" };
        public string OutputType { get; } = "OutputArray";

        public Dictionary<int, string> OutputHeaderData { get; set; } = new Dictionary<int, string>() { {0, "SentNumber"},
                                                                                                        {1, "Classification"},
                                                                                                        {2, "Class_Prob"},
                                                                                                        {3, "Class_Number"},
                                                                                                        {4, "Prob_VeryNeg"},
                                                                                                        {5, "Prob_Neg"},
                                                                                                        {6, "Prob_Neut"},
                                                                                                        {7, "Prob_Pos"},
                                                                                                        {8, "Prob_VeryPos"},
                                                                                                        {9, "SentenceText"} };
        public bool InheritHeader { get; } = false;



        #region Plugin Details and Info

        public string PluginName { get; } = "CoreNLP Sentiment Analysis";
        public string PluginType { get; } = "Sentiment Analysis";
        public string PluginVersion { get; } = "1.1.0";
        public string PluginAuthor { get; } = "Ryan L. Boyd (ryan@ryanboyd.io)";
        public string PluginDescription { get; } = "Built around Stanford's CoreNLP for .NET (v3.9.1, English model). Uses a trained RNN model to classify sentences from \"very negative\" to \"very positive\". Produces sentence-level scores as output" + Environment.NewLine + Environment.NewLine +
            "Manning, Christopher D., Mihai Surdeanu, John Bauer, Jenny Finkel, Steven J. Bethard, and David McClosky. 2014. The Stanford CoreNLP Natural Language Processing Toolkit In Proceedings of the 52nd Annual Meeting of the Association for Computational Linguistics: System Demonstrations, pp. 55-60.";

        public bool TopLevel { get; } = false;
        public string PluginTutorial { get; } = "Coming Soon";
        public Icon GetPluginIcon
        {
            get
            {
                return Properties.Resources.icon;
            }
        }
        #endregion



        private bool includeSentenceText { get; set; } = true;
        private bool useBuiltInSentenceSplitter { get; set; } = true;

        #region Setup Tagger Details
        public static string jarRoot = Path.Combine(Path.GetDirectoryName(AppDomain.CurrentDomain.BaseDirectory),
                                        "Plugins" + Path.DirectorySeparatorChar +
                                        "Dependencies" + Path.DirectorySeparatorChar + @"stanford-corenlp-full-2018-02-27" + Path.DirectorySeparatorChar);

        private StanfordCoreNLP pipeline { get; set; }
        #endregion



        public void ChangeSettings()
        {
            using (var form = new SettingsForm_CoreNLPSentiment(builtInSplitter: useBuiltInSentenceSplitter, textInOutput: includeSentenceText))
            {

                form.Icon = Properties.Resources.icon;

                var result = form.ShowDialog();
                if (result == DialogResult.OK)
                {
                    useBuiltInSentenceSplitter = form.useBuiltInSentenceSplitter;
                    includeSentenceText = form.includeSentenceText;
                }
            }
        }





        public Payload RunPlugin(Payload Input)
        {



            Payload pData = new Payload();
            pData.FileID = Input.FileID;
            bool trackSegmentID = false;
            if (Input.SegmentID.Count > 0)
            {
                trackSegmentID = true;
            }
            else
            {
                pData.SegmentID = Input.SegmentID;
            }



            for (int i = 0; i < Input.StringList.Count; i++)
            {

                //seems to prematurely exit sometimes. checking to see what might cause that -- maybe blank docs?
                if (!string.IsNullOrEmpty(Input.StringList[i]) && !string.IsNullOrWhiteSpace(Input.StringList[i])) {

                    Annotation annotation = new edu.stanford.nlp.pipeline.Annotation();
                    ArrayList sentences = new ArrayList();
                    List<double> SentimentValues = new List<double>();

                    annotation = new edu.stanford.nlp.pipeline.Annotation(Input.StringList[i]);
                    pipeline.annotate(annotation);
                    sentences = annotation.get(new CoreAnnotations.SentencesAnnotation().getClass()) as ArrayList;
                                        

                    int SentenceCount = 0;

                    foreach (CoreMap sentence in sentences)
                    {

                        SentenceCount++;
                        Tree tree = sentence.get(new SentimentCoreAnnotations.SentimentAnnotatedTree().getClass()) as Tree;

                        //add this sentence to our overall list of sentiment scores
                        SentimentValues.Add(RNNCoreAnnotations.getPredictedClass(tree));

     
                        string[] OutputString_SentenceLevel = new string[10] { "", "", "", "", "", "", "", "", "", ""};

                        string Classification = GetClassification((double)RNNCoreAnnotations.getPredictedClass(tree));

                        //this pulls out the prediction probabilites for each class
                        string Predictions = RNNCoreAnnotations.getPredictionsAsStringList(tree).ToString();
                        string[] Predictions_Split = Predictions.Replace("[", "").Replace("]", "").Split(',');
                    

                        OutputString_SentenceLevel[0] = SentenceCount.ToString();
                        OutputString_SentenceLevel[1] = Classification;
                        OutputString_SentenceLevel[2] = RNNCoreAnnotations.getPredictedClassProb(tree.label()).ToString();
                        OutputString_SentenceLevel[3] = RNNCoreAnnotations.getPredictedClass(tree).ToString();
                        OutputString_SentenceLevel[4] = Predictions_Split[0];
                        OutputString_SentenceLevel[5] = Predictions_Split[1];
                        OutputString_SentenceLevel[6] = Predictions_Split[2];
                        OutputString_SentenceLevel[7] = Predictions_Split[3];
                        OutputString_SentenceLevel[8] = Predictions_Split[4];
                        if (includeSentenceText) OutputString_SentenceLevel[9] = sentence.ToString();

                        pData.StringArrayList.Add(OutputString_SentenceLevel);
                        pData.SegmentNumber.Add(Input.SegmentNumber[i]);
                        if (trackSegmentID)
                        {
                            pData.SegmentID.Add(Input.SegmentID[i]);
                        }

                    }

                }
                else
                {
                    pData.StringArrayList.Add(new string[10] { "", "", "", "", "", "", "", "", "", "" });
                    pData.SegmentNumber.Add(Input.SegmentNumber[i]);
                    if (trackSegmentID)
                    {
                        pData.SegmentID.Add(Input.SegmentID[i]);
                    }
                }

                

            }

            return (pData);

        }



        public void Initialize()
        {
            var props = new java.util.Properties();


            props.setProperty("annotators", "tokenize, ssplit, parse, sentiment");

            //if we're using an external sentence segmentation strategy, then this is how we're going to do it
            //https://stackoverflow.com/a/28017131
            if (!useBuiltInSentenceSplitter) props.put("ssplit.isOneSentence", "true");




            props.setProperty("sutime.binders", "0");
            Directory.SetCurrentDirectory(jarRoot);
            pipeline = new StanfordCoreNLP(props);

            

        }

        public Payload FinishUp(Payload Input)
        {
            return (Input);
        }

        private string GetClassification(double y)
        {

            if (y < 0.8) return "Very Negative";
            else if (y < 1.6) return "Negative";
            else if (y < 2.4) return "Neutral";
            else if (y < 3.2) return "Positive";
            else if (y <= 4) return "Very Positive";
            else return "";


        }


        public bool InspectSettings()
        {
            return true;
        }



        #region Import/Export Settings
        public void ImportSettings(Dictionary<string, string> SettingsDict)
        {
            includeSentenceText = Boolean.Parse(SettingsDict["IncludeSentenceText"]);
            useBuiltInSentenceSplitter = Boolean.Parse(SettingsDict["useBuiltInSentenceSplitter"]);
        }

        public Dictionary<string, string> ExportSettings(bool suppressWarnings)
        {
            Dictionary<string, string> SettingsDict = new Dictionary<string, string>();
            SettingsDict.Add("IncludeSentenceText", includeSentenceText.ToString());
            SettingsDict.Add("useBuiltInSentenceSplitter", includeSentenceText.ToString());
            return (SettingsDict);
        }
        #endregion





    }
}
