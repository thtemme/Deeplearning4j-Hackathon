package de.opitzconsulting.deeplearning4j.hackathon;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.*;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Random;

import static java.lang.Math.toIntExact;

public class CatsDogsClassification {

    protected static final Logger log = LoggerFactory.getLogger(CatsDogsClassification.class);
    protected static long seed = 42;
    protected static Random rng = new Random(seed);

    protected static int height = 100;
    protected static int width = 100;
    protected static int channels = 3;
    protected static int batchSize = 20;
    protected static int usedTrainImages = 3000;
    protected static int numLabels = 2;

    protected static int epochs = 25;

    private static String trainingPathDir = "src\\main\\resources\\PetImages";
    private static String validationDir = "src\\main\\resources\\ValidationPetImages";
    private static String modelFileName = "model.zip";

    public static void main(String args[]) throws Exception {


        //Normalize grey values of image channels between 0 and 1
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);

        RecordReaderDataSetIterator trainDataIter = generateRecordReaderDataSetIterator(trainingPathDir);

        MultiLayerNetwork network = hackathonBasicNetwork();
        log.info("Get model...");
        try {
            //ToDo: Try to load saved model, if it is available in file "modelFileName".
            //Look at class ModelSerializer https://deeplearning4j.org/modelpersistence
            //network = ...
        }
        catch (Exception e) {
            //If model cannot be deserialized from file, a fresh model is loaded.
            network = hackathonBasicNetwork();
        }

        //Visit http://localhost:9000/train to watch training progress
        startUIServer(network);

        log.info("Train model....");
        while (true) {
            trainModel(network,scaler,trainDataIter, 1);
            log.info("Evaluate model....");
            evaluateModel(scaler, network);


            //ToDo: Save model in file "model.zip" after each training epoch.
            //This will allow you to resume training and reuse your model for recognition, later.
            //Look at class ModelSerializer https://deeplearning4j.org/modelpersistence
            //File modelFile = new File(modelFileName);
            //...
        }
    }

    private static void trainModel(MultiLayerNetwork network, DataNormalization scaler, RecordReaderDataSetIterator iter, int numEpochs) {
        scaler.fit(iter);
        iter.setPreProcessor(scaler);
        network.fit(iter, numEpochs);
    }

    //ToDo: There's a big bug within this method. Can you fix it?
    private static void evaluateModel(DataNormalization scaler, MultiLayerNetwork network) throws Exception {
        DataSetIterator evalIter = generateRecordReaderDataSetIterator(trainingPathDir);
        scaler.fit(evalIter);
        evalIter.setPreProcessor(scaler);
        Evaluation eval = network.evaluate(evalIter);
        log.info(eval.stats(true));
    }

    private static RecordReaderDataSetIterator generateRecordReaderDataSetIterator(String pathDir) throws Exception {
        //ParentPathLabelGenerator will automatically treat image folders as output neurons
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        File mainPath = new File(System.getProperty("user.dir"), pathDir);
        FileSplit fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, rng);
        int numExamples = toIntExact(fileSplit.length());

        //We expect each subdirectory as a separate class with images
        int numLabels = fileSplit.getRootDir().listFiles(File::isDirectory).length;
        BalancedPathFilter pathFilter = new BalancedPathFilter(rng, labelMaker, numExamples, numLabels, usedTrainImages);
        InputSplit[] inputSplit = fileSplit.sample(pathFilter);
        InputSplit data = inputSplit[0];

        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);

        recordReader.initialize(data, null);
        return new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);

    }

    private static void startUIServer(MultiLayerNetwork network) {
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        network.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(100));
    }


    /**
     * ToDo: This network configuration works but it's results are not the best.
     * You can improve the network configuration by
     * researching for better image classification networks and adapt the MultiLayerConfiguration.
     *
     * You can try to improve this network or directly have a look at a better architecture.
     *
     * Recommendation:
     * You may have a look at AlexNet (https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
     *
     * Steps: Adapt Activation function, Updater (https://deeplearning4j.org/updater -> Momentum instead of easy gradient descent)
     * Add more layers, Check size of convolutional filters, etc.
     *
     * If you get stuck at this point, you can also have a look at the Deeplearning4j ModelZoo (https://deeplearning4j.org/model-zoo).
     * You can also load complete Networkconfigurations as Maven dependency in the project.
     */

    public static MultiLayerNetwork hackathonBasicNetwork() {

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .l2(0.005)
                .activation(Activation.SIGMOID)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(0.0001,0.9))
                .list()
                .layer(0, new ConvolutionLayer.Builder(
                        //Kernel size
                        new int[]{5, 5},
                        //Stride
                        new int[]{1, 1},
                        //Padding
                        new int[]{0, 0})
                        .name("cnn1")
                        .nIn(channels)
                        .nOut(25)
                        .biasInit((double) 0).build())
                .layer(1, new SubsamplingLayer.Builder(
                        //Kernel
                        new int[]{2, 2},
                        //Stride
                        new int[]{2, 2})
                        .name("maxpool1").build())
                .layer(2, new ConvolutionLayer.Builder(
                        //Kernel
                        new int[]{5, 5},
                        //Stride
                        new int[]{5, 5},
                        //Pad
                        new int[]{1, 1})
                        .name("cnn2")
                        .nOut(500)
                        .biasInit((double) 0).build())
                .layer(3, new SubsamplingLayer.Builder(
                        //Kernel
                        new int[]{2, 2},
                        //Stride
                        new int[]{2, 2})
                        .name("maxool2").build())
                .layer(4, new DenseLayer.Builder().nOut(500).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(numLabels)
                        .activation(Activation.SOFTMAX)
                        .build())
                .backprop(true).pretrain(false)
                .setInputType(InputType.convolutional(height, width, channels))
                .build();

        return new MultiLayerNetwork(conf);

    }


}

