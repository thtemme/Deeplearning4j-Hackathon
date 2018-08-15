package de.opitzconsulting.deeplearning4j.hackathon;

import org.datavec.api.io.filters.RandomPathFilter;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import sun.awt.image.FileImageSource;
import sun.awt.image.JPEGImageDecoder;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

public class DataPreprocessing {

    private static long seed = 333;
    private static double splitValidation = 0.2;
    private static Random rng = new Random(seed);
    private static String originalDir = "src\\main\\resources\\PetImages";
    private static String validationDir = System.getProperty("user.dir") + "\\" + "src\\main\\resources\\ValidationPetImages";
    private static String classOneName = "Cat";
    private static String classTwoName = "Dog";

    public static void main(String args[]) throws Exception {
        //ToDo: Delete corrupt jpeg data (Fullfill implementation of method).
        deleteCorruptJpegData();
        //ToDo: Fulfill implementation of this method: Split up training and test set
        splitUpTrainingAndTestSet();
    }

    //ToDo: Fullfill implementation of this method
    //Some files are corrupted. Have a look at Cat\10404.jpg for example.
    //If we put them into the training pipeline, Deeplearning4j will crash during training.
    //Therefore we need to validate the data and delete corrupt images, before starting the training pipeline
    private static void deleteCorruptJpegData() throws IOException, URISyntaxException {
        File dirCats = new File(originalDir + "/" + classOneName);
        File dirDogs = new File(originalDir + "/" + classTwoName);
        List<File> fileList = Arrays.stream(dirCats.listFiles()).collect(Collectors.toList());
        fileList.addAll(Arrays.stream(dirDogs.listFiles()).collect(Collectors.toList()));

        File[] directoryListing = dirCats.listFiles();
        if (directoryListing != null) {
            for (File child : directoryListing) {
                try {
                    //ToDo: Use class JPEGImageDecoder in package sun.awt.image to Open JPEG file.
                    //If it succeeds, the JPEG file is valid. If not, an exception will be thrown.
                    //JPEGImageDecoder decoder = ...
                    //decoder.produceImage();
                }
                catch (Exception e) {
                    //ToDo: If an exception is thrown for file, delete it.
                    //You can use java.nio.file.Files for that.
                }
            }
        }
    }

    //ToDo: Split up test set (20%) of images into separate folder.
    //Hint: You can use the functionality of the classes:
    //FileSplit, InputSplit and RandomPathFilter
    //You can find the documentation here:
    //https://deeplearning4j.org/docs/latest/datavec-overview
    private static void splitUpTrainingAndTestSet() {

        File mainPath = new File(System.getProperty("user.dir"), originalDir);
        //RandomPathFilter pathFilter = ...
        //FileSplit fs = ...
        //Hint: You can use sample()-function of FileSplit to create the InputSplit.
        //InputSplit testData = ...

        //Move validation data to different directory
        new File(validationDir + "/" +classOneName).mkdirs();
        new File(validationDir + "/" +classTwoName).mkdirs();

        //ToDo: Comment out and fullfil implementation:
        //Iterate over testData and move Directories to "validationDir"
        //for(Iterator<URI> iter = ...
            //Path path = Paths.get(iter.next());

            //Path filename = path.getFileName();
            //try {
                //if(path.toString().contains(classOneName))
                    //Files.move(path,Paths.get( validationDir + "/" + classOneName + "/" + filename));
                //else
                    //Files.move(path,Paths.get( validationDir + "/" + classTwoName + "/" + filename));
            //} catch (Exception e) {
            //    e.printStackTrace();
            //}
        //}
    }



}
