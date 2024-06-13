package Utilities;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.nio.file.Files;
import java.nio.file.Paths;

/**
 * A class for ment for saving models, and saves them in binary.
 * Therefore, this saving and loading process only works for files made
 * from this whole project only.
 * 
 * As this does not directly correlate to the topic of deep learning,
 * yet seemed handy, it was entirely copy-pasted from this source:
 * 
 * https://www.youtube.com/watch?v=-xW0pBZqpjU
 */
public class Model {

    /**
     * saves the serializable object in a file named 'filename',
     * that is inside the src file. 
     * 
     * @param object any Serialized object.
     * @param filename the name of the file.
     */
    public static void save(Serializable object, String filename) {
        try(ObjectOutputStream objectOutputStream = new ObjectOutputStream(Files.newOutputStream(Paths.get(filename)))) {
            objectOutputStream.writeObject(object);
            System.out.println("Successfully saved model!!");
        } catch (Exception e) {
            e.printStackTrace();
            System.out.println("Couldn't save the model...");
        }
    }

    /**
     * If the reference instanced object is not only an Object,
     * it may be converted to it's specific form using Polymorphism.
     * 
     * @param filename the filename of the model we want to load.
     * @return the object inside the file.
     */
    public static Object load(String filename) {
        try(ObjectInputStream objectInputStream = new ObjectInputStream(Files.newInputStream(Paths.get(filename)))) {
            return objectInputStream.readObject();

        } catch(Exception e){
            e.printStackTrace();
            System.out.println("Couldn't load object...");
            return null;
        }
    }
}
