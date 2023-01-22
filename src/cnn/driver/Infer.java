/**
 * 
 */
package cnn.driver;

import static cnn.tools.Util.checkNotEmpty;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import com.neocoretechs.neurovolve.NeuralNet;
import com.neocoretechs.neurovolve.Neurosome;
import com.neocoretechs.neurovolve.NeurosomeInterface;
import com.neocoretechs.neurovolve.activation.SoftMax;
import com.neocoretechs.neurovolve.relatrix.ArgumentInstances;
import com.neocoretechs.neurovolve.relatrix.Storage;
import com.neocoretechs.relatrix.DuplicateKeyException;
import com.neocoretechs.relatrix.client.RelatrixClient;

import cnn.components.Plate;
import cnn.tools.Util;

/**
 * @author groff
 *
 */
public class Infer {
	
	// We'll hardwire these in, but more robust code would not do so.
	private static enum Category {
		airplanes, butterfly, flower, grand_piano, starfish, watch
	};
	
	//input nodes, output nodes, hidden nodes, hidden layers
	static int iNodes = 16384;//65536;
	static int oNodes = 6;
	static int hNodes = 300;
	static int hLayers = 1;

	public static int NUM_CATEGORIES = Category.values().length;

	// Store the categories as strings.
	public static List<String> categoryNames = new ArrayList<>();
	static {
		for (Category cat : Category.values()) {
			categoryNames.add(cat.toString());
		}
	}
	/**
	 * Returns the prediction accuracy of this classifier on the test set.
	 * Loads the image or directory of images, creates the Neurosome of stored GUID designated on cmdl, 
	 * then for each image, creates the float vector input, performs 'execute' of neurosome to get the out vector
	 * then calls {@link cnn.driver.Infer.classify} to get the predicted category name which is compared to image name
	 * that presumably contains the category as part of it.
	 * @param args <GUID of stored Neurosome> <image_dir>
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		if(args.length < 5)
			throw new Exception("Usage:java cnn.driver.Infer <LocalIP Client> <Remote IpServer> <DB Port> [<DB Output>] <GUID of Neurosome> <Image file or directory>");
		RelatrixClient ri = new RelatrixClient(args[0], args[1], Integer.parseInt(args[2]));
		boolean directoryIsLabel = false;
		if(args.length == 6) {
			Dataset dataset = null;
			if(args[5].charAt(0) == '/') {
				directoryIsLabel = true;
				dataset = Util.loadDataset(new File(args[5].substring(1)), null, directoryIsLabel);
			} else {
				dataset = Util.loadDataset(new File(args[5]), null, directoryIsLabel);
			}
			System.out.printf("Dataset from %s loaded with %d images%n", args[5], dataset.getSize());
			RelatrixClient ro = new RelatrixClient(args[0], args[1], Integer.parseInt(args[3]));
			testData(ri, ro, dataset, args[4], true);
			ri.close();
			ro.close();
		} else { // no db output
			Dataset dataset = null;
			if(args[4].charAt(0) == '/') {
				directoryIsLabel = true;
				dataset = Util.loadDataset(new File(args[4].substring(1)), null, directoryIsLabel);
			} else {
				dataset = Util.loadDataset(new File(args[4]), null, directoryIsLabel);
			}
			System.out.printf("Dataset from %s loaded with %d images%n", args[4], dataset.getSize());
			test(ri, dataset, args[3], true);
			ri.close();
		}
	}
	/**
	 * Returns the prediction accuracy of this classifier on the test set.
	 * 
	 * Here, accuracy is numCorrectlyClassified/numExamples.
	 * @throws IOException 
	 * @throws IllegalAccessException 
	 * @throws ClassNotFoundException 
	 * @throws IllegalArgumentException 
	 */
	public static double test(RelatrixClient ri, Dataset testSet, String guid, boolean verbose) throws IllegalArgumentException, ClassNotFoundException, IllegalAccessException, IOException {
		int errCount = 0;
		NeurosomeInterface ni = new Neurosome(guid);
		Neurosome n = (Neurosome) Storage.loadSolver2(ri, ni);
		if(n == null)
			throw new RuntimeException("could not locate GUID "+guid+" in database");
		//NeuralNet.SHOWEIGHTS = true;
		//  neurosome, input nodes, output nodes, hidden nodes, hidden layers
		Neurosome.netSet((NeuralNet)n,  iNodes, oNodes, hNodes, hLayers);
		System.out.println("Neurosome "+n.getRepresentation());
		for (Instance img : testSet.getImages()) {
			Plate[] plates = instanceToPlate(img);
			double[] d = packPlates(Arrays.asList(plates));
			double[] outNeuro = n.execute(d);
			System.out.println(/*"Input "+img.toString()+*/" Output:"+Arrays.toString(outNeuro));
			String predicted = classify(img, outNeuro);
			if (!predicted.equals(img.getLabel())) {
				errCount++;
			}	
			if (verbose) {
				System.out.printf("Predicted: %s\t\tActual:%s File:%s\n", predicted, img.getLabel(), img.getName());
			}
		}
		
		double accuracy = ((double) (testSet.getSize() - errCount)) / testSet.getSize();
		if (verbose) {
			System.out.printf("Final accuracy was %.9f\n", accuracy);
		}
		return accuracy;
	}
	
	/**
	 * Generates transfer learning multi task data.
	 * Reads guid Neurosome from db ri, generates output from dataset, writes each output vector to db ro
	 * @throws IOException 
	 * @throws IllegalAccessException 
	 * @throws ClassNotFoundException 
	 * @throws IllegalArgumentException 
	 */
	public static double testData(RelatrixClient ri, RelatrixClient ro, Dataset testSet, String guid, boolean verbose) throws IllegalArgumentException, ClassNotFoundException, IllegalAccessException, IOException {
		int errCount = 0;
		NeurosomeInterface ni = new Neurosome(guid);
		Neurosome n = (Neurosome) Storage.loadSolver2(ri, ni);
		if(n == null)
			throw new RuntimeException("could not locate GUID "+guid+" in database");
		//NeuralNet.SHOWEIGHTS = true;
		//  neurosome, input nodes, output nodes, hidden nodes, hidden layers
		Neurosome.netSet((NeuralNet)n,  iNodes, oNodes, hNodes, hLayers);
		System.out.println("Neurosome "+n.getRepresentation());
		for (Instance img : testSet.getImages()) {
			Plate[] plates = instanceToPlate(img);
			double[] d = packPlates(Arrays.asList(plates));
			double[] outNeuro = n.execute(d);
			System.out.println(/*"Input "+img.toString()+*/" Output:"+Arrays.toString(outNeuro));
			Object[] o = new Object[outNeuro.length];
			for(int i = 0; i < outNeuro.length; i++) {
				o[i] = new Double(outNeuro[i]);
			}
			ArgumentInstances ai = new ArgumentInstances(o);
			try {
				ro.store(guid, img.getName(), ai);
				System.out.println(img.getName()+" Stored!");
			} catch (IllegalAccessException | IOException | DuplicateKeyException e) {
				e.printStackTrace();
			}
			String predicted = classify(img, outNeuro);
			if (!predicted.equals(img.getLabel())) {
				errCount++;
			}	
			if (verbose) {
				System.out.printf("Predicted: %s\t\tActual:%s File:%s\n", predicted, img.getLabel(), img.getName());
			}
		}
		
		double accuracy = ((double) (testSet.getSize() - errCount)) / testSet.getSize();
		if (verbose) {
			System.out.printf("Final accuracy was %.9f\n", accuracy);
		}
		return accuracy;
	}

	/** Returns the predicted label for the image. 
	public static String classify(Instance img, double[] probs) {
		double maxProb = -1;
		int bestIndex = -1;
		for (int i = 0; i < probs.length; i++) {
			if (probs[i] > maxProb) {
				maxProb = probs[i];
				bestIndex = i;
			}
		}
		return categoryNames.get(bestIndex);
	}
	*/
	/** Returns the predicted label for the image. */
	public static String classify(Instance img, double[] probs) {
		//double[] probs = computeOutput(img);
		return classify(probs);
		/*
		double maxProb = -1;
		int bestIndex = -1;
		for (int i = 0; i < probs.length; i++) {
			if (probs[i] > maxProb) {
				maxProb = probs[i];
				bestIndex = i;
			}
		}
		return classes.get(bestIndex);
		*/
	}
	public static String classify(double[] dprobs) {
		double maxProb = -1;
		int bestIndex = -1;
		SoftMax sf = new SoftMax(dprobs);
		for (int i = 0; i < dprobs.length; i++) {
			double smax = sf.activate( dprobs[i]);
			if (smax > maxProb) {
				maxProb = smax;
				bestIndex = i;
			}
		}
		return categoryNames.get(bestIndex);
	}
	private static Plate[] instanceToPlate(Instance instance) {
			return new Plate[] {
					//new Plate(intImgToDoubleImg(instance.getRedChannel())),
					//new Plate(intImgToDoubleImg(instance.getBlueChannel())),
					//new Plate(intImgToDoubleImg(instance.getGreenChannel())),
					new Plate(intImgToDoubleImg(instance.getGrayImage())),
			};
	}
	
	private static double[][] intImgToDoubleImg(int[][] intImg) {
		double[][] dblImg = new double[intImg.length][intImg[0].length];
		for (int i = 0; i < dblImg.length; i++) {
			for (int j = 0; j < dblImg[i].length; j++) {
				dblImg[i][j] = ((double) 255 - intImg[i][j]) / 255;
			}
		}
		return dblImg;
	}
	
	/** 
	 * Pack the plates into a single, 1D double array. Used to connect the plate layers
	 * with the fully connected layers.
	 */
	private static double[] packPlates(List<Plate> plates) {
		checkNotEmpty(plates, "Plates to pack", false);
		int flattenedPlateSize = plates.get(0).getTotalNumValues();
		double[] result = new double[flattenedPlateSize * plates.size()];
		for (int i = 0; i < plates.size(); i++) {
			System.arraycopy(
					plates.get(i).as1DArray(),
					0 /* Copy the whole flattened plate! */,
					result,
					i * flattenedPlateSize,
					flattenedPlateSize);
		}
		return result;
	}
}
