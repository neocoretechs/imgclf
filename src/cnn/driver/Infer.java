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

import com.neocoretechs.neurovolve.relatrix.Storage;
import com.neocoretechs.neurovolve.worlds.RelatrixWorld;

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

		public static int NUM_CATEGORIES = Category.values().length;

		// Store the categories as strings.
		public static List<String> categoryNames = new ArrayList<>();
		static {
			for (Category cat : Category.values()) {
				categoryNames.add(cat.toString());
			}
		}
	/**
	 * @param args <GUID of stored Neurosome> <image_dir>
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		if(args.length < 2)
			throw new Exception("Usage:java Infer <GUID of Neurosome> <Image file or directory>");
		Dataset dataset = Util.loadDataset(new File(args[1]), null, true);
		System.out.printf("Dataset from %s loaded with %d images%n", args[1], dataset.getSize());
		// Construct a new world to spin up remote connection
		try {
			new RelatrixWorld(new String[] {});
		} catch (IllegalAccessException | IOException e) {
			e.printStackTrace();
			System.exit(1);
		}
		test(dataset, args[0], true);
	}
	/**
	 * Returns the prediction accuracy of this classifier on the test set.
	 * 
	 * Here, accuracy is numCorrectlyClassified/numExamples.
	 */
	public static double test(Dataset testSet, String guid, boolean verbose) {
		int errCount = 0;
		for (Instance img : testSet.getImages()) {
			Plate[] plates = instanceToPlate(img);
			double[] d = packPlates(Arrays.asList(plates));
			float[] inFloat = new float[d.length];
			for(int i = 0; i < d.length; i++)
				inFloat[i] = (float) d[i];
			float[] outNeuro = Storage.executeSolver(guid, inFloat);
			String predicted = classify(img, outNeuro);
			if (!predicted.equals(img.getLabel())) {
				errCount++;
			}	
			if (verbose) {
				System.out.printf("Predicted: %s\t\tActual:%s\n", predicted, img.getLabel());
			}
		}
		
		double accuracy = ((double) (testSet.getSize() - errCount)) / testSet.getSize();
		if (verbose) {
			System.out.printf("Final accuracy was %.9f\n", accuracy);
		}
		return accuracy;
	}
	
	/** Returns the predicted label for the image. */
	public static String classify(Instance img, float[] probs) {
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
	
	private static Plate[] instanceToPlate(Instance instance) {
			return new Plate[] {
					new Plate(intImgToDoubleImg(instance.getRedChannel())),
					new Plate(intImgToDoubleImg(instance.getBlueChannel())),
					new Plate(intImgToDoubleImg(instance.getGreenChannel())),
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
