package cnn.tools;

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
import cnn.driver.Dataset;
import cnn.driver.Instance;


public class InferTest3 {
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
	 * Generates transfer learning multi task data.
	 * Reads guid Neurosome from db ri, generates output from dataset, writes each output vector to db ro
	 * @throws IOException 
	 * @throws IllegalAccessException 
	 * @throws ClassNotFoundException 
	 * @throws IllegalArgumentException 
	 */
	public static double xferTests(/*RelatrixClient ri,*/String db, Dataset testSet, String guid, String guidc, String guidt, boolean verbose) throws IllegalArgumentException, ClassNotFoundException, IllegalAccessException, IOException {
		int errCount = 0;
		int improved = 0;
		int degraded = 0;
		int bothwrong = 0;
		int bothright = 0;
		int total = 0;
		//NeurosomeInterface ni = new Neurosome(guid);
		//Neurosome n = (Neurosome) Storage.loadSolver(ri, ni);
		Neurosome n = (Neurosome) Storage.loadSolver(guid, db);
		if(n == null)
			throw new RuntimeException("could not locate parent GUID "+guid+" in database");
		Neurosome nc = (Neurosome) Storage.loadSolver(guidc, db);
		if(nc == null)
			throw new RuntimeException("could not locate child GUID "+guidc+" in database");
		Neurosome nt = (Neurosome) Storage.loadSolver(guidt, db);
		if(nt == null)
			throw new RuntimeException("could not locate combined GUID "+guidt+" in database");
		//  neurosome, input nodes, output nodes, hidden nodes, hidden layers
		//Neurosome.netSet((NeuralNet)n,  iNodes, oNodes, hNodes, hLayers);
		//NeuralNet.SHOWEIGHTS = true;
		System.out.println("Neurosome Parent "+n.getRepresentation());
		System.out.println("Neurosome Child "+nc.getRepresentation());
		System.out.println("Neurosome Conbined "+nt.getRepresentation());
		for (Instance img : testSet.getImages()) {
			System.out.println("===== "+total+" =====");
			Plate[] plates = instanceToPlate(img);
			double[] d = packPlates(Arrays.asList(plates));
			double[] outNeuroC = n.execute(d);
			// chain the output
			double[] outNeuroPC = nc.execute(outNeuroC);
			System.out.println("Input "+img.getLabel()+" Output Parent->child:"+Arrays.toString(outNeuroPC));

			// exec same input individually
			double[] outNeuro = nt.execute(d);
			System.out.println("Input "+img.getLabel()+" Output Combined:"+Arrays.toString(outNeuro));			
			boolean oInErr = false;
			String opredicted = classify(img, outNeuroPC);
			if (!opredicted.equals(img.getLabel())) {
				oInErr = true;
			}	
			if (verbose) {
				System.out.printf("Predicted P->C: %s\t\tActual:%s cat=%d File:%s\n", opredicted, img.getLabel(),categoryNames.indexOf(img.getLabel()), img.getName());
			}
			boolean nInErr = false;
			String predicted = classify(img, outNeuro);
			if (!predicted.equals(img.getLabel())) {
				nInErr = true;
				errCount++;
			}	
			if (verbose) {
				System.out.printf("Predicted Comb: %s\t\tActual:%s File:%s\n", predicted, img.getLabel(), img.getName());
			}
			if(nInErr && oInErr) {
				++bothwrong;
				System.out.println("Both Still wrong!");
			} else {
				if(oInErr && !nInErr) {
					System.out.println(">>>>>>>>>>>>>PREDICTION CORRECTED!");
					++improved;
				} else {
					if(!oInErr && nInErr) {
						System.out.println("**********PREDICTION DEGRADED!!!!!!!");
						++degraded;
					} else {
						if(!oInErr && !nInErr) {
							++bothright;
							System.out.println("Both are right...");
						} else {
							throw new RuntimeException("Absolutely impossible condition! Universe in peril!");
						}
					}
				}
			}
			// if both true they both are wrong
			// if both false they are both right
			++total;
		}
		
		double accuracy = ((double) (testSet.getSize() - errCount)) / testSet.getSize();
		if(improved+degraded+bothwrong+bothright != total)
			System.out.printf("Discrepency in total, total=%d, stats=%d%n",total, (improved+degraded+bothwrong+bothright));
		
		if (verbose) {
			System.out.printf("Final accuracy was %.9f total=%d improved=%d degraded=%d both wrong=%d both right=%d\n", accuracy, total, improved, degraded, bothwrong, bothright);
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
		System.out.print("Output :[");
		SoftMax sf = new SoftMax(dprobs);
		for (int i = 0; i < dprobs.length; i++) {
			double smax = sf.activate( dprobs[i]);
			System.out.print(smax+", ");
			if (smax > maxProb) {
				maxProb = smax;
				bestIndex = i;
			}
		}
		System.out.println("] ="+bestIndex);
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
//		if(args.length != 6)
//			throw new Exception("Usage:java cnn.tools.InferTest <LocalIP Client> <Remote IpServer> <DB Port> <GUID of Neurosome> <GUID of xfer Neurosome> <Image file or directory>");
//		RelatrixClient ri = new RelatrixClient(args[0], args[1], Integer.parseInt(args[2]));
		if(args.length != 5)
			throw new Exception("Usage:java cnn.tools.InferTest3 <DB> <GUID of Parent Neurosome> <GUID of child Neurosome> <GUID of Combined Neurosome> <Image file or directory>");
		boolean directoryIsLabel = false;
		//if(args.length == 6) {
			Dataset dataset = null;
			//if(args[5].charAt(0) == '/') {
			if(args[3].charAt(0) == '/') {
				directoryIsLabel = true;
				//dataset = Util.loadDataset(new File(args[5].substring(1)), null, directoryIsLabel);
				dataset = Util.loadDataset(new File(args[4].substring(1)), null, directoryIsLabel);
			} else {
				//dataset = Util.loadDataset(new File(args[5]), null, directoryIsLabel);
				dataset = Util.loadDataset(new File(args[4]), null, directoryIsLabel);
			}
			/*
			System.out.printf("Dataset from %s loaded with %d images%n", args[5], dataset.getSize());
			xferTests(ri, dataset, args[3], args[4], true);
			ri.close();
			*/
			System.out.printf("Dataset from %s loaded with %d images%n", args[3], dataset.getSize());
			xferTests(args[0], dataset, args[1], args[2], args[3], true);
		//}
	}
}
