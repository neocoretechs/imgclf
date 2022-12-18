package cnn.tools;

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Random;

import javax.imageio.ImageIO;

import com.neocoretechs.machinevision.ParallelCannyEdgeDetector;
import com.neocoretechs.neurovolve.Matrix;
import com.neocoretechs.neurovolve.Neurosome;
import com.neocoretechs.neurovolve.NeurosomeInterface;
import com.neocoretechs.neurovolve.activation.ActivationInterface;
import com.neocoretechs.neurovolve.activation.ReLU;
import com.neocoretechs.neurovolve.activation.Sigmoid;
import com.neocoretechs.neurovolve.relatrix.Storage;
import com.neocoretechs.neurovolve.worlds.RelatrixWorld;
import com.neocoretechs.relatrix.client.RelatrixClient;

import cnn.ConvolutionalNeuralNetwork;
import cnn.components.FullyConnectedLayer;
import cnn.driver.Dataset;
import cnn.driver.Instance;

/** Utility methods and objects used throughout the network. */
public final class Util {
	public static final int SEED = 0;
	public static final Random RNG = new Random(SEED);
	
	/** Performs v1 * v2^T. */
	public static double[][] outerProduct(double[] v1, double[] v2) {
		checkVectorNotNullOrEmpty(v1);
		checkVectorNotNullOrEmpty(v2);
		double[][] result = new double[v1.length][v2.length];
		for (int i = 0; i < result.length; i++) {
			for (int j = 0; j < result[i].length; j++) {
				result[i][j] = v1[i] * v2[j];
			}
		}
		return result;
	}
	
	/** Computes the inner product of two vectors. */
	public static double innerProduct(double[] v1, double[] v2) {
		checkVectorNotNullOrEmpty(v1);
		checkVectorNotNullOrEmpty(v2);
		checkVectorDimensionsMatch(v1, v2);
		double result = 0;
		for (int i = 0; i < v1.length; i++) {
			result += v1[i] * v2[i];
		}
		return result;
	}
	
	/** Performs vector scalar multiplication. See description for 3D version. */
	public static double[] scalarMultiply(double scalar, double[] vector, boolean inline) {
		return scalarMultiply(scalar, new double[][][]{{ vector }}, inline)[0][0];
	}
	
	/** Performs matrix scalar multiplication. See description for 3D version. */
	public static double[][] scalarMultiply(double scalar, double[][] matrix, boolean inline) {
		return scalarMultiply(scalar, new double[][][]{ matrix }, inline)[0];
	}
	
	/**
	 * Performs tensor (3D matrix) scalar multiplication.
	 * 
	 * If inline is true, this method directly mutates the given tensor.
	 */
	public static double[][][] scalarMultiply(double scalar, double[][][] tensor, boolean inline) {
		checkTensorNotNullOrEmpty(tensor);
		double[][][] result = inline
				? tensor
				: new double[tensor.length][tensor[0].length][tensor[0][0].length];
		for (int i = 0; i < tensor.length; i++) {
			for (int j = 0; j < tensor[i].length; j++) {
				for (int k = 0; k < tensor[i][j].length; k++) {
					result[i][j][k] = tensor[i][j][k] * scalar;
				}
			}
 		}
		return result;
	}
	
	/** Performs v1 - v2 (for vectors). See description for 3D version. */
	public static double[] tensorSubtract(double[] v1, double[] v2, boolean inline) {
		return tensorSubtract(new double[][][] {{ v1 }}, new double[][][]{{ v2 }}, inline)[0][0];
	}
	
	/** Performs m1 - m2 (for matrices). See description for 3D version. */
	public static double[][] tensorSubtract(double[][] m1, double[][] m2, boolean inline) {
		return tensorSubtract(new double[][][]{ m1 }, new double[][][]{ m2 }, inline)[0];
	}
	
	/** Performs t1 - t2 (for 3D tensors). See description for 3D tensorAdd. */
	public static double[][][] tensorSubtract(double[][][] t1, double[][][] t2, boolean inline) {
		return tensorAdd(t1, scalarMultiply(-1, t2, inline), inline);
	}
	
	/** Performs v1 + v2 (for vectors). See description for 3D version. */
	public static double[] tensorAdd(double[] v1, double[] v2, boolean inline) {
		return tensorAdd(new double[][][]{{ v1 }}, new double[][][]{{ v2 }}, inline)[0][0];
	}
	
	/** Performs m1 + m2 (for matrices). See description for 3D version. */
	public static double[][] tensorAdd(double[][] m1, double[][] m2, boolean inline) {
		return tensorAdd(new double[][][]{ m1 }, new double[][][]{ m2 }, inline)[0];
	}
	
	/**
	 * Performs t1 + t2 (for 3D tensors).
	 * 
	 * If inline is true, this method directly mutates t1.
	 */
	public static double[][][] tensorAdd(double[][][] t1, double[][][] t2, boolean inline) {
		checkTensorNotNullOrEmpty(t1);
		checkTensorNotNullOrEmpty(t2);
		checkTensorDimensionsMatch(t1, t2);
		double[][][] result = inline ? t1 : new double[t1.length][t1[0].length][t1[0][0].length];
		for (int i = 0; i < result.length; i++) {
			for (int j = 0; j < result[i].length; j++) {
				for (int k = 0; k < result[i][j].length; k++) {
					result[i][j][k] = t1[i][j][k] + t2[i][j][k];
				}
			}
		}
		return result;
	}
	
	private static void checkVectorNotNullOrEmpty(double[] vector) {
		checkNotNull(vector, "Vector arg");
		checkPositive(vector.length, "Vector length", false);
	}
	
	private static void checkVectorDimensionsMatch(double[] v1, double[] v2) {
		if (v1.length != v2.length) {
			throw new IllegalArgumentException(
					String.format(
							"Lengths %d and %d do not match for inner product.\n",
							v1.length,
							v2.length));
		}
	}
	
	/** Verifies that the tensor is not null and that all 3 dimensions have length > 0. */
	private static void checkTensorNotNullOrEmpty(double[][][] tensor) {
		checkNotNull(tensor, "Tensor arg");
		checkPositive(tensor.length, "Tensor dimension 1", false);
		checkPositive(tensor[0].length, "Tensor dimension 2", false);
		checkPositive(tensor[0][0].length, "Tensor dimension 3", false);
	}
	
	/** Verifies that the tensors have the same dimensions. */
	private static void checkTensorDimensionsMatch(double[][][] t1, double[][][] t2) {
		if (t1.length != t2.length 
				|| t1[0].length != t2[0].length
				|| t1[0][0].length != t2[0][0].length) {
			throw new IllegalArgumentException(
					String.format(
							"Tensor dimensions do not match...\tT1:%dx%dx%d\tT2:%dx%dx%d\n",
							t1.length,
							t1[0].length,
							t1[0][0].length,
							t2.length,
							t2[0].length,
							t2[0][0].length));
		}
	}
	
	/** Verifies that the value with the given name is in the range [min, max). */
	public static void checkValueInRange(int val, int min, int max, String name) {
		if (val < min || val > max) {
			throw new IllegalArgumentException(
					String.format(
							"%s was %d, but should be in range [%d, %d)", name, val, min, max));
		}
	}
	
	/** Verifies that the object with the given name is not null. */
	public static void checkNotNull(Object obj, String name) {
		if (obj == null) {
			throw new NullPointerException(String.format("%s was null!", name));
		}
	}
	
	/**
	 * Verifies that the value with the given name is not null.
	 * 
	 * The boolean parameter specifies which type of exception to throw.
	 */
	public static void checkPositive(double val, String name, boolean sourceIsStateful) {
		if (val <= 0) {
			if (sourceIsStateful) {
				throw new IllegalStateException(String.format("%s was not set!", name));
			} else {				
				throw new IllegalArgumentException(String.format("%s must be positive!", name));
			}
		}
	}
	
	/**
	 * Checks that the collection with the given name is nonempty.
	 * 
	 * Again, the boolean parameter specifies the type of exception to throw.
	 */
	public static void checkNotEmpty(Collection<?> coll, String name, boolean sourceIsStateful) {
		if (coll.isEmpty()) {
			if (sourceIsStateful) {
				throw new IllegalStateException(
						String.format("%s must have at least one value!", name));
			} else {
				throw new IllegalArgumentException(String.format("%s must be nonempty!", name));
			}
		}
	}
	
	/** Prints current heap usage. */
	public static void printMemory() {
		// Get current size of heap in bytes
		long heapSize = Runtime.getRuntime().totalMemory(); 

		// Get amount of free memory within the heap in bytes. This size will increase
		// after garbage collection and decrease as new objects are created.
		long heapFreeSize = Runtime.getRuntime().freeMemory(); 

		// Get maximum size of heap in bytes. The heap cannot grow beyond this size.
		// Any attempt will result in an OutOfMemoryException.
		long heapMaxSize = Runtime.getRuntime().maxMemory();
		
		System.out.println("\nCurrent Heap Size: " + heapSize / Math.pow(2,20) + " MB");
		System.out.println("Free Heap Size: " + heapFreeSize / Math.pow(2, 20) + " MB");
		System.out.println("Max Heap Size: " + heapMaxSize / Math.pow(2, 20) + " MB");
	}
	
	public static Dataset loadDataset(File dir, String label, boolean directoryIsLabel) {
		Dataset d = new Dataset();
		ArrayList<File> fileList = new ArrayList<File>();
		if(dir.isFile()) {
			fileList.add(dir);
		} else {
			for (File file : dir.listFiles()) {
				// check all files
				if (!file.isFile() || !file.getName().endsWith(".jpg")) {
					continue;
				}
				fileList.add(file);
			}
		}
		for(File fi: fileList ) {
			// String path = file.getAbsolutePath();
			BufferedImage img = null, scaledBI = null;
			try {
				System.out.println("Reading "+fi.getName());
				// load in all images
				img = ImageIO.read(fi);
				// every image's name is in such format: label_image_XXXX(4 digits) though this code could handle more than
				// 4 digits.
				String name = fi.getName();
				int locationOfUnderscoreImage = name.indexOf("_image");

				// Resize the image if requested. Any resizing allowed, but should really be one of 8x8, 16x16, 32x32, or
				// 64x64 (original data is 128x128).
				if (img.getHeight() != 128 || img.getWidth() != 128) {
					scaledBI = new BufferedImage(128, 128, BufferedImage.TYPE_INT_RGB);
					Graphics2D g = scaledBI.createGraphics();
					g.drawImage(img, 0, 0, 128, 128, null);
					g.dispose();
				}
				if(label == null && directoryIsLabel)
					name = dir.getName();
				else
					if(label == null && locationOfUnderscoreImage == -1)
						name = "UNNOWN";
					else
						if(label == null)
							name = name.substring(0, locationOfUnderscoreImage);
						else
							name = label;
				
				Instance instance = new Instance(fi.getName(), scaledBI == null ? img : scaledBI, name);

				d.add(instance);

			} catch (IOException e) {
				System.err.println("Error: cannot load in the image file");
				System.exit(1);
			}
		}
		return d;
	}
	
	public static Neurosome storeAsNeurosome(RelatrixClient ri, ConvolutionalNeuralNetwork cnn) {
		// build Neurovolve neurosome and store
		List<FullyConnectedLayer> fcc = cnn.getFullyConnectedLayers();
		Matrix[] weights = new Matrix[fcc.size()];
		int allLayers = 0;
		ActivationInterface ai = null;
		for(FullyConnectedLayer fcl : fcc) {
			double[][] elems = fcl.getWeights();
			float[][] felems = new float[elems.length][elems[0].length];
			ActivationFunction af = fcl.getActivationFunction();
			if( af.equals(ActivationFunction.RELU) )
				ai = new ReLU();
			else
				if(af.equals(ActivationFunction.SIGMOID))
					ai = new Sigmoid();
				else
					throw new RuntimeException("Source activation function not recognized:"+af);
			for(int i = 0; i < elems.length; i++)
				for(int j = 0; j < elems[0].length; j++)
					felems[i][j] = (float) elems[i][j];
			weights[allLayers++] = new Matrix(felems, ai);
		}
		
		// Construct a new world to spin up remote connection
		try {
			new RelatrixWorld(new String[] {});
		} catch (IllegalAccessException | IOException e) {
			e.printStackTrace();
			System.exit(1);
		}
		// input nodes, output nodes, hidden nodes, hidden layers, weights, activation
		int hiddenLayers = weights.length-1;
		int inputNodes = weights[0].getColumns();
		int hiddenNodes = weights[0].getRows();
		int outputNodes = weights[weights.length-1].getRows();
		//NeuralNet(int input, int hidden, int output, int hiddenLayers, Matrix[] weights, ActivationInterface activationFunction) {
		NeurosomeInterface neurosome = new Neurosome(inputNodes, hiddenNodes, outputNodes, hiddenLayers, weights, ai);
		Storage.storeSolver(ri, neurosome, new Class[] {float[].class}, float[].class);
		return (Neurosome) neurosome;
	}
	
	public static void main(String[] args) throws Exception {
		String prefix = "D:/etc/images/trainset/";
		String oprefix = "D:/etc/images/trainsetedge/";
		Dataset dataset = Util.loadDataset(new File(prefix), null, false);
		System.out.printf("Dataset from %s loaded with %d images%n", prefix, dataset.getSize());
		ParallelCannyEdgeDetector detector = new ParallelCannyEdgeDetector("EDGEDETECT");
		detector.setLowThreshold(0.1f);
		detector.setHighThreshold(.5f);
		//apply it to an image
	    List<Instance> images = dataset.getImages();
	    images.stream().forEach(i -> {
		 detector.setSourceImage(i.getImage());
		 detector.process();
		 BufferedImage edges = detector.getEdgesImage();
		 try {
			ImageIO.write(edges, "JPEG", new File(oprefix+i.getName()));
		} catch (IOException e) {
			e.printStackTrace();
		}
	    });
	}

}
