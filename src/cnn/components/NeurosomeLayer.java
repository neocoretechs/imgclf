package cnn.components;

import static cnn.tools.Util.checkNotNull;
import static cnn.tools.Util.checkPositive;
import static cnn.tools.Util.outerProduct;
import static cnn.tools.Util.scalarMultiply;
import static cnn.tools.Util.tensorSubtract;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;

import com.neocoretechs.neurovolve.Neurosome;
import com.neocoretechs.neurovolve.NeurosomeInterface;
import com.neocoretechs.neurovolve.multiprocessing.SynchronizedFixedThreadPoolManager;
import com.neocoretechs.neurovolve.relatrix.Storage;
import com.neocoretechs.relatrix.client.RelatrixClient;

import cnn.driver.Main;
import cnn.tools.ActivationFunction;

/** 
 * Your standard fully-connected ANN using Neurosome protocol.
 * 
 * This class stores the weights between inputs and nodes, and provides
 * functionality for computing the output given an input vector and for
 * back-propagating errors.
 */
public class NeurosomeLayer implements LayerInterface {
	private final DoubleMatrix weights;
	private final double[] lastInput;
	private double[] lastOutput;
	DoubleActivationInterface act;
	private static int maxThreads = 48;
	private static AtomicInteger threadIndex = new AtomicInteger(0);

	private NeurosomeLayer(double[][] weights, ActivationFunction activation) {
		this.act = activation;
		//System.out.println("weights.length="+weights.length+" weights[0].length="+weights[0].length);
		this.weights = new DoubleMatrix(weights, activation);
		this.lastInput = new double[weights[0].length];
		this.lastOutput = new double[weights.length];
		// Set the last value to be the offset. This will never change.
		this.lastInput[this.lastInput.length - 1] = -1;
		SynchronizedFixedThreadPoolManager.getInstance().init(maxThreads, maxThreads, "COMPUTE");
	}
	
	public double[][] getWeights() {
		return weights.matrix;
	}
	
	public ActivationFunction getActivationFunction() {
		return (ActivationFunction) act;
	}
	
	/** Compute the output of the given input vector. */
	public double[] computeOutput(double[] input) {
		if (input.length != lastInput.length - 	1) {
			throw new IllegalArgumentException(
					String.format(
							"Input length in fully connected layer was %d, should be %d.",
							input.length,
							lastInput.length));
		}
		/*
		System.arraycopy(input, 0, lastInput, 0, input.length);
		for (int i = 0; i < lastOutput.length; i++) {
			float sum = 0;
			for (int j = 0; j < lastInput.length; j++) {
				sum += weights.get(i,j) * ((float)lastInput[j]);
			}
			lastOutput[i] = act.activate(sum);
		}
		return lastOutput;
		*/
		System.arraycopy(input, 0, lastInput, 0, input.length);
		DoubleMatrixInterface m = weights.singleColumnMatrixFromArray(lastInput);
		DoubleMatrixInterface mo = weights.dot(m);
		DoubleMatrixInterface moa = mo.activate();
		//System.out.println("lastInput.length="+lastInput.length+" lastOutput.length="+lastOutput.length+" moa.rows="+moa.getRows()+" moa.cols="+moa.getColumns());
		lastOutput = moa.toArray();
		return lastOutput;
	}

	/** 
	 * Given the error from the previous layer, update the weights and return the error
	 * for this layer.
	 */
	public double[] propagateError(double[] proppedDelta, double learningRate) {
		if (proppedDelta.length != weights.getRows()) {
			throw new IllegalArgumentException(
					String.format(
							"Got length %d delta, expected length %d!",
							proppedDelta.length,
							weights.getRows()));
		}
		
		// Compute deltas for the next layer.
		double[] delta = new double[weights.getColumns() - 1]; // Don't count the offset here.
		threadIndex.set(0);
		Future<?>[] jobs = new Future[delta.length];
		for (int i = 0; i < delta.length; i++) {
			    jobs[i] = SynchronizedFixedThreadPoolManager.submit(new Runnable() {
			    	@Override
			    	public void run() {
			    		int i = threadIndex.getAndIncrement();
			    		for (int j = 0; j < weights.getRows(); j++) {
			    			delta[i] += proppedDelta[j] * weights.get(j,i) * ((ActivationFunction) act).applyDerivative(lastInput[i]);
			    		}
			      	} // run
				},"COMPUTE"); // spin
		}
		SynchronizedFixedThreadPoolManager.waitForCompletion(jobs);
		
		// Update the weights using the propped delta.
		tensorSubtract(
				getWeights(),
				scalarMultiply(
						learningRate,
						outerProduct(proppedDelta, lastInput),
						true /* inline */),
				true /* inline */);
		return delta;
	}
	
	@Override
	public String toString() {
		StringBuilder builder = new StringBuilder();
		builder.append("\n------\tFully Connected Neurosome Layer\t------\n\n");
		builder.append(
				String.format("Number of inputs: %d (plus a bias)\n", weights.getColumns() - 1));
		builder.append(String.format("Number of nodes: %d\n", weights.getRows()));
		builder.append(String.format("Activation function: %s\n", act.toString()));
		builder.append("\n\t------------\t\n");
		return builder.toString();
	}
	
	/** Returns a new builder. */
	public static Builder<? extends LayerInterface> newBuilder() { return new Builder<>(); }
	
	/** Simple builder pattern for organizing parameters. */
	public static class Builder<T extends LayerInterface> implements BuildableLayerInterface {
		private ActivationFunction func = null;
		private int numInputs = 0;
		private int numNodes = 0;
		
		public Builder() {}

		public Builder<T> setActivationFunction(ActivationFunction func) {
			checkNotNull(func, "Neurosome activation function");
			this.func = func;
			return this;
		}
		
		public Builder<T> setNumInputs(int numInputs) {
			checkPositive(numInputs, "Number of Neurosome inputs", false);
			this.numInputs = numInputs;
			return this;
		}
		
		public Builder<T> setNumNodes(int numNodes) {
			checkPositive(numNodes, "Number of Neurosome nodes", false);
			this.numNodes = numNodes;
			return this;
		}
		
		public T build() {
			checkNotNull(func, "Neurosome activation function");
			checkPositive(numInputs, "Number of Neurosome inputs", true);
			checkPositive(numNodes, "Number of Neurosome nodes", true);
			double[][] weights = new double[numNodes][numInputs + 1];
			for (int i = 0; i < weights.length; i++) {
				for (int j = 0; j < weights[i].length; j++) {
					weights[i][j] = Main.getRandomWeight(numInputs, numNodes);
				}
			}
			return (T) new NeurosomeLayer(weights, func);
		}
	}
	
	/**
	 * Build up a list of NeurosomeLayers to send to the ConvoltionalNeuralNetwork builder.
	 * @param args
	 */
	public static List<LayerInterface> loadSolver(RelatrixClient ri, String sguid) {
		List<LayerInterface> li = new ArrayList<LayerInterface>();
		NeurosomeInterface guid;// = new Neurosome(sguid);
		try {
			guid = Storage.loadSolver(ri,  sguid);
		} catch (IllegalArgumentException e) {
			throw new RuntimeException(e);
		}
		// guid should be loaded up wth layers
		double[][][] na = guid.toArray();
		for(int i = 0; i < na.length; i++) {
			li.add(new NeurosomeLayer(na[i], ActivationFunction.SIGMOID));
		}
		return li;
	}
}
