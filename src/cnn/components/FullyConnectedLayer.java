package cnn.components;

import static cnn.tools.Util.checkNotNull;
import static cnn.tools.Util.checkPositive;
import static cnn.tools.Util.outerProduct;
import static cnn.tools.Util.scalarMultiply;
import static cnn.tools.Util.tensorSubtract;

import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;

import com.neocoretechs.neurovolve.multiprocessing.SynchronizedFixedThreadPoolManager;

import cnn.components.NeurosomeLayer.Builder;
import cnn.driver.Main;
import cnn.tools.ActivationFunction;

/** 
 * Your standard fully-connected ANN.
 * 
 * This class stores the weights between inputs and nodes, and provides
 * functionality for computing the output given an input vector and for
 * back-propagating errors.
 */
public class FullyConnectedLayer implements LayerInterface {
	private final double[][] weights;
	private final double[] lastInput;
	private final double[] lastOutput;
	private final ActivationFunction activation;
	private static AtomicInteger threadIndex = new AtomicInteger(0);
	private static int maxThreads = 48;
	
	public FullyConnectedLayer() {
		this.weights = null;
		this.lastInput = null;
		this.lastOutput = null;
		this.activation = null;
		SynchronizedFixedThreadPoolManager.getInstance().init(maxThreads, maxThreads, "COMPUTE");
	}
	
	private FullyConnectedLayer(double[][] weights, ActivationFunction activation) {
		this.weights = weights;
		this.lastInput = new double[weights[0].length];
		this.lastOutput = new double[weights.length];
		this.activation = activation;
		
		// Set the last value to be the offset. This will never change.
		this.lastInput[this.lastInput.length - 1] = -1;
		SynchronizedFixedThreadPoolManager.getInstance().init(maxThreads, maxThreads, "COMPUTE");
	}
	
	public double[][] getWeights() {
		return weights;
	}
	
	public ActivationFunction getActivationFunction() {
		return activation;
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
		
		System.arraycopy(input, 0, lastInput, 0, input.length);
		for (int i = 0; i < lastOutput.length; i++) {
			double sum = 0;
			for (int j = 0; j < lastInput.length; j++) {
				sum += weights[i][j] * lastInput[j];
			}
			lastOutput[i] = activation.apply(sum);
		}
		return lastOutput;
	}

	/** 
	 * Given the error from the previous layer, update the weights and return the error
	 * for this layer.
	 */
	public double[] propagateError(double[] proppedDelta, double learningRate) {
		if (proppedDelta.length != weights.length) {
			throw new IllegalArgumentException(
					String.format(
							"Got length %d delta, expected length %d!",
							proppedDelta.length,
							weights.length));
		}
		
		// Compute deltas for the next layer.
		double[] delta = new double[weights[0].length - 1]; // Don't count the offset here.
		for (int i = 0; i < delta.length; i++) {
			Future<?>[] jobs = new Future[weights.length];
			threadIndex.set(i);
			    jobs[i] = SynchronizedFixedThreadPoolManager.submit(new Runnable() {
			    	@Override
			    	public void run() {
			    		int i = threadIndex.getAndIncrement();
			    		for (int j = 0; j < weights.length; j++) {
			    			delta[i] += proppedDelta[j] * weights[j][i] * activation.applyDerivative(lastInput[i]);
			    		}
			      	} // run
			    },"COMPUTE"); // spin
			SynchronizedFixedThreadPoolManager.waitForCompletion(jobs);
		}
		
		// Update the weights using the propped delta.
		tensorSubtract(
				weights,
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
		builder.append("\n------\tFully Connected Layer\t------\n\n");
		builder.append(
				String.format("Number of inputs: %d (plus a bias)\n", weights[0].length - 1));
		builder.append(String.format("Number of nodes: %d\n", weights.length));
		builder.append(String.format("Activation function: %s\n", activation.toString()));
		builder.append("\n\t------------\t\n");
		return builder.toString();
	}
	
	/** Returns a new builder. */
	public static Builder<? extends LayerInterface> newBuilder() { return new Builder<>(); }
	
	/** Simple builder pattern for organizing parameters. */
	public static class Builder<T extends LayerInterface> implements BuildableLayerInterface{
		private ActivationFunction func = null;
		private int numInputs = 0;
		private int numNodes = 0;
		
		public Builder() {}

		public Builder<T> setActivationFunction(ActivationFunction func) {
			checkNotNull(func, "Fully connected activation function");
			this.func = func;
			return this;
		}
		
		public Builder<T> setNumInputs(int numInputs) {
			checkPositive(numInputs, "Number of fully connected inputs", false);
			this.numInputs = numInputs;
			return this;
		}
		
		public Builder<T> setNumNodes(int numNodes) {
			checkPositive(numNodes, "Number of fully connected nodes", false);
			this.numNodes = numNodes;
			return this;
		}
		
		public T build() {
			checkNotNull(func, "Fully connected activation function");
			checkPositive(numInputs, "Number of fully connected inputs", true);
			checkPositive(numNodes, "Number of fully connected nodes", true);
			double[][] weights = new double[numNodes][numInputs + 1];
			for (int i = 0; i < weights.length; i++) {
				for (int j = 0; j < weights[i].length; j++) {
					weights[i][j] = Main.getRandomWeight(numInputs, numNodes);
				}
			}
			return (T) new FullyConnectedLayer(weights, func);
		}
	}
}
