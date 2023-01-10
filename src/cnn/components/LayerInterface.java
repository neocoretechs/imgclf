package cnn.components;

public interface LayerInterface {

	double[] computeOutput(double[] vec);

	double[] propagateError(double[] fcError, double learningRate);
	

}
