package cnn.components;


import cnn.tools.ActivationFunction;

public interface BuildableLayerInterface {
	public BuildableLayerInterface setActivationFunction(ActivationFunction func);
	
	public BuildableLayerInterface setNumInputs(int numInputs);
	
	public BuildableLayerInterface setNumNodes(int numNodes);
	
	public LayerInterface build();
}
