package cnn.components;

public interface DoubleMatrixInterface {

	/**
	   * Number of rows in result matrix layer equals rows in this matrix by columns in matrix n.<p/>
	   * Number of rows in n has to equal the number of columns in this, the
	   * previous layer matrix.<p/>
	   * So the matrix is organized by layer+1 node rows by layer column nodes representing weights from
	   * column to row in a feedforward network<p/>.
	   * The bias is contained in an extra column on creation.<p/>
	   * @param n The matrix to multiply and sum against this one
	   * @return The result of the weighted sum of the elements in this matrix multiplied by matrix n
	   */
	DoubleMatrixInterface dot(DoubleMatrixInterface n);

	int getRows();

	int getColumns();

	void randomize();

	/**
	 * Uses the activation function from this matrix and creates a new one with a single column from input  
	 * @param arr array to form new matrix
	 * @return Single column matrix with activation from 'this'
	 */
	DoubleMatrixInterface singleColumnMatrixFromArray(double[] arr);

	double[] toArray();

	/**
	 * Create a new 1 column matrix with rows+1 of this matrix.  
	 * @return The 1 column, rows+1 matrix with elements from column 0 of each row and bias 1 in column 0 of rows+1.
	 */
	DoubleMatrixInterface addBias();

	/**
	 * Perform the activation function on each column of each row of this matrix.  
	 * @return the result of the activation function on each element of each column by each row of this matrix.
	 */
	DoubleMatrixInterface activate();

	//float relu(float x) {
	//       return Math.max(0,x);
	//}
	/**
	 * If a random value exceeds mutation rate from parameters, add a random nextGaussian divided by 5 to
	 * each element of each column of each row of this matrix. If we exceed the thresholds cap at -1 to 1. 
	 * @param mutationRate
	 */
	void mutate(double mutationRate);

	/**
	 * Perform a crossover with the partner Matrix   
	 * @param partner
	 * @return The result of the crossover
	 */
	DoubleMatrixInterface doCross(DoubleMatrixInterface partner);

	double incRan();

	double get(int i, int j);
	void put(int i, int j, double f);

}