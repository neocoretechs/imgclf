package cnn.components;

import java.io.Serializable;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Implementation of a matrix specifically representing neural net nodes.<p/>
 * We include an activation function specified in the constructor.<br/>
 * We can:<br/>
 * Take a dot product<br/>	   
 * Number of rows in result matrix layer equals rows in this matrix by columns in matrix n.<p/>
 * Number of rows in n has to equal the number of columns in this, the
 * previous layer matrix.<p/>
 * So the matrix is organized by layer+1 node rows by layer column nodes representing weights from
 * column to row in a feedforward network<p/>.
 * The bias is contained in an extra column on creation.<p/>
 * <p/>
 * Randomize the matrix<br/>
 * Generate a single column matrix from a float array<br/>
 * Add bias values<br/>
 * Generate a new matrix based on activation function<br/>
 * Clone<br/>
 * Mutate<br/>
 * Crossbreed<br/>
 * 
 * @author Jonathan Groff (C) NeoCoreTechs 6/2020
 *
 */
public class DoubleMatrix implements DoubleMatrixInterface, Serializable, Cloneable {
	private static final long serialVersionUID = -5112564161732998513L;
	int rows, cols;
	double[][] matrix;
	DoubleActivationInterface activationFunction;
	  
	DoubleMatrix(int r, int c, DoubleActivationInterface activationFunction) {
	     rows = r;
	     cols = c;
	     matrix = new double[rows][cols];
	     this.activationFunction = activationFunction;
	}
	   
	public DoubleMatrix(double[][] m, DoubleActivationInterface activationFunction) {
	      matrix = m;
	      rows = matrix.length;
	      cols = matrix[0].length;
		  this.activationFunction = activationFunction;
	}
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
	@Override
	public DoubleMatrixInterface dot(DoubleMatrixInterface n) {
	     DoubleMatrix result = new DoubleMatrix(rows, n.getColumns(), activationFunction);     
	     if(cols != n.getRows()) {
	    	 throw new RuntimeException("Source columns "+cols+" not equal to target rows "+n.getRows()+" for matrix dot product");
	     }
	     for(int i = 0; i < rows; i++) {
	        for(int j = 0; j < n.getColumns(); j++) {
	              double sum = 0;
	              for(int k = 0; k < cols; k++) {
	                 sum += matrix[i][k]*n.get(k,j);
	              }  
	              result.matrix[i][j] = sum;
	        }
	     }
	     return result;
	}
	 
	@Override
	public int getRows() {
		return rows;
	}
	
	@Override
	public int getColumns() {
		return cols;
	}
	
	@Override
	public void randomize() {
	      for(int i = 0; i < rows; i++) {
	         for(int j = 0; j < cols; j++) {
	            matrix[i][j] = incRan();//r.doubles(-1,1); 
	         }
	      }
	}
	/**
	 * Uses the activation function from this matrix and creates a new one with a single column from input  
	 * @param arr array to form new matrix
	 * @return Single column matrix with activation from 'this'
	 */
	@Override
	public DoubleMatrixInterface singleColumnMatrixFromArray(double[] arr) {
	      DoubleMatrix n = new DoubleMatrix(arr.length, 1, activationFunction);
	      for(int i = 0; i < arr.length; i++) {
	         n.matrix[i][0] = arr[i]; 
	      }
	      return n;
	}
	   
	@Override
	public double[] toArray() {
	      double[] arr = new double[rows*cols];
	      for(int i = 0; i < rows; i++) {
	         for(int j = 0; j < cols; j++) {
	            arr[j+i*cols] = matrix[i][j]; 
	         }
	      }
	      return arr;
	}
	/**
	 * Create a new 1 column matrix with rows+1 of this matrix.  
	 * @return The 1 column, rows+1 matrix with elements from column 0 of each row and bias 1 in column 0 of rows+1.
	 */
	@Override
	public DoubleMatrix addBias() {
	      DoubleMatrix n = new DoubleMatrix(rows+1, 1, activationFunction);
	      for(int i = 0; i < rows; i++) {
	         n.matrix[i][0] = matrix[i][0]; 
	      }
	      n.matrix[rows][0] = 1;
	      return n;
	}
	/**
	 * Perform the activation function on each column of each row of this matrix.  
	 * @return the result of the activation function on each element of each column by each row of this matrix.
	 */
	@Override
	public DoubleMatrixInterface activate() {
	      DoubleMatrix n = new DoubleMatrix(rows, cols, activationFunction);
	      for(int i = 0; i < rows; i++) {
	         for(int j = 0; j < cols; j++) {
	            n.matrix[i][j] = activationFunction.activate(matrix[i][j]); //relu(matrix[i][j]); 
	         }
	      }
	      return n;
	}
	   
	//float relu(float x) {
	//       return Math.max(0,x);
	//}
	 /**
	  * Ultimately, mutation should be rare and relies on 2 parameters: mutation probability from parameter, or
	  * the probability that mutation will occur at all, whch should be a few percent or less, and then if
	  * mutation is to occur, the mutation rate here, or the probability that any matrix cell can be mutated,
	  * which means supplying a new random value in the range -1 to 1.
	  * If a random value exceeds mutation rate from parameters, select a new random value for the matrix element
	  * @param mutationRate
	  */
	@Override
	public void mutate(double mutationRate) {
	      for(int i = 0; i < rows; i++) {
	         for(int j = 0; j < cols; j++) {
	            double rand = ThreadLocalRandom.current().nextDouble();
	            if(rand<mutationRate) {
	               matrix[i][j] = incRan();
	            }
	         }
	      }
	}
	/**
	 * Perform a crossover with the partner Matrix.
	 * Arithmatic crossover: weighted average of this and partner nodes.
	 * offs = weight1.multiply(alpha).add(weight2.multiply(1-alpha)).
	 * Where alpha is weighting factor from 0 to 1. 
	 * alpha 0 = favor second parent.
	 * alpha 1 = favor first parent.
	 * alpha.5 = equal blending. 
	 * A random alpha introduces max variation. An alpha with more weight to favor better parent.
	 * @param partner
	 * @return The result of the crossover
	 */
	@Override
	public DoubleMatrix doCross(DoubleMatrixInterface partner) {
	      DoubleMatrix child = new DoubleMatrix(rows, cols, activationFunction);   
	      //int randC = ThreadLocalRandom.current().nextInt(cols);
	      //int randR = ThreadLocalRandom.current().nextInt(rows);
	      double alpha = ThreadLocalRandom.current().nextDouble();
	      for(int i = 0; i < rows; i++) {
	         for(int j = 0;  j < cols; j++) {
	            //if((i < randR) || (i == randR && j <= randC)) {
	            //   child.matrix[i][j] = matrix[i][j]; 
	            //} else {
	            //  child.matrix[i][j] = partner.get(i,j);
	            //}
	        	 child.matrix[i][j] = (matrix[i][j] * alpha) + (partner.get(i,j) * (1.0 - alpha));
	         }
	      }
	      return child;
	}
	
	@Override  
	protected DoubleMatrix clone() {
	      DoubleMatrix clone = new DoubleMatrix(rows, cols, activationFunction);
	      for(int i = 0; i < rows; i++) {
	         for(int j = 0; j < cols; j++) {
	            clone.matrix[i][j] = matrix[i][j]; 
	         }
	      }
	      return clone;
	}
	   
	@Override
	public double incRan() {
		return (2 * ThreadLocalRandom.current().nextDouble() - 1);   // between -1.0 and 1.0
	}
	   
	public String toString() {
		   StringBuilder s = new StringBuilder("\r\nDoubleMatrix:\r\n");
		   for(int i = 0; i < rows; i++) {
			   s.append("["+i+"] ");
			   for(int j = 0; j < cols; j++) {
			      s.append(matrix[i][j]+" "); 
			   }
			   s.append("\r\n");
		   }
		   s.append("End DoubleMatrix\r\n");
		   return s.toString();
	}

	@Override
	public double get(int i, int j) {
		return matrix[i][j];
	}

	@Override
	public void put(int i, int j, double f) {
		matrix[i][j] = f;	
	}
}
