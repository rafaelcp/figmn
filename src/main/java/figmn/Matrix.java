/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package figmn;

import no.uib.cipr.matrix.DenseMatrix;

/**
 *
 * @author rafa
 */
public class Matrix extends DenseMatrix {

    public double det;
    
    public Matrix(int rows, int columns) {
        super(rows, columns);
    }
    
    public Matrix(int rows, int columns, double det) {
        super(rows, columns);
        this.det = det;
    }
    
    public Matrix(DenseMatrix m) {
        super(m);
    }
    
    public Matrix(DenseMatrix m, double det) {
        this(m);
        this.det = det;
    }
}
