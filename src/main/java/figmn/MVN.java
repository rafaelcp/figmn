/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package figmn;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import no.uib.cipr.matrix.DenseVector;
import no.uib.cipr.matrix.Vector;
import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.Matrices;


/**
 *
 * @author rafa
 */
//public class MVN extends MultivariateNormalDistribution implements Serializable {
public class MVN implements Serializable {

    public double count = 1;
    //public ArrayList<Double> distances;
    //transient public HashMap<Integer, Double> distances;
    public DenseMatrix invCov;
    //transient CholeskyDecomposition cd;
    public double logdet;
    public double lognormalizer;
    public DenseVector mean;
    public DenseVector diag;
    public double age = 1;
    public double[] correlateds;
    protected double chisq = Double.NaN;
    public boolean useDiagonal  = false;
    public int[] features = null;
    public double attrselection = 1;
        
    public int size() {
        return mean.size();
    }
    
    public double getPartialCorrelationBetween(int i, int j) {
        //System.out.println(invCov);
        return -invCov.get(i, j) / Math.sqrt(invCov.get(i, i) * invCov.get(j, j));
    }

    public double getPartialCorrelationBetween(int i, int[] js) {
        double sum = 0;
        for (int k = 0; k < js.length; k++) {
            sum += getPartialCorrelationBetween(i, js[k]);
        }
        return sum / js.length;
    }
    
    public double[] getPartialCorrelations(int js[]) {
        double[] c = new double[mean.size() - js.length];
        double total = 0;
        if (js.length == 0) {
            js = Matrices.index(0, mean.size());
        }
        for (int i = 0; i < c.length; i++) {
            c[i] = Math.abs(getPartialCorrelationBetween(i, js));
            total += c[i];
        }
        if (total > 0) {
            for (int i = 0; i < c.length; i++) {
                c[i] /= total;
            }
        }
        return c;
    }
    
    public int[] getOrderedFeatures(int[] js, double threshold) {
        //if (this.features != null) return this.features;
        Integer[] inds = new Integer[mean.size() - js.length];
        double total = 0;
        for (int i = 0; i < inds.length; i++) {
            inds[i] = i;
        }
        double[] corrs = getPartialCorrelations(js);
        Arrays.sort(inds, new Comparator<Integer>() {
            @Override public int compare(final Integer o1, final Integer o2) {
                return Double.compare(corrs[o2], corrs[o1]);
            }
        });
        int[] f = new int[inds.length];
        int sz = inds.length;
        for (int i = 0; i < inds.length; i++) {
            f[i] = inds[i];
            total += corrs[inds[i]];
            if (total > threshold) {
                sz = i + 1;
                break;
            }
        }
        this.features = Arrays.copyOf(f, sz);
        return this.features;
    }
    
    public double[] getCorrelationsFor(int attr) {
        if (correlateds != null) return correlateds;
        int D = size();
        double[] corr = new double[D];
        DenseMatrix c = invCov;//new CholeskyDecomposition(invCov).getSolver().getInverse();
        double stdAttr = c.get(attr, attr);//Math.sqrt(c.get(attr, attr));
        if (stdAttr == 0) return corr;
        double total = 0;
        for (int i = 0; i < D; i++) {
            double entry = c.get(i, i);
            if (i == attr) continue;
            //corr[i] = Math.abs(c.get(attr, i));// / (Math.sqrt(c.get(i, i))*stdAttr);
            corr[i] = Double.MIN_VALUE + Math.abs(c.get(attr, i) / (Math.sqrt(entry*stdAttr)));
            //corr[i] = Double.MIN_VALUE + Math.abs(c.get(attr, i));
            //System.out.println(corr[i]);
            total += corr[i];
        }
        if (total == 0) return corr;
        for (int i = 0; i < D; i++) {
            if (i == attr) continue;
            //corr[i] /= total;
        }
        //System.out.println(Arrays.toString(corr));
        correlateds = corr;
        return corr;        
    }
    
    public double[] getCorrelations() {
        //if (correlateds != null) return correlateds;
        int D = size();
        double[] corr = new double[D];
        double total = 0;
        for (int i = 0; i < D; i++) {
            double[] corrsi = getCorrelationsFor(i);
            for (int j = 0; j < D; j++) {
                if (i == j) continue;
                corr[i] += corrsi[j];
            }
            total += corr[i];
        }
        for (int i = 0; i < D; i++) {
            corr[i] /= total;
        }
        //correlateds = corr;
        return corr;
    }
    
    public int[] getMostCorrelateds(int attr, double threshold) {
        int D = size();
        //ArrayList<Integer> indices = new ArrayList<Integer>();
        double total = 0;
        double[] corrs = getCorrelationsFor(attr);
        final DenseVector v = new DenseVector(corrs);
        final DenseVector v2 = v.copy().scale(1.0/v.copy().norm(Vector.Norm.One));
        Integer[] inds = new Integer[corrs.length];
        for (int i = 0; i < inds.length; i++) {
            inds[i] = i;
        }
        Arrays.sort(inds, new Comparator<Integer>() {
            @Override public int compare(final Integer o1, final Integer o2) {
                return Double.compare(v2.get(o2), v2.get(o1));
            }
        });
        int i;
        for (i = 0; i < inds.length; i++) {
            //if (corrs[i] > 0) {
                //indices.add(inds[i]);
                total += v2.get(inds[i]);//corrs[inds[i]];
                if (total >= threshold) break;
            //}
        }        
        int[] r = new int[Math.min(i+1,inds.length)];
        for (int j = 0; j < r.length; j++) {
            r[j] = inds[j];
        }
        return r;
    }

    public int[] getMostCorrelateds(double threshold) {
        int D = size();
        //ArrayList<Integer> indices = new ArrayList<Integer>();
        double total = 0;
        final double[] corrs = getCorrelations();
        Integer[] inds = new Integer[corrs.length];
        for (int i = 0; i < inds.length; i++) {
            inds[i] = i;
        }
        Arrays.sort(inds, new Comparator<Integer>() {
            @Override public int compare(final Integer o1, final Integer o2) {
                return Double.compare(corrs[o2], corrs[o1]);
            }
        });
        int i;
        for (i = 0; i < inds.length; i++) {
            //if (corrs[i] > 0) {
                //indices.add(inds[i]);
                total += corrs[inds[i]];
                if (total >= threshold) break;
            //}
        }        
        int[] r = new int[Math.min(i+1,inds.length)];
        for (int j = 0; j < r.length; j++) {
            r[j] = inds[j];
        }
        return r;
    }
    
    public int getMostCorrelatedTo(int attr) {
        double[] corrs = getCorrelationsFor(attr);
        int most = 0;//(int)Math.floor(Math.random()*corrs.length);
        double mostValue = 0;
        for (int i = 1; i < corrs.length; i++) {
            if (corrs[i] > mostValue) {
                mostValue = corrs[i];
                most = i;
            }
        }
        return most;
    }
    
    public double totalCorrelationFor(int attr) {
        double mc = 0;
        int D = size();
        for (int i = 0; i < D; i++) {
            if (i == attr) continue;
            mc += invCov.get(attr, i);
        }
        return mc;
    }

    public MVN(double[] mean, DenseMatrix invCov, double d) throws Exception {
        logdet = d;
        this.invCov = invCov;
        this.mean = new DenseVector(mean);
        if (Double.isNaN(logdet) || Double.isInfinite(logdet)) {
            throw(new Exception("Invalid log determinant: "+logdet));
        }
        setLogNormalizer();
    }
    
    public void setLogNormalizer() {
        lognormalizer = - 0.5 * (logdet +  this.mean.size() * Math.log(2*Math.PI));        
    }

    public MVN(double[] mean, DenseMatrix invCov, double d, boolean diagonal) throws Exception {
        DenseVector diag = new DenseVector(new double[invCov.numColumns()]);
        for (int i = 0; i < diag.size(); i++) {
            diag.set(i, invCov.get(i, i));
        }
        if (Double.isNaN(logdet) || Double.isInfinite(logdet)) {
            throw(new Exception("Invalid log determinant: "+logdet));
        }
        
        this.diag = diag;
        this.useDiagonal = true;
        double[] rv = new double[mean.length];
        for (int i = 0; i < mean.length; i++) {
            rv[i] = 1 / diag.get(i);
        }
        this.diag = new DenseVector(rv);
        this.mean = new DenseVector(mean);
        setLogNormalizer();
        System.out.println("Using diagonal covariance");

    }
    
    public MVN(double[] mean, DenseVector diag, double d) throws Exception {
        if (Double.isNaN(logdet) || Double.isInfinite(logdet)) {
            throw(new Exception("Invalid log determinant: "+logdet));
        }
        
        this.diag = diag;
        this.useDiagonal = true;
        double[] rv = new double[mean.length];
        for (int i = 0; i < mean.length; i++) {
            rv[i] = 1 / diag.get(i);
        }
        this.diag = new DenseVector(rv);
        this.mean = new DenseVector(mean);
        //normalizer = 1.0 / Math.sqrt(Math.pow(2*Math.PI,mean.length) * det);
        setLogNormalizer();
        //if (Double.isNaN(normalizer) || Double.isInfinite(normalizer) || normalizer < Double.MIN_VALUE) {
        //    normalizer = Double.MIN_VALUE;
        //}
        //System.out.println("d "+logdet+ " n "+lognormalizer);
       //distances = new HashMap<Integer, Double>();
        //System.out.println("logDet: "+logdet+" logNormalizer: "+lognormalizer);
    }
    
    
    public double[] getMeans() {
        return mean.getData();
    }

    @Override
    public String toString() {
        //return count + ":" + Arrays.toString(getMeans()) + Arrays.toString(diag.getData()) + "\n";
        return count + ":" + Arrays.toString(getMeans()) + "\n";
        //return count + ":" + Arrays.toString(getMeans()) + "\nInv Cov: " + getInverseCov() + "\n";
    }

    public DenseVector getMeansVector() {
        return mean.copy();
    }
    
    public double mahalanobis(final DenseVector x) {
        //DenseVector x_ = x.copy();
        ArrayList<Integer> inds = new ArrayList<>();
        ArrayList<Integer> outputs = new ArrayList<>();
        
        for (int i = 0; i < x.size(); i++) {
            if (Double.isNaN(x.get(i))) {
               x.set(i, getMeansVector().get(i));
               outputs.add(i);
            }
            else {
                inds.add(i);
            }
        }
        int[] inds_ = inds.stream().mapToInt(i -> i).toArray();
        int[] outputs_ = outputs.stream().mapToInt(i -> i).toArray();
        DenseVector diff = (DenseVector) x.add(-1, mean);
        if (useDiagonal || count == 1) return diagMahalanobis(diff);
        if (attrselection < 1) {
            inds_ = getOrderedFeatures(outputs_, attrselection);// 1 84.35000000000001% 0.5 68.34%
            //System.out.println("features "+Arrays.toString(inds_));
        }
        //DenseVector subx = new DenseVector(Matrices.getSubVector(x, inds_));
        //DenseVector submean = new DenseVector(Matrices.getSubVector(mean, inds_));
        DenseVector subdiff = new DenseVector(Matrices.getSubVector(diff, inds_));
        /*double m = diagMahalanobis(diff,chisq);
        if (m > chisq) {
            return m;
        }*/
        //System.out.println("x_ "+x_+" diff "+diff+" inv "+getInverseCov()+" mean "+getMeansVector());
        /*if (useDiagonal) {
            System.out.println("Using diagonal covariance to compute density");
            return diff.ebeDivide(diag).dotProduct(diff);
        }
        else {*/
        //    return getInverseCov().mult(invCov, invCov).preMultiply(diff).dotProduct(diff);
        //System.out.println("full: "+count);
        //DenseVector temp = new DenseVector(x.size());
        //getInverseCov().mult(diff, temp);
        DenseVector temp = new DenseVector(subdiff.size());
        DenseMatrix subinvcov = new DenseMatrix(Matrices.getSubMatrix(invCov, inds_, inds_));
        //getInverseCov().mult(diff, temp);
        subinvcov.mult(subdiff, temp);
        //System.out.println(getInverseCov());  
        //System.out.println("FOI "+Arrays.toString(inds_)+" for "+Arrays.toString(outputs_)+" subx "+temp.dot(subdiff));
        return temp.dot(subdiff);            
        //}
    }
    
    public double diagMahalanobis(DenseVector diff) {
        double dist = 0;
        //System.out.println("diag mahalanobis" + count);
        if (diag == null) {
            diag = new DenseVector(invCov.numColumns());
            for (int i = 0; i < invCov.numRows(); i++) {
                    diag.set(i, invCov.get(i, i));
            }
        }
        for (int i = 0; i < diff.size(); i++) {
            dist += Math.pow(diff.get(i),2) * diag.get(i);
        }
        return dist;
    }
    
    public double diagMahalanobis(DenseVector diff, double max) {
        double dist = 0;
        for (int i = 0; i < diff.size(); i++) {
            dist += Math.pow(diff.get(i),2) / diag.get(i);
            if (dist > max) break;
        }
        return dist;
    }

    public double mahalanobis(final double[] x) {
        return mahalanobis(new DenseVector(x));
    }

    /*public double bhattacharyya(MVN other) {
        double bd;
        DenseVector diff = getMeansVector().subtract(other.getMeansVector());
        DenseMatrix meanCov = getCovariances().add(other.getCovariances()).scalarMultiply(0.5);
        //CholeskyDecomposition cd1 = new CholeskyDecomposition(getCovariances());
        //CholeskyDecomposition cd2 = new CholeskyDecomposition(other.getCovariances());
        CholeskyDecomposition cd = new CholeskyDecomposition(meanCov);
        DecompositionSolver cds = cd.getSolver();
        bd = cds.getInverse().preMultiply(diff).dotProduct(diff) / 8.0;
        bd = bd + 0.5 * Math.log(cd.getDeterminant() / Math.sqrt(getDeterminant() * other.getDeterminant()));
        return bd;
    }*/
    
    public DenseMatrix getInverseCov() {
        return invCov;
    }

    /*public double getDeterminant() {
        return det;
    }

    public static double getDeterminantFrom(DenseMatrix rm) {  
        CholeskyDecomposition cd_ = new CholeskyDecomposition(rm);
        return cd_.getDeterminant();
    }*/

    /*public double bhattCoeff(MVN other) {
        double bc;
        int otherHash = other.hashCode();
        if (distances.containsKey(otherHash)) {
            bc = distances.get(otherHash);
            //System.out.println("CACHED");
        } else {
            bc = 1.0 / Math.exp(bhattacharyya(other));
            distances.put(otherHash, bc);
        }
        return bc;
    }*/

/*    public MVN merge(MVN other) {
        DenseVector m1 = getMeansVector();
        DenseVector m2 = other.getMeansVector();
        DenseVector m = m1.mapMultiply(count).add(m2.mapMultiply(other.count)).mapDivide(count + other.count);
        DenseMatrix c = getCovariances().scalarMultiply(count).add(other.getCovariances().scalarMultiply(other.count));
        c = c.add(m1.outerProduct(m1).scalarMultiply(count)).add(m2.outerProduct(m2).scalarMultiply(other.count));
        c = c.scalarMultiply(1.0 / (count + other.count));
        c = c.subtract(m.outerProduct(m));
        MVN newComp = new MVN(m.getData(), c.getData());
        newComp.count = count + other.count;
        return newComp;
    }
*/
    public static int countNaNs(double[] input) {
        int c = 0;
        for (int i = 0; i < input.length; i++) {
            if (Double.isNaN(input[i])) {
                c++;
            }
        }
        return c;
    }

//    @Override
    public double density(final double[] input) {
        return Math.exp(logdensity(input));
    }

    public double logdensity(final double[] input) {
        double[] ret = input.clone();
        for (int i = 0; i < input.length; i++) {
            if (Double.isNaN(input[i])) {
                ret[i] = getMeans()[i];
            }
        }
        //MultivariateNormalDistribution mvn = new MultivariateNormalDistribution(this.mean.getData(),new CholeskyDecomposition(this.invCov).getSolver().getInverse().getData());
        //double d0 = super.density(ret);
        //double d = mvn.density(ret);
        //double d = Math.exp(-0.5 * mahalanobis(ret)) * normalizer + Double.MIN_VALUE;
        double d = -0.5 * mahalanobis(ret) + lognormalizer;
        //System.out.println("d: "+d+" maha "+mahalanobis(ret)+" lognormalizer "+lognormalizer+" v "+(-0.5 * mahalanobis(ret) + lognormalizer));
        /*if (Double.isNaN(d) || Double.isInfinite(d)) {
        //System.out.println("den "+d+" logdet "+logdet+ " lognorm "+lognormalizer +" maha "+mahalanobis(ret));
            d = Double.MIN_VALUE;
        }*/
        //assert(!Double.isNaN(d));
        return d;
    }

    /*public double boundedDensity(double[] input) {
        return density(input) * Math.sqrt(Math.pow(2 * Math.PI, input.length) * getDeterminant());
    }*/
    
    /*public double mutualDensity(MVN other) {
        return Math.min(boundedDensity(other.getMeans()),other.boundedDensity(getMeans()));        
    }*/
    private void readObject(
      ObjectInputStream aInputStream
    ) throws ClassNotFoundException, IOException {
        aInputStream.defaultReadObject();
    }
   
    private void writeObject(
      ObjectOutputStream aOutputStream
    ) throws IOException {
      aOutputStream.defaultWriteObject();
    }   

}
