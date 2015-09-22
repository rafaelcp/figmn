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
import java.util.Enumeration;
import java.util.concurrent.ForkJoinPool;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.DenseVector;
import no.uib.cipr.matrix.Matrices;
import no.uib.cipr.matrix.Vector;
import no.uib.cipr.matrix.sparse.LinkedSparseMatrix;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.misc.FIGMNClassifier;
import weka.core.AdditionalMeasureProducer;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

/**
 *
 * @author rafa
 */
public class FIGMN extends AbstractClassifier implements Serializable, AdditionalMeasureProducer {
    
    protected ArrayList<MVN> distributions;
    private double[][] sigmaIni;
    private double logDetSigmaIni;
    private boolean verbose = false;
    private boolean skipComponents = true;
    private boolean autoSigma = false;
    private double attrSelection = 1;
    protected double beta = Double.MIN_VALUE;
    protected double delta = 1;//0.3;
    private double prune = Double.POSITIVE_INFINITY;
    protected double chisq = Double.NaN;
    private int inputLength = -1;
    static final long serialVersionUID = 3932117032546553728L;
    int cores;
    public boolean useDiagonal = false;
    
    public FIGMN() {
        cores = Runtime.getRuntime().availableProcessors();
        distributions = new ArrayList<MVN>();
    }
    
    public FIGMN(double[][] data, double delta) {
        this();
        this.delta = delta;
        inputLength = data[0].length;
        updateSigmaIni(data.clone());
    }
    
    public FIGMN(double delta, double[][] sigmaIni) {
        this();
        this.delta = delta;
        this.sigmaIni = sigmaIni.clone();
    }
    
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

    /*public static double[][] knn(double[] x, double[][] data, int k) {
        return data;
    }*/
    
    protected void updateSigmaIni(double[] datum) {
        int D = datum.length;
        sigmaIni = Matrices.getArray(Matrices.identity(D).scale(delta));
        logDetSigmaIni = 0;
        for (int i = 0; i < sigmaIni.length; i++) {
            if (Double.isNaN(sigmaIni[i][i]) || sigmaIni[i][i] < Float.MIN_VALUE)
                sigmaIni[i][i] = Float.MIN_VALUE;
            logDetSigmaIni += Math.log(sigmaIni[i][i]);
            sigmaIni[i][i] = 1.0 / sigmaIni[i][i];
        }
        //output("sigmaini: "+Arrays.deepToString(sigmaIni));
    }
    
    protected void updateSigmaIni(double[][] data) {
        int N = data.length;
        if (N == 0) return;
        int D = data[0].length;
        //MultivariateSummaryStatistics mss = new MultivariateSummaryStatistics(D,true);
        double[] mean = new double[D];
        double[] counts = new double[D];
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < D; j++) {
                if (!Double.isNaN(data[i][j])) {
                    //System.out.println("Data: "+data[i][j]);
                    mean[j] += data[i][j];
                    counts[j]++;
                }
            }
            //ToDo: adicionar só os vizinhos mais próximos do dado em questão
            //System.out.println(Arrays.toString(data[i]));
            //mss.addValue(data[i]);
        }
        //System.out.println("Total: "+Arrays.toString(mean));
        for (int j = 0; j < D; j++) {
            mean[j] /= counts[j];
        }
        //double[] std = mss.getStandardDeviation();
        double[] var = new double[D];
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < D; j++) {
                if (!Double.isNaN(data[i][j])) {
                    var[j] += Math.pow(data[i][j] - mean[j],2) / (counts[j] - 1);
                }
            }
            //ToDo: adicionar só os vizinhos mais próximos do dado em questão
            //System.out.println(Arrays.toString(data[i]));
            //mss.addValue(data[i]);
        }
        for (int i = 0; i < var.length; i++) {
            //std[i] *= std[i];
            //System.out.println(Double.MIN_VALUE+" / "+(Double.MIN_VALUE / delta));
            if (var[i] < Double.MIN_VALUE)
                var[i] = max(var);
        }
        //System.out.println(Arrays.toString(std));
        sigmaIni = Matrices.getArray(FIGMN.diagonalMatrix(new DenseVector(var)).scale(delta));         
        //System.out.println("Antes: "+MatrixUtils.createDenseMatrix(sigmaIni));
        logDetSigmaIni = 0;
        for (int i = 0; i < sigmaIni.length; i++) {
            if (Double.isNaN(sigmaIni[i][i]) || sigmaIni[i][i] < Float.MIN_VALUE)
                sigmaIni[i][i] = Float.MIN_VALUE;
            //detSigmaIni *= sigmaIni[i][i];
            logDetSigmaIni += Math.log(sigmaIni[i][i]);
            sigmaIni[i][i] = 1.0 / sigmaIni[i][i];
        }
        output("SigmaIni: " + Arrays.deepToString(sigmaIni));
        //System.out.println("logdetsigmaini: "+logDetSigmaIni+" SigmaIni: "+MatrixUtils.createDenseMatrix(sigmaIni));
        
        //if (Double.isNaN(logDetSigmaIni) || Double.isInfinite(logDetSigmaIni)){// || detSigmaIni < Double.MIN_VALUE) {
        //    System.out.println("logdetsigmaini "+logDetSigmaIni);
            //    logDetSigmaIni = 0;
        //}
        //System.out.println("Counts: "+Arrays.toString(counts)+" Mean: "+Arrays.toString(mean)+" Var: "+Arrays.toString(var));
        //System.out.println("Depois: "+MatrixUtils.createDenseMatrix(sigmaIni));
        //ToDo: DEBUGAR!
        //System.out.println(MatrixUtils.createDenseMatrix(sigmaIni));
    }

    public void output(String text) {
        if (verbose) {
            System.out.println(text);
        }
    }
    
    public String deltaTipText() {
        return "Initial component size as a factor (0 and above) of the dataset standard deviations for each dimension.";
    }

    public void setDelta(double value) {
        delta = value;
    }
    
    public double getDelta() {
        return delta;
    }
    
    public double getPrune() {
        return prune;
    }
    
    public void setPrune(double value) {
        prune = value;
    }

    public boolean getSkipComponents() {
        return skipComponents;
    }
    
    public void setSkipComponents(boolean value) {
        skipComponents = value;
    }

    public boolean getAutoSigma() {
        return autoSigma;
    }
    
    public void setAutoSigma(boolean value) {
        autoSigma = value;
    }

    public double getAttrSelection() {
        return attrSelection;
    }
    
    public void setAttrSelection(double value) {
        attrSelection = value;
    }

    public String betaTipText() {
        return "Minimum activation (between 0 and 1) for updating component.";
    }
    
    public void setBeta(double value) {
        beta = value;
        chisq = Statistics.chi2inv(1-beta,inputLength);
    }
    
    public double getBeta() {
        return beta;
    }

    public String verboseTipText() {
        return "Whether to log each processing step or not.";
    }
    
    public void setVerbose(boolean value) {
        verbose = value;
    }
    
    public boolean getVerbose() {
        return verbose;
    }
    
    public ArrayList<MVN> getDistributions() {
        return distributions;
    }
    
    public void setSigmaIni(double[][] value) {
        sigmaIni = value.clone();
    }
    
    public void setSigmaIni(double[] value) {
        sigmaIni = Matrices.getArray(FIGMN.diagonalMatrix(new DenseVector(value)).scale(delta));
    }
    
    public void setSigmaIni(DenseVector value) {
        setSigmaIni(value.getData());
    }
    
    
/*    public double[] getBoundedLikelihoods(double[] x) {
        int s = distributions.size();
        double[] activations = new double[s];
        for (int i = 0; i < s; i++) {
            MVN mvn = distributions.get(i);
            activations[i] = mvn.boundedDensity(x);
        }
        return activations;
    }
*/    
    public double[] getLikelihoods(final double[] x) {
        //int s = distributions.size();
        //double[] activations = new double[s];
        //boolean fix = true;
        //double max = Double.NEGATIVE_INFINITY;
        //double total = 0;
        //final double[] x_ = x.clone();
        double[] activations = distributions.stream().mapToDouble(mvn -> mvn.logdensity(x)).toArray();
        final double max = Arrays.stream(activations).max().getAsDouble();
        /*for (int i = 0; i < s; i++) {
            activations[i] = distributions.get(i).logdensity(x);
            total += activations[i];
            if (activations[i] > max) {
                max = activations[i];
                //if (max > 1 * total / (i+1)) {
                //    break;
                //}
            }
        }*/
        activations = Arrays.stream(activations).map(a -> a - max).toArray();
        /*for (int i = 0; i < s; i++) {
            activations[i] -= max;
        }*/
        return activations;
    }

    public double[] getDistances(final double[] x) {
        //int s = distributions.size();
        //double[] activations = new double[s];
        double[] activations = distributions.stream().parallel().mapToDouble(mvn -> mvn.mahalanobis(x)).toArray();
        /*for (int i = 0; i < s; i++) {
            MVN mvn = distributions.get(i);
            activations[i] = mvn.mahalanobis(x.clone());
        }*/
        return activations;
    }

    
    public double[] getPosteriors(double[] x) {
        int s = distributions.size();
        if (s == 1) return new double[]{1};
        double[] activations = getLikelihoods(x.clone());
        //System.out.println("act: "+Arrays.toString(activations));
        //output("Likelihoods: "+Arrays.toString(activations));
        //System.out.println(Arrays.toString(activations));
        double[] posteriors = new double[s];
        double total = 0;
        for (int i = 0; i < s; i++) {
            posteriors[i] = Math.exp(activations[i] + Math.log(getPrior(i)));// + Double.MIN_VALUE;
            //assert(!Double.isNaN(posteriors[i]));
            total += posteriors[i];
        }
        //assert(!Double.isNaN(total));
        //assert(total > 0);
        //System.out.println("tot "+total);
        for (int i = 0; i < s; i++) {
            posteriors[i] = posteriors[i] / total;
            //posteriors[i] = Math.exp(posteriors[i] - total);
        }
        //output("Posteriors: "+Arrays.toString(posteriors));
        return posteriors;
    }

    public static double min(double[] x) {
        double m = Double.POSITIVE_INFINITY;
        for (double m_ : x) {
            if (m_ < m)
                m = m_;
        }
        return m;
    }
    
    public static double max(double[] x) {
        double m = Double.NEGATIVE_INFINITY;
        for (double m_ : x) {
            if (m_ > m)
                m = m_;
        }
        return m;
    }
    
    public static DenseMatrix diagonalMatrix(Vector v) {
        DenseMatrix m = new DenseMatrix(v.size(),v.size());
        for (int i = 0; i < v.size(); i++) {
            m.set(i, i, v.get(i));
        }
        return m;
    }    

    protected boolean testUpdate(double[] input) {
        return testUpdate(input, 0);
    }

    protected boolean testUpdate(double[] input, int numClasses) {
        //double[] input_ = input.clone();
        //input_[input_.length-1] = Double.NaN;
        if (Double.isNaN(chisq)) {
            if (beta == 0)
                chisq = Double.POSITIVE_INFINITY;
            else
                chisq = Statistics.chi2inv(1-beta,input.length);
        }
        int s = distributions.size();
        for (int i = 0; i < s; i++) {
            MVN mvn = distributions.get(i);
            if (mvn.mahalanobis(input.clone()) < chisq) return true;
        }
        return false;
        
        //double[] activations = getDistances(input.clone());
        //double m = min(activations);
        /*if (numClasses > 0) {
            int best = FIGMNClassifier.argmin(activations);
            int bestClass = FIGMNClassifier.argmax(distributions.get(best).getMeans(), input.length - numClasses, input.length);
            int currClass = FIGMNClassifier.argmax(input, input.length - numClasses, input.length);
            //System.out.println("CORRECT: "+bestClass+" CURRENT: "+currClass);
            if (bestClass != currClass) return false;
            //else return true;
        }*/
        //if (m > 3)
        //System.out.println("Activation: "+m+" Beta: "+beta+" ChiSq: "+chisq+" D: "+input.length+" Activations: "+Arrays.toString(activations));
        //MVN mvn = new MVN(input,MatrixUtils.createDenseMatrix(sigmaIni),detSigmaIni);
        //int nearest = new DenseVector(activations).getMinIndex();
        //double m2 = distributions.get(nearest).mahalanobis(input);
        //return m < chisq;// && m2 < chisq;
    }

    public void learn(final double[] input) throws Exception {
        learn(input, 0);
    }
    
    public void learn(final double[] input, int numClasses) throws Exception {
        int s = distributions.size();
        if (s > 0 && beta < 1 && (beta == 0 || testUpdate(input.clone(), numClasses))) {
            //System.out.println("UPDATE");
            output(totalCount()+" Updating...");
            //System.out.println("UPDATE");
            updateComponents(input.clone());
        }
        else {
            //System.out.println("ADD");
            output("Adding... "+(s+1));
            addComponent(input.clone());
        }
        if (!Double.isInfinite(prune)) {
            output("Pruning...");
            pruneComponents();
        }
    }
    
    public void learn(double[][] inputs) throws Exception {
        for (int i = 0; i < inputs.length; i++) {
            //System.out.println("#"+i);
            output("Learning datum #" + i);
            //output("Input: "+Arrays.toString(inputs[i]));
            learn(inputs[i]);
        }
    }

    public void updateComponents(double[] x) {
        int s = distributions.size();
        double meanP = 1.0 / s;
        double[] activations = getPosteriors(x.clone());
        double highP = max(activations);
        //meanP = highP;
        //System.out.println("Max. Activation: "+highP+" Mean Activation: "+meanP);
        
        for (int i = s - 1; i >= 0; i--) {
            //System.out.println(":"+i+" -> "+distributions.get(i).age);
            if (!skipComponents || activations[i] >= meanP || activations[i] >= highP) {
                updateComponent(i, x, 1);//activations[i]);
                break;
            }
        }        
    }
    
    public static DenseVector subVector(DenseVector v, int[] indices) {
        return (DenseVector) Matrices.getSubVector(v, indices);        
    }
    
    public void updateComponent(int i, final double[] x, final double activation) {
        MVN mvn = distributions.get(i);
        DenseVector mean = mvn.mean;
        //System.out.println("input "+Arrays.toString(x)+" mean "+mean);
        int D = mean.size();
        DenseVector input = new DenseVector(x);
        mvn.count += activation;
        //System.out.println("COUNT "+mvn.count);
        double learningRate = activation / mvn.count;
        DenseVector diff = (DenseVector) input.copy().add(-1, mean);
        diff.scale(learningRate);
        
        //New mean
        mean.add(diff);
        input.add(-1, mean);
        DenseMatrix invCov = new DenseMatrix(mvn.invCov);
        DenseVector newdiag = null;
        double logdet = 0;
        
            if (useDiagonal) {
                //newdiag = mvn.diag.scale(1 - learningRate).add(diff.ebeMultiply(diff).scale(learningRate));
            }
            else {
                invCov.scale(1.0 / (1 - learningRate));
                input.scale(Math.sqrt(learningRate));
                //System.out.println("invcov0 "+invCov+" input "+input+" diff "+diff);
                logdet = rankOneMatrixUpdate(invCov, input, D*Math.log1p(-learningRate) + mvn.logdet);
                //System.out.println("invcov1 "+invCov);
                logdet = rankOneMatrixDowndate(invCov,diff,logdet);
                //System.out.println("invcov2 "+invCov);
            }
        try { //ToDo: TEM QUE VER ISSO AÍ!
            if (useDiagonal) {
                double d = 1;
                for (int j = 0; j < newdiag.size(); j++) {
                    d *= newdiag.get(j);
                }
            }
            else {
                if (Double.isInfinite(logdet)) logdet = Double.MAX_VALUE;
                //mvn.mean = mean;    //Se der erro no determinante, a média já foi atualizada pelo menos
                mvn.invCov = invCov;
                mvn.age++;
                mvn.features = null;
                if (Double.isNaN(logdet) || Double.isInfinite(logdet)) {
                    //return;
                    throw(new Exception("Invalid log determinant: "+logdet));
                }
                mvn.logdet = logdet;
                mvn.setLogNormalizer();
            }

        }
        catch (Exception e) {
            System.out.println(e);
            if (verbose) {
                Logger.getLogger(FIGMN.class.getName()).log(Level.WARNING, "Precision Matrix Error: {0}", new Object[]{e.getMessage()});
                e.printStackTrace();
            }
        }
    }
    
    /*
    public Matrix rankKMatrixUpdate(DenseMatrix M, DenseMatrix U, DenseMatrix V, double det) {
        DenseMatrix temp1 = (Matrices.identity(U.numColumns()).add(V.mult(M).mult(U)));
        CholeskyDecomposition lutemp = new CholeskyDecomposition(temp1);
        double newdet = Math.log(lutemp.getDeterminant()) + det;
        DenseMatrix itemp1 = lutemp.getSolver().getInverse();
        DenseMatrix newm = M.subtract(M.multiply(U).multiply(itemp1).multiply(V).multiply(M));
        return new Matrix(newm,newdet);
    }
*/
    
    public static DenseMatrix outerProduct(DenseVector v1, DenseVector v2) {
        DenseMatrix m = new DenseMatrix(v1.size(),v1.size());
        for (int i = 0; i < v1.size(); i++) {
            for (int j = 0; j < v2.size(); j++) {
                m.set(i, j, v1.get(i) * v2.get(j));                
            }            
        }
        return m;
    }
    
    public double rankOneMatrixUpdate(DenseMatrix m, DenseVector v, double det) {
        DenseVector temp1 = new DenseVector(v.size());// = m.preMultiply(v);
        m.mult(v, temp1);
        double temp2 = 1 + temp1.dot(v);
        //System.out.println("update: "+temp2);
        double newdet = det;
        if (temp2 > 0) {
            newdet += Math.log(temp2);
        }
        else {
            //System.out.println("INVALID update temp2 "+det);
        }
        m.add(-1,FIGMN.outerProduct(temp1, temp1).scale(1.0 / temp2));        
        return newdet;
    }
    
    public double rankOneMatrixDowndate(DenseMatrix m, final DenseVector v, double det) {
        DenseVector temp1 = new DenseVector(v.size());
        m.mult(v, temp1);
        double temp2 = 1 - temp1.dot(v);
        //System.out.println("downdate: "+temp2);
        double newdet = det;
        if (temp2 > 0) {
            newdet += Math.log(temp2);
        }
        else {
            //System.out.println("INVALID downdate temp2 "+det);
        }        
        m.add(FIGMN.outerProduct(temp1,temp1).scale(1.0 / temp2));
        return newdet;
    }

    public int getNumComponents() {
        return distributions.size();
    }
    
    protected void clearComponents() {
        //distributions = new ArrayList<MVN>();
        distributions.clear();
    }
    
    public double totalCount() {
        double c = 0;
        for (MVN distribution : distributions) {
            c += distribution.count;
        }
        return c;
        //return distributions.stream().collect(Collectors.summingDouble(mvn -> mvn.count));
    }
    
    public double totalAge() {
        double c = 0;
        for (MVN distribution : distributions) {
            c += distribution.age;
        }
        return c;
        //return distributions.stream().collect(Collectors.summingDouble(mvn -> mvn.age));
    }

    public double meanAge() {
        return totalAge() / distributions.size();
    }
 
    public double getPrior(int i) {
        return distributions.get(i).count / totalCount();
    }
    
    public double meanCount() {
        return totalCount() / distributions.size();
    }
    
    public double varCount() {
        int d = 0;
        int K = distributions.size();
        double m = meanCount();
        for (int i = 0; i < K; i++) {
            d += Math.pow(distributions.get(i).count - m, 2);
        }
        return d / K;
    }
    
    public double stdCount() {
        return Math.sqrt(varCount());
    }

    public void pruneComponents() {
        if (Double.isInfinite(prune)) return;
        int K = distributions.size();
        //int D = distributions.get(0).size();
        double m = meanCount();
        double am = meanAge();
        //System.out.println("mean count "+m+" mean age "+am);
        //double s = stdCount();
        //System.out.println("Total Counter: "+totalCount()+ " Mean Counter: "+m+" Std Counter: "+s);
//        for (int i = (int)(K / 2); i >= 0; i--) {
        for (int i = K-1; i >= 0; i--) {
            MVN mvn = distributions.get(i);
            double c = mvn.count;
            double a = mvn.age;
//            if (c == 1 && c < m - prune * s) {
            if (c < m && a > am) {
                verbose = true;
                output("Pruning component #" + i+" with counter "+c+" and age "+a+" (mean count: "+m+" mean age: "+am+")");
                verbose = false;
                distributions.remove(i);
            }
        }
    }

    //TODO: JavaDocs
    /** recall
     * 
     * @param input
     * @return Input reconstruction.
     * @throws Exception 
     */
    public double[] recall(double[] input) throws Exception {
        int s = distributions.size();
        double[] p = getPosteriors(input.clone());
        double meanP = 1.0 / s;
        double highP = max(p);
        //meanP = highP;
        //System.out.println("p: "+Arrays.toString(p));
                //System.out.println("Input: "+Arrays.toString(input)+ "Posteriors: "+Arrays.toString(p)+" mean: "+meanP+" highP: "+highP);
        DenseVector r = new DenseVector(new double[input.length]);
        double total = 0;
        for (int i = 0; i < s; i++) {
            if (!skipComponents || p[i] >= meanP || p[i] >= highP) {
                r.add(p[i], new DenseVector(recallFor(i,input)));
                total += p[i];
                //break;
            }
            //else
        }
        //System.out.println(highP+" > "+(meanP));
        //System.out.println(r);
        //System.out.println(":"+r);
        return r.scale(1/total).getData();
    }
    
    public static Vector subVector(final Vector v, int[] indices) {
        return Matrices.getSubVector(v, indices);
        /*
        DenseVector r = new DenseVector(indices.length);
        for (int i = 0; i < indices.length; i++) {
            r.set(i, v.get(indices[i]));
        }
        return r;//MatrixUtils.createColumnRealMatrix(v.toArray()).getSubMatrix(indices, new int[1]).getColumnVector(0);        
        */
    }
    
    public static Vector subVector(final Vector v, int start, int len) {
        /*int[] idx = new int[len];
        for (int i = 0; i < len; i++) {
            idx[i] = start + i;            
        }*/
        return Matrices.getSubVector(v, Matrices.index(start, start+len));
    }
        
    public static Vector setSubVector(final Vector v, int start, final Vector v2) {
        Vector vnew = v;
        for (int i = 0; i < v2.size(); i++) {
            vnew.set(start + i, v2.get(i));
        }
        return vnew;
    }
    
    protected double[] recallFor(int i, final double[] input) {
        int l = input.length;
        double[] inp = input.clone();
        MVN mvn = distributions.get(i);
        int noutputs = 0;
        for (int j = 0; j < l; j++) {
            if (Double.isNaN(inp[j])) {
                inp[j] = mvn.getMeans()[j];
                noutputs++;
            }
        }
        int[] outputs = new int[noutputs];
        int k = 0;
        for (int j = 0; j < l; j++) {
            if (Double.isNaN(input[j])) {
                outputs[k] = j;
                k++;
            }
        }

        DenseVector m = mvn.getMeansVector();
        //System.out.println(":"+m);
        if (useDiagonal || mvn.count == 1) {
            return m.getData();
        }
        DenseMatrix ci = new DenseMatrix(mvn.invCov);
        DenseVector a = new DenseVector(inp);
        int aStart = 0, aLength = l-noutputs, aEnd = l-1-noutputs;
        int bStart = l-noutputs, bLength = noutputs, bEnd = l-1;
        DenseVector ma;
        DenseMatrix cba;
        if (attrSelection < 1) {
            mvn.features = null;
            int[] r = mvn.getOrderedFeatures(outputs, attrSelection);
            System.out.println("Feature Selection: "+Arrays.toString(r)+" for "+Arrays.toString(outputs)+" Total: "+r.length);            
            a = new DenseVector(Matrices.getSubVector(a, r));
            ma = new DenseVector (Matrices.getSubVector(m, r));
            cba = new DenseMatrix(Matrices.getSubMatrix(ci, Matrices.index(bStart, bStart+bLength), r));
        }
        else {
            a = new DenseVector (FIGMN.subVector(a, aStart, aLength));//a.getSubVector(aStart, aLength);
            ma = new DenseVector (FIGMN.subVector(m, aStart, aLength));//m.getSubVector(aStart, aLength);
            cba = new DenseMatrix (Matrices.getSubMatrix(ci, Matrices.index(bStart, bStart+bLength), Matrices.index(aStart, aStart+aLength)));
        }

        DenseVector mb = new DenseVector(Matrices.getSubVector(m, Matrices.index(bStart, bStart+bLength)));
        a.add(-1,ma);
        //System.out.println("a "+a.size()+" ma "+ma.size()+" cba "+cba.getRowDimension()+"x"+cba.getColumnDimension());
//System.out.println("("+ci+")"+bStart+" / "+bEnd+" / "+bLength+" / "+Arrays.toString(Matrices.index(bStart, bEnd+1)));
        DenseMatrix cb = new DenseMatrix(Matrices.getSubMatrix(ci, Matrices.index(bStart, bEnd+1), Matrices.index(bStart, bEnd+1)));
        DenseMatrix I = (DenseMatrix)(Matrices.identity(cb.numColumns()));
        DenseMatrix cbinv = I;
        cb.solve(I, cbinv);
        DenseMatrix cbinvcba = new DenseMatrix(mb.size(),a.size());
//        System.out.println(I+"\n:"+cba+"\n"+cbinvcba);
        cbinv.mult(-1, cba, cbinvcba);

        DenseVector temp = new DenseVector(bLength);
        //System.out.println(cbinvcba+":"+diff+"\n:"+temp);
        cbinvcba.mult(a, temp);
        mb.add(temp);
        //m.setEntry(l-1, xb.get(0));
        
        //System.out.println("m: "+m);
        m = new DenseVector(setSubVector(m, bStart, mb));
        //System.out.println("m: "+m);
        return m.getData();
    }
    
/*    public double[] calcDists(double[] x) {
        double[] dists = new double[distributions.size()];
        for (int i = 0; i < distributions.size(); i++) {
            
        }
        return dists;
    }
  */  
    public double[] nearestDistance(double[] x) {
        double ndist = Double.MAX_VALUE, dist;
        double[] ndists = new double[x.length], dists = new double[x.length];
        for (MVN distribution : distributions) {
            dist = 0;
            for (int j = 0; j < distribution.getMeans().length; j++) {
                dists[j] = Math.pow(distribution.getMeans()[j] - x[j], 2);
                if (dists[j] <= sigmaIni[j][j]) dists[j] = sigmaIni[j][j];            
                dist += dists[j];
                if (dist >= ndist) break;
            }
            if (dist < ndist) {
                ndist = dist;
                ndists = dists.clone();
            }
        }
        return ndists;
    }
    
    public double[][] calcSigmaIni(double[] x) {
        int s = distributions.size();
        double invP = 1.0 / s;
        int l = x.length;
        double totalCount = 0;
        if (s > 0) {
            DenseMatrix sig = new DenseMatrix(l,l);
            for (int i = 0; i < s; i++) {
                MVN mvn = distributions.get(i);
                double c = mvn.count;
                if (c < invP) continue;
                sig.add(c, mvn.invCov);
                totalCount += c;
            }
            sig.scale(1.0 / totalCount);
            //double det = 1;
            for (int i = 0; i < l; i++) {
                //det *= sigmaIni[i][i];
                sigmaIni[i][i] = sig.get(i, i);
            }
            //return MatrixUtils.createRealDiagonalMatrix(nearestDistance(x)).scale(1).getData();
        }
        return sigmaIni;
    }

    public double[][] calcSigmaIniDiag(double[] x) {
        int s = distributions.size();
        double invP = 1.0 / s;
        int l = x.length;
        double totalCount = 0;
        if (s > 0) {
            DenseVector sig = new DenseVector(new double[l]);
            for (int i = 0; i < s; i++) {
                MVN mvn = distributions.get(i);
                double c = mvn.count;
                if (c < invP) continue;
                sig.add(c, mvn.diag);
                totalCount += c;
            }
            sig.scale(1.0 / totalCount);
            //double det = 1;
            for (int i = 0; i < l; i++) {
                //det *= sigmaIni[i][i];
                sigmaIni[i][i] = sig.get(i);
            }
            //return MatrixUtils.createRealDiagonalMatrix(nearestDistance(x)).scale(1).getData();
        }
        return sigmaIni;
    }
    
    public double calcDetSigmaIni(double[][] si) {
        double d = 0;
        double log1 = Math.log(1.0);
        for (int i = 0; i < si.length; i++) {
            d +=  (log1 - Math.log(si[i][i]));
        }
        return d;
    }
    
    public void addComponent(double[] mean) throws Exception {
        //System.out.println(Arrays.toString(mean));
        //System.out.println(Arrays.deepToString(sigmaIni));
        //System.out.println("adding "+(distributions.size()+1));
        MVN newComp;
        if (autoSigma) {
            double[][] si = calcSigmaIni(mean.clone());
            //System.out.println("calcdetsigmaini "+calcDetSigmaIni(si));
            newComp = new MVN(mean, new DenseMatrix(mean.length, mean.length), calcDetSigmaIni(si));
        }
        else {
            //System.out.println(MatrixUtils.createDenseMatrix(sigmaIni));
            //System.out.println("logdetsigmaini "+logDetSigmaIni+" exp "+Math.exp(logDetSigmaIni));
            if (useDiagonal) {
                newComp = new MVN(mean, new DenseMatrix(sigmaIni), logDetSigmaIni, true);            
            }
            else {                
                newComp = new MVN(mean, new DenseMatrix(sigmaIni), logDetSigmaIni);
            }
        }
        /*int s = distributions.size();
        double[] activations = getPosteriors(mean);
        double maxP = max(activations) * s / (s + 1);
        if (maxP > 0.5) {
            beta *= 0.9;
        System.out.println("MAXP "+maxP+" Beta "+beta);
        }*/
        //double selfActivation = activations[];
        //System.out.println("SA "+selfActivation);
        //System.out.println(":"+newComp);
        newComp.attrselection = this.attrSelection;
        distributions.add(newComp);
    }
    
    public static double[][] instancesToDoubleArrays(Instances data) {
        int D = data.numAttributes(), N = data.size(), C = data.numClasses();
        /*if (data.classAttribute().isNominal()) {
            D = D + C - 1;
        }*/
        double [][] result = new double[N][D];
        for (int i = 0; i < N; i++) {
            result[i] = FIGMNClassifier.instanceToArray(data.get(i));
        }
        return result;
    }
    

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        DataSource source;
        try {
            source = new DataSource("/home/rafa/Downloads/weka-3-7-10/data/iris.arff");
            Instances dt = source.getDataSet();
            //dt.randomize(new Random());
            if (dt.classIndex() == -1)
               dt.setClassIndex(dt.numAttributes() - 1);
            double[][] data;
            data = FIGMN.instancesToDoubleArrays(dt);
            FIGMN igmn = new FIGMN(data,0.1);
            igmn.verbose = false;
            igmn.learn(data);
            System.out.println("# of Components: ["+igmn.distributions.size() + "] List: " + igmn.distributions);
            int c = dt.classIndex();
            double err = 0;
            int errcount = 0;
            for (int i = 0; i < data.length; i++) {
                double[] input = data[i].clone();//IGMN.doubleArrayToWrapper(data[i]);
                input[c] = Double.NaN;
                double[] result = igmn.recall(input);
                double err_ = Math.pow(result[c] - data[i][c],2);
                System.out.println("#"+i+" Target: " + data[i][c] + " Output: " + result[c] + " Error: " + err_ + " Reconstruction: " + Arrays.toString(result));
                err += err_;
                if (err_ != 0) errcount++;
            }
            err /= data.length;
            System.out.println("MSE: " + err + " Errors: " + errcount + " Accuracy: " + (data.length - errcount)/(double)data.length);
        } catch (Exception ex) {
            Logger.getLogger(FIGMN.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    
    public int numberOfClusters() {
        return distributions.size();
    }
        
    public int clusterInstance(double[] ins) throws Exception {
        double dens;
        double maxDens = 0;
        int best = -1;
        for (int i = 0; i < numberOfClusters(); i++) {
            dens = distributions.get(i).density(ins);
            if (dens > maxDens) {
                maxDens = dens;
                best = i;
            }
        }
        return best;
    }

    public int clusterInstance(Instance ins) throws Exception {
        return clusterInstance(ins.toDoubleArray());
    }
    
    public int clusterInstance(Double[] ins) throws Exception {
        return clusterInstance(FIGMN.DoubleArrayToPrimive(ins));
    }

    public double[] distributionForInstance(double[] ins) throws Exception {
        double[] distribution = new double[numberOfClusters()];
        for (int i = 0; i < numberOfClusters(); i++) {
            distribution[i] = distributions.get(i).density(ins);
        }
        return distribution;
    }
    
    public static double[] DoubleArrayToPrimive(Double[] d) {
        double[] ret = new double[d.length];
        for (int i = 0; i < d.length; i++) {
            ret[i] = d[i];
        }
        return ret;
    }

    public static Double[] doubleArrayToWrapper(double[] d) {
        Double[] ret = new Double[d.length];
        for (int i = 0; i < d.length; i++) {
            ret[i] = d[i];
        }
        return ret;
    }

    public double[] distributionForInstance(Double[] ins) throws Exception {
        return distributionForInstance(FIGMN.DoubleArrayToPrimive(ins));
    }

    public double[] distributionForInstance(Instance ins) throws Exception {
        return distributionForInstance(ins.toDoubleArray());
    }
    
    @Override
    public String toString() {
        return "Delta: " + delta + " Beta: " + beta + " #Clusters: " + distributions.size() + "\n" + distributions.toString();
    }

    @Override
    public void buildClassifier(Instances i) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public Enumeration<String> enumerateMeasures() {
        java.util.Vector<String> v = new java.util.Vector<String>();
        v.add("measureNumClusters");
        return v.elements();
    }

    @Override
    public double getMeasure(String measureName) {
        if (measureName == "measureNumClusters") return this.numberOfClusters();
        return Double.NaN;
    }
    

}
