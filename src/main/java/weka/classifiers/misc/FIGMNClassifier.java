/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
//C:\Program Files\Weka-3-7>java -Xmx8G -cp weka.jar;C:\Users\Rafael\Dropbox\NetBeansProjects\FIGMN\dist\FIGMN.jar weka.classifiers.misc.FIGMNClassifier -t "C:\Program Files\Weka-3-7\data\iris.arff" -x 150 -D 2 -B 0.2 -S -AS 1
package weka.classifiers.misc;

import figmn.FIGMN;
import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.Vector;
import java.util.concurrent.ForkJoinPool;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.UpdateableClassifier;
import weka.classifiers.bayes.NaiveBayesUpdateable;
import weka.classifiers.functions.SGD;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.CapabilitiesHandler;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.SelectedTag;
import weka.core.TechnicalInformation;
import weka.core.Utils;
import weka.core.converters.ArffLoader;

/**
 *
 * @author rafa
 */
public class FIGMNClassifier extends FIGMN implements UpdateableClassifier, CapabilitiesHandler, Serializable, OptionHandler {

    static final long serialVersionUID = 3932117032546553738L;
    protected boolean autoDelta = false;

    public FIGMNClassifier(double[][] data, double delta) {
        super(data, delta);
    }

    public FIGMNClassifier() {
        super();
    }
    
    public static int[] nominalToBinary(int value, int max) {
        int[] out = new int[max];
        out[value] = 1;
        return out;
    }
    
    public static Instances fillMissing(Instances ins) {
        ins = new Instances(ins);
        int D = ins.numAttributes(), N = ins.size();
        for (int i = 0; i < N; i++) {
            ins.set(i, fillMissing(ins.get(i)));
        }
        return ins;
    }

    public static Instance fillMissing(Instance ins) {
        int D = ins.numAttributes();
        for (int j = 0; j < D; j++) {
            if (ins.isMissing(j) && j != ins.classIndex()) {
                //System.out.println(j+" Replace Missing by: "+ins.meanOrMode(j)+" Instance: "+ins.get(i));
                ins.setValue(j, ins.dataset().meanOrMode(j));
            }
        }
        return ins;
    }

    @Override
    public void buildClassifier(Instances i) throws Exception {
        output("Building model from " + i.numInstances() + " instances...");
        i = new Instances(i);
        getCapabilities().testWithFail(i);
        i = fillMissing(i);
        //System.out.println(i);
        if (getAutoDelta() && i.classAttribute().isNominal()) {
            delta = 3.0 / i.numClasses();
        }
        double[][] data = FIGMN.instancesToDoubleArrays(i);
        //System.out.println(Arrays.deepToString(data));
        clearComponents();
        updateSigmaIni(data);
        chisq = Double.NaN;
        ForkJoinPool commonPool = ForkJoinPool.commonPool();
        System.out.println("Number of Threads: "+commonPool.getParallelism());       
        learn(data);
        System.out.println(this+" # of Instances: "+i.size()+", # of Attributes: "+i.numAttributes()+", # of Classes: "+i.numClasses());
        output("Components: "+getDistributions().toString());
        //System.out.println("Components: "+getDistributions().toString());
    }
    
    public void setAutoDelta(boolean value) {
        autoDelta = value;
    }
    
    public boolean getAutoDelta() {
        return autoDelta;
    }
    
    public void initializeSigmaIni(Instances ins) {
        double[] sini = new double[ins.numAttributes()];
        for (int i = 0; i < sini.length; i++) {
            sini[i] = ins.variance(i);
        }
        
        setSigmaIni(sini);
    }
    
    public double[] toBinaryClass(double[] datum, int numClasses) {
        int d = datum.length + numClasses - 1;
        double[] r = new double[d];
        int[] binClass = nominalToBinary((int) datum[datum.length - 1], numClasses);
        for (int i = 0; i < datum.length - 1; i++) {
            r[i] = datum[i];
        }
        for (int i = 0; i < numClasses; i++) {
            r[datum.length - 1 + i] = binClass[i];
        }
        return r;
    }

    @Override
    public double classifyInstance(Instance instnc) throws Exception {
        //System.out.println("CHAMOU O CLASSIFY");
        double[] reconstruction = recall(instanceToArray(instnc));       
        //return Math.round(reconstruction[instnc.classIndex()]);
        return argmax(reconstruction,instnc.numAttributes()-1,instnc.numAttributes()-1+instnc.numClasses());
    }
    
    public static double[] instanceToArray(Instance ins) {
        double[] input;
        if (ins.classAttribute().isNominal()) {
            input = new double[ins.numAttributes() + ins.numClasses() - 1];
            //System.out.println("" + input.length + " / " + ins.numAttributes() + " / " + ins.numClasses());
            for (int i = 0; i < ins.numAttributes() - 1; i++) {
                input[i] = ins.value(i);
            }
            for (int i = 0; i < ins.numClasses(); i++) {
                if (ins.classIsMissing()) {
                    input[i + ins.numAttributes() - 1] = Double.NaN;
                }
                else if (ins.classValue() == i) {
                    input[i + ins.numAttributes() - 1] = 1;                    
                }
                else {
                    input[i + ins.numAttributes() - 1] = 0;
                }
            }
        }
        else {
            input = ins.toDoubleArray();
        }
        //System.out.println(""+input[0]+" , "+input[1]+" , "+input[2]+" , "+input[3]+" , "+input[4]+" , "+input[5]+" , "+input[6]);
        return input;
    }
    
    public static int argmax(double[] arr) {
        return argmax(arr, 0, arr.length);
    }
    public static int argmax(double[] arr, int from, int to) {
        int argmax = from;
        for (int i = from + 1; i <= to && i < arr.length; i++) {
            if (arr[i] > arr[argmax]) {
                argmax = i;
            }
        }
        return argmax - from;
    }

    public static int argmin(double[] arr) {
        return argmin(arr, 0, arr.length);
    }
    public static int argmin(double[] arr, int from, int to) {
        int argmin = from;
        for (int i = from + 1; i <= to && i < arr.length; i++) {
            if (arr[i] < arr[argmin]) {
                argmin = i;
            }
        }
        return argmin - from;
    }
    
    @Override
    public double[] distributionForInstance(Instance ins) throws Exception {
        //ins = fillMissing((Instance)ins.copy());
        //System.out.println("Recall: "+ins);
        //System.out.println("argmax "+argmax(new double[]{0.1,0.2,0.3,0.4,0.5,0.8,9.0},4,5));
        //System.out.println(ins+":"+Arrays.toString(instanceToArray(ins)));
        double[] dist = new double[ins.numClasses()];
        final double[] input = instanceToArray(ins);//        double[] r = new double[d];
        //input[ins.classIndex()] = Double.NaN;
        double[] reconstruction = recall(input);
        if (ins.classAttribute().isNominal()) {
            double sum = 0;
            for (int i = 0; i < dist.length; i++) {
                dist[i] = reconstruction[ins.numAttributes() - 1 + i];//0;//1.0/dist.length;
                if (dist[i] < 0) dist[i] = 0;
                if (dist[i] > 1) dist[i] = 1;
                sum += dist[i];
            }
            for (int i = 0; i < dist.length; i++) {
                dist[i] /= sum;
            }
            //System.out.println("ClassIndex: "+ins.classIndex()+" distsize: "+dist.length + " recsize: "+reconstruction.length);
            //System.out.println(Arrays.toString(reconstruction)+":"+argmax(reconstruction,ins.numAttributes(),ins.numAttributes() + ins.numClasses()));
            //int c = (int)Math.round(reconstruction[ins.classIndex()]);
            //if (c > ins.numClasses()-1) c = ins.numClasses()-1;
            //if (c < 0) c = 0;
            //dist[c] = 1;
        }
        else {
            dist[0] = reconstruction[ins.classIndex()];
        }
        output("Instance: "+ins);
        output("Input: "+Arrays.toString(input));
        output("Reconstruction: "+Arrays.toString(reconstruction));
        output("Distribution: " + Arrays.toString(dist));
        return dist;
    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities result = new Capabilities(this);
        // attributes
        result.disableAll();
        result.enable(Capability.BINARY_ATTRIBUTES);
        result.enable(Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capability.MISSING_VALUES);
        // class
        result.enable(Capability.BINARY_CLASS);
        result.enable(Capability.NOMINAL_CLASS);
        result.enable(Capability.NUMERIC_CLASS);
        result.enable(Capability.MISSING_CLASS_VALUES);
        result.setMinimumNumberInstances(0);
        return result;
    }

/*    @Override
    protected boolean testUpdate(double[] input) {
        if (classAttribute.isNominal()) {
            return input[classAttribute.index()] ==
        }
        else {
            return super.testUpdate(input);
        }
    }*/

    public static void main(String[] args) {
        if (args.length == 0) {
            args = "-t /home/rafa/Downloads/weka-3-7-10/data/iris.arff -x 150".split("\\s");
        }
        int N = 1;
        for (int i = 0; i < N; i++) {
            //FIGMNClassifier igmn = new FIGMNClassifier();
            //igmn.setBeta(Double.MIN_VALUE);
            //igmn.setDelta(1e-9);
            //igmn.setSkipUpdates(true);
            //System.out.println(Arrays.toString(args));
            //System.out.println("Starting: "+igmn);
            //runClassifier(igmn, args.clone());
            //System.out.println("Components: " +igmn.getDistributions());
            
            ArffLoader loader = new ArffLoader();
            Instances structure = null;
            try {
                loader.setFile(new File("C:\\Users\\Rafael\\mnist_train.arff"));
                //loader.setFile(new File("C:\\Program Files\\Weka-3-7\\data\\iris.arff"));
                
                structure = loader.getStructure();
                FIGMNClassifier classifier = new FIGMNClassifier();
                structure.setClassIndex(structure.numAttributes() - 1); 
                classifier.buildClassifier(structure);
                //classifier.setPrune(3);
                classifier.setBeta(0.001);
                classifier.setDelta(10000);
                classifier.setAttrSelection(1);
                
                //classifier.setVerbose(true);
                //classifier.setBeta(0);
                Instance current;
                int j = 0;
                while ((current = loader.getNextInstance(structure)) != null) {
                    j++;
                    if (j % 100 == 0) {
                        System.out.println("Learning instance "+j+" Num. Clusters: "+classifier.getNumComponents());
                        //int[] f = classifier.distributions.get(0).getOrderedFeatures(new int[]{784,785,786,787,788,789,790,791,792,793}, 0.5);
                        //int[] f = classifier.distributions.get(0).getOrderedFeatures(new int[]{}, 1);
                        //System.out.println("Features: "+Arrays.toString(f)+" Total: "+f.length);
                    }
                    ((UpdateableClassifier)classifier).updateClassifier(current);            
                                    //System.out.println(classifier.distributions.get(0).invCov);
                    //if (j > 1000) break;
                    /*if (j % 60000 == 0 && j < 200000) {
                        loader.reset();
                        loader.setFile(new File("C:\\Users\\Rafael\\mnist_train_pca40.arff"));
                        structure = loader.getStructure();
                        structure.setClassIndex(structure.numAttributes() - 1); 
                    }*/
                }
                
                //for (int k = 0; k < classifier.distributions.size(); k++) {
                //    System.out.println("invcov"+k+": "+classifier.distributions.get(k).invCov);
                //}

                System.out.println("Final # of clusters: "+classifier.getNumComponents());
                loader.setFile(new File("C:\\Users\\Rafael\\mnist_test.arff"));
                //loader.setFile(new File("C:\\Program Files\\Weka-3-7\\data\\iris.arff"));
                structure = loader.getStructure();
                structure.setClassIndex(structure.numAttributes() - 1);
                int correct = 0;
                int total = 10000;//structure.numInstances();
                j = 0;
                while ((current = loader.getNextInstance(structure)) != null) {
                    double target = current.classValue();
                    current.setClassMissing();
                    double c = classifier.classifyInstance(current);
                    if (c == target) {
                        //System.out.println("ACERTOU");
                        correct++;
                    }
                    j++;
                    if (j % 1 == 0) {
                        System.out.println("Testing instance "+j+" with class "+target+" and output was "+c+" Accuracy: "+(correct/(double)j*100)+"% Max. Accuracy: "+((total-(j-correct))/(double)total*100));
                    }
                                        //if (j > 1000) break;
                }
                System.out.println(classifier.distributions);
                System.out.println("Accuracy: "+(correct/(double)j*100)+"%");
                
            } catch (IOException ex) {
                System.out.println(ex);
                Logger.getLogger(FIGMNClassifier.class.getName()).log(Level.SEVERE, null, ex);
            } catch (Exception ex) {
                System.out.println(": "+ex);
                Logger.getLogger(FIGMNClassifier.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
    }

    public String globalInfo() {

        return  "FIGMN Classifier."
          + getTechnicalInformation().toString();
    }

    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation 	result;

        result = new TechnicalInformation(TechnicalInformation.Type.ARTICLE);
        result.setValue(TechnicalInformation.Field.AUTHOR, "Rafael C. Pinto");
        result.setValue(TechnicalInformation.Field.YEAR, "2014-2015");
        result.setValue(TechnicalInformation.Field.TITLE, "Fast IGMN");
        //result.setValue(Field.JOURNAL, "Machine Learning");
        //result.setValue(Field.VOLUME, "6");
        //result.setValue(Field.PAGES, "37-66");

        return result;
    }

    @Override
   public Enumeration listOptions() {

    Vector newVector = new Vector(8);

    newVector.addElement(new Option(
	      "\tVerbose (for debug purposes)",
	      "V", 0, "-V"));
    newVector.addElement(new Option(
	      "\tDelta: Initial size of components.",
	      "D", 1,"-D <proportion of dataset standard deviations>"));
    newVector.addElement(new Option(
	      "\tBeta: Threshold for creating new components (higher = more components).",
	      "B", 1,"-B <percentile>"));
    newVector.addElement(new Option(
	      "\tPrune: Prune low prior components.",
	      "P", 1,"-P <Number of standard deviations below the mean of the priors>"));
    newVector.addElement(new Option(
	      "\tSkip Components: Skips updating and using for recalling low posterior components (faster but less precise).",
	      "S", 0, "-S"));
    newVector.addElement(new Option(
	      "\tAuto-Sigma: Automatically adjusts the initial component size for each new component.",
	      "A", 0, "-A"));
    newVector.addElement(new Option(
	      "\tAttribute Selection: Amount of partial correlation preserved, from 0 to 1.",
	      "AS", 1, "-AS"));
    newVector.addElement(new Option(
	      "\tAuto-Delta: Automatically adjusts delta to 3/#Classes for classification tasks.",
	      "AD", 0, "-AD"));
    return newVector.elements();
  }

    @Override
  public void setOptions(String[] options) throws Exception {

    System.out.println("Options: "+Arrays.toString(options));
    setVerbose(Utils.getFlag('V', options));
    setSkipComponents(Utils.getFlag('S', options));
    setAutoSigma(Utils.getFlag('A', options));
    setAutoDelta(Utils.getFlag("AD", options));
    //setAttrSelection(Utils.getFlag("AS", options));
    String attrString = Utils.getOption("AS", options);
    if (attrString.length() != 0) {
      setAttrSelection(Double.parseDouble(attrString));
    }
    String deltaString = Utils.getOption('D', options);
    if (deltaString.length() != 0) {
      setDelta(Double.parseDouble(deltaString));
    }
    String betaString = Utils.getOption('B', options);
    if (betaString.length() != 0) {
      setBeta(Double.parseDouble(betaString));
    }
    String pruneString = Utils.getOption('P', options);
    if (pruneString.length() != 0) {
      setPrune(Double.parseDouble(pruneString));
    }
    
    /*String nnSearchClass = Utils.getOption('A', options);
    if(nnSearchClass.length() != 0) {
      String nnSearchClassSpec[] = Utils.splitOptions(nnSearchClass);
      if(nnSearchClassSpec.length == 0) { 
        throw new Exception("Invalid NearestNeighbourSearch algorithm " +
                            "specification string."); 
      }
      String className = nnSearchClassSpec[0];
      nnSearchClassSpec[0] = "";

      setNearestNeighbourSearchAlgorithm( (NearestNeighbourSearch)
                  Utils.forName( NearestNeighbourSearch.class, 
                                 className, 
                                 nnSearchClassSpec)
                                        );
    }
    else 
      this.setNearestNeighbourSearchAlgorithm(new KDTree());
*/
    Utils.checkForRemainingOptions(options);
  }
    
    @Override
  public String [] getOptions() {

    String [] options = new String [10];
    int current = 0;
    if (getVerbose()) {
      options[current++] = "-V";
    }
    if (getSkipComponents()) {
      options[current++] = "-S";
    }
    if (getAutoSigma()) {
      options[current++] = "-A";
    }
    if (getAutoDelta()) {
      options[current++] = "-AD";
    }
    /*if (getAttrSelection()) {
      options[current++] = "-AS";
    }*/
    options[current++] = "-AS"; options[current++] = "" + getAttrSelection();
    options[current++] = "-D"; options[current++] = "" + getDelta();
    options[current++] = "-B"; options[current++] = "" + getBeta();
    options[current++] = "-P"; options[current++] = "" + getPrune();
    
    while (current < options.length) {
      options[current++] = "";
    }
    
    return options;
  }    
     @Override

  public String toString() {
      return "FIGMN "+Arrays.toString(getOptions())+" # of Components: ["+getNumComponents()+"]";
  }

    @Override
    public void updateClassifier(Instance instnc) throws Exception {
        //output("Incremental Learning from Instance: " +  instnc);
        if (getNumComponents() == 0) {
            updateSigmaIni(instanceToArray(instnc));
        }
        learn(instanceToArray(instnc), instnc.numClasses());
        //System.out.println("Step: "+totalCount()+" Num.Clusters: "+getNumComponents());
        //output("Instance: "+Arrays.toString(instanceToArray(instnc)));
        //output("Clusters: "+Arrays.toString(Arrays.copyOfRange(distributions.get(0).getMeans(),784,793)));
        //output("Cov: "+Arrays.toString(distributions.get(0).getInverseCov().getColumn(784)));
    }
}
