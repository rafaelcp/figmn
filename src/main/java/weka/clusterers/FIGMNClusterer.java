/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package weka.clusterers;

import figmn.FIGMN;
import java.io.Serializable;
import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.CapabilitiesHandler;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

/**
 *
 * @author rafa
 */
public class FIGMNClusterer extends FIGMN implements Clusterer, CapabilitiesHandler, Serializable {
        
    Attribute classAttribute;
    
    public FIGMNClusterer() {
        super();
    }
    
    public FIGMNClusterer(double[][] data, double delta) {
        super(data, delta);
    }
        
    public static void main(String[] args) {
        if (args.length == 0) {
            DataSource source;
            try {
                //source = new DataSource("C:\\Program Files\\Weka-3-7\\data\\iris.arff");
                source = new DataSource("C:\\Users\\Rafael\\mnist_train.arff");
                Instances dt = source.getDataSet();
                //dt.randomize(new Random());
                if (dt.classIndex() == -1)
                   dt.setClassIndex(dt.numAttributes() - 1);
                double[][] data;
                data = FIGMN.instancesToDoubleArrays(dt);
                FIGMNClusterer igmn = new FIGMNClusterer(data,9);
                //igmn.setBeta(0);
                //igmn.setDelta(0.1);
                igmn.setVerbose(true);
                igmn.learn(data);
                System.out.println("# of Components: "+igmn.numberOfClusters() + " List: " + igmn.getDistributions());
            } catch (Exception ex) {
                Logger.getLogger(FIGMNClusterer.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
        else {
            System.out.println(Arrays.toString(args));
            AbstractClusterer.runClusterer(new FIGMNClusterer(),args);
        }
    }
    
    
/*
    @Override
    public void updateClusterer(Instance instnc) throws Exception {
        learn(instnc.toDoubleArray());
    }

    @Override
    public void updateFinished() {
        
    }
*/
    @Override
    public void buildClusterer(Instances i) throws Exception {
        getCapabilities().testWithFail(i);
        double[][] data = FIGMN.instancesToDoubleArrays(i);
        clearComponents();
        updateSigmaIni(data);
        learn(data);
        classAttribute = i.classAttribute();
    }

    
    @Override
    public Capabilities getCapabilities() {
        Capabilities result = new Capabilities(this);
        // attributes
        result.disableAll();
        result.enable(Capability.BINARY_ATTRIBUTES);
        result.enable(Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capability.NUMERIC_ATTRIBUTES);
        // class
        result.enable(Capability.NO_CLASS);
        return result;
    }

}
