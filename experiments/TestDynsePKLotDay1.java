package experiments;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;

import br.ufpr.dynse.classificationengine.KnoraEliminateClassificationEngine;
import br.ufpr.dynse.classifier.factory.HoeffdingTreeFactory;
import br.ufpr.dynse.core.StreamDynse;
import br.ufpr.dynse.pruningengine.AgeBasedPruningEngine;
import moa.core.Utils;

public class TestDynsePKLotDay1 {
    public static void main(String[] args) throws Exception {
        System.out.println("Testing Dynse on Day 1...");
        String arffPath = "datasets/PKLot-ConceptDrift/(1)UFPR04/(1)2012-12-07/UFPR04_LBP_U2.arff";
        
        // Define attributes: 59 numeric + 1 class {-1, +1}
        List<Attribute> attributes = new ArrayList<>();
        for (int i = 1; i <= 59; i++) {
            attributes.add(new Attribute("attr_" + i));
        }
        List<String> classValues = new ArrayList<>();
        classValues.add("-1");
        classValues.add("+1");
        attributes.add(new Attribute("class", classValues));
        
        Instances dataset = new Instances("PKLot", attributes, 0);
        dataset.setClassIndex(59);
        InstancesHeader header = new InstancesHeader(dataset);
        
        // Read lines
        List<Instance> allInstances = new ArrayList<>();
        List<Instance> posInstances = new ArrayList<>();
        List<Instance> negInstances = new ArrayList<>();
        
        BufferedReader br = new BufferedReader(new FileReader(arffPath));
        String line;
        while ((line = br.readLine()) != null) {
            line = line.trim();
            if (line.isEmpty() || line.startsWith("@") || line.startsWith("%")) continue;
            String[] parts = line.split(",");
            double[] attValues = new double[60];
            for (int i = 0; i < 59; i++) {
                attValues[i] = Double.parseDouble(parts[i]);
            }
            int classIdx = parts[59].equals("+1") ? 1 : 0;
            attValues[59] = classIdx;
            
            DenseInstance inst = new DenseInstance(1.0, attValues);
            inst.setDataset(dataset);
            allInstances.add(inst);
            if (classIdx == 1) posInstances.add(inst);
            else negInstances.add(inst);
        }
        br.close();
        System.out.println("Read " + allInstances.size() + " instances (" + posInstances.size() + " pos, " + negInstances.size() + " neg)");
        
        // Setup Dynse with K-E (k=5, l=0), M=4, pool=25, Age-based pruning, Hoeffding Tree
        int trainBatchSize = 100;
        int mWindowBatches = 4;
        KnoraEliminateClassificationEngine ce = new KnoraEliminateClassificationEngine(5, 0);
        AgeBasedPruningEngine pe = new AgeBasedPruningEngine(25);
        HoeffdingTreeFactory cf = new HoeffdingTreeFactory();
        StreamDynse dynse = new StreamDynse(cf, trainBatchSize, mWindowBatches, ce, pe);
        dynse.setModelContext(header);
        
        // Sample 50 pos and 50 neg for training
        Random rnd = new Random(42);
        Collections.shuffle(posInstances, rnd);
        Collections.shuffle(negInstances, rnd);
        List<Instance> trainBatch = new ArrayList<>();
        for (int i = 0; i < 50; i++) {
            trainBatch.add(posInstances.get(i));
            trainBatch.add(negInstances.get(i));
        }
        Collections.shuffle(trainBatch, rnd);
        
        long tTrainStart = System.currentTimeMillis();
        for (Instance inst : trainBatch) {
            dynse.trainOnInstance(inst);
        }
        long tTrainEnd = System.currentTimeMillis();
        System.out.println("Training 100 instances took " + (tTrainEnd - tTrainStart) + " ms");
        
        // Evaluate on all instances
        long tTestStart = System.currentTimeMillis();
        int correct = 0;
        for (Instance inst : allInstances) {
            double[] votes = dynse.getVotesForInstance(inst);
            int pred = Utils.maxIndex(votes);
            if (pred == (int) inst.classValue()) {
                correct++;
            }
        }
        long tTestEnd = System.currentTimeMillis();
        double acc = (double) correct / allInstances.size() * 100.0;
        System.out.printf("Tested %d instances in %d ms. Accuracy: %.2f%%\n", allInstances.size(), (tTestEnd - tTestStart), acc);
    }
}
