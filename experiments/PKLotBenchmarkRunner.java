package experiments;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
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
import dartx.core.DartXClassifier;
import moa.core.Utils;

public class PKLotBenchmarkRunner {

    public static class DayData {
        public final int order;
        public final String lot;
        public final String dayDirName;
        public final String arffPath;
        public final List<Instance> instances = new ArrayList<>();
        public final List<Instance> posInstances = new ArrayList<>();
        public final List<Instance> negInstances = new ArrayList<>();

        public DayData(int order, String lot, String dayDirName, String arffPath) {
            this.order = order;
            this.lot = lot;
            this.dayDirName = dayDirName;
            this.arffPath = arffPath;
        }
    }

    public static class DayMetrics {
        public int day;
        public String lot;
        public String date;
        public int total;
        public int pos;
        public int neg;
        public int tp;
        public int tn;
        public int fp;
        public int fn;
        public double accuracy;
        public double precision;
        public double recall;
        public double f1;
        public long inferenceTimeMs;
        public long trainTimeMs;
        public int windowSize;
        public long switches;
        public long suppressedSwitches;
        public long fallbackCount;
    }

    public static void main(String[] args) throws Exception {
        System.out.println("=================================================================");
        System.out.println("  DART-X Phase 2: PKLot 82-Day Comparative Benchmark Runner");
        System.out.println("=================================================================");

        String datasetRoot = "datasets/PKLot-ConceptDrift";
        List<Attribute> attributes = new ArrayList<>();
        for (int i = 1; i <= 59; i++) {
            attributes.add(new Attribute("attr_" + i));
        }
        List<String> classValues = new ArrayList<>();
        classValues.add("-1");
        classValues.add("+1");
        attributes.add(new Attribute("class", classValues));

        Instances datasetHeaderInst = new Instances("PKLot", attributes, 0);
        datasetHeaderInst.setClassIndex(59);
        InstancesHeader header = new InstancesHeader(datasetHeaderInst);

        // Discover and index all 82 days
        List<DayData> allDays = loadAndSortDayData(datasetRoot, datasetHeaderInst);
        System.out.println("Loaded " + allDays.size() + " days.");

        long masterSeed = 42L;

        // Run Models
        System.out.println("\n[1/5] Running Original Dynse Baseline (K-E, k=5, l=0, M=4, D=25)...");
        List<DayMetrics> dynseMetrics = runDynse(allDays, header, masterSeed);

        System.out.println("\n[2/5] Running DART-X Complete Framework...");
        List<DayMetrics> dartxMetrics = runDartX(allDays, header, masterSeed, true, true, true, true);

        System.out.println("\n[3/5] Running DART-X Ablation 1 (No Adaptive Window, static M=4)...");
        List<DayMetrics> noWindowMetrics = runDartX(allDays, header, masterSeed, false, true, true, true);

        System.out.println("\n[4/5] Running DART-X Ablation 2 (No Hysteresis, h=0)...");
        List<DayMetrics> noHystMetrics = runDartX(allDays, header, masterSeed, true, false, true, true);

        System.out.println("\n[5/5] Running DART-X Ablation 3 (No Reliability Scoring, pure hit count)...");
        List<DayMetrics> noRelMetrics = runDartX(allDays, header, masterSeed, true, true, false, true);

        // Write CSV outputs
        writeDailyCsv("results/baseline/pklot_dynse_daily.csv", dynseMetrics);
        writeDailyCsv("results/dartx/pklot_dartx_daily.csv", dartxMetrics);

        writeComparisonCsv("results/comparison/pklot_comparison_daily.csv", dynseMetrics, dartxMetrics);
        writeAblationCsv("results/comparison/pklot_ablation_daily.csv", dartxMetrics, noWindowMetrics, noHystMetrics, noRelMetrics);

        writeSummaryCsv("results/baseline/pklot_baseline_summary.csv", "Dynse", dynseMetrics);
        writeSummaryCsv("results/dartx/pklot_dartx_summary.csv", "DART-X", dartxMetrics);

        printSummary("Original Dynse Baseline", dynseMetrics);
        printSummary("DART-X Proposed Framework", dartxMetrics);
        printSummary("DART-X (No Adaptive Window)", noWindowMetrics);
        printSummary("DART-X (No Hysteresis)", noHystMetrics);
        printSummary("DART-X (No Reliability Scoring)", noRelMetrics);

        System.out.println("\nAll PKLot comparative experiments completed successfully!");
    }

    private static List<DayData> loadAndSortDayData(String baseDir, Instances datasetHeader) throws Exception {
        List<DayData> days = new ArrayList<>();
        File root = new File(baseDir);
        String[] lots = new String[]{"(1)UFPR04", "(2)UFPR05", "(3)PUC"};

        for (String lot : lots) {
            File lotDir = new File(root, lot);
            if (!lotDir.exists()) continue;
            File[] dayDirs = lotDir.listFiles(File::isDirectory);
            if (dayDirs == null) continue;

            for (File dDir : dayDirs) {
                String dName = dDir.getName();
                if (!dName.startsWith("(")) continue;
                int order = Integer.parseInt(dName.substring(1, dName.indexOf(")")));

                File[] arffs = dDir.listFiles((dir, name) -> name.endsWith(".arff"));
                if (arffs == null || arffs.length == 0) continue;
                String arffPath = arffs[0].getAbsolutePath();

                DayData dd = new DayData(order, lot, dName, arffPath);

                // Read instances
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
                    inst.setDataset(datasetHeader);
                    dd.instances.add(inst);
                    if (classIdx == 1) dd.posInstances.add(inst);
                    else dd.negInstances.add(inst);
                }
                br.close();
                days.add(dd);
            }
        }

        Collections.sort(days, (a, b) -> Integer.compare(a.order, b.order));
        return days;
    }

    private static List<DayMetrics> runDynse(List<DayData> allDays, InstancesHeader header, long seed) throws Exception {
        int trainBatchSize = 100;
        int mWindowBatches = 4;
        KnoraEliminateClassificationEngine ce = new KnoraEliminateClassificationEngine(5, 0);
        AgeBasedPruningEngine pe = new AgeBasedPruningEngine(25);
        HoeffdingTreeFactory cf = new HoeffdingTreeFactory();
        StreamDynse dynse = new StreamDynse(cf, trainBatchSize, mWindowBatches, ce, pe);
        dynse.setModelContext(header);

        Random rnd = new Random(seed);
        List<DayMetrics> metricsList = new ArrayList<>();
        List<Instance> prevDayTrain = null;

        for (int i = 0; i < allDays.size(); i++) {
            DayData day = allDays.get(i);
            DayMetrics dm = new DayMetrics();
            dm.day = day.order;
            dm.lot = day.lot;
            dm.date = day.dayDirName.substring(day.dayDirName.indexOf(")") + 1);
            dm.total = day.instances.size();
            dm.pos = day.posInstances.size();
            dm.neg = day.negInstances.size();

            // Training phase: 50 from previous day (or current day if Day 1)
            List<Instance> trainBatch = new ArrayList<>();
            List<Instance> sourcePos = (i == 0) ? day.posInstances : allDays.get(i - 1).posInstances;
            List<Instance> sourceNeg = (i == 0) ? day.negInstances : allDays.get(i - 1).negInstances;

            Collections.shuffle(sourcePos, rnd);
            Collections.shuffle(sourceNeg, rnd);
            for (int k = 0; k < 50; k++) {
                trainBatch.add(sourcePos.get(k));
                trainBatch.add(sourceNeg.get(k));
            }
            Collections.shuffle(trainBatch, rnd);

            long t0 = System.currentTimeMillis();
            for (Instance inst : trainBatch) {
                dynse.trainOnInstance(inst);
            }
            dm.trainTimeMs = System.currentTimeMillis() - t0;

            // Testing phase
            long tTest0 = System.currentTimeMillis();
            int tp = 0, tn = 0, fp = 0, fn = 0;

            for (Instance inst : day.instances) {
                double[] votes = dynse.getVotesForInstance(inst);
                int pred = Utils.maxIndex(votes);
                int trueClass = (int) inst.classValue();

                if (pred == 1 && trueClass == 1) tp++;
                else if (pred == 0 && trueClass == 0) tn++;
                else if (pred == 1 && trueClass == 0) fp++;
                else if (pred == 0 && trueClass == 1) fn++;
            }
            dm.inferenceTimeMs = System.currentTimeMillis() - tTest0;

            dm.tp = tp;
            dm.tn = tn;
            dm.fp = fp;
            dm.fn = fn;
            dm.accuracy = (double) (tp + tn) / dm.total * 100.0;
            dm.precision = (tp + fp > 0) ? (double) tp / (tp + fp) * 100.0 : 0.0;
            dm.recall = (tp + fn > 0) ? (double) tp / (tp + fn) * 100.0 : 0.0;
            double p = dm.precision;
            double r = dm.recall;
            dm.f1 = (p + r > 0) ? 2.0 * p * r / (p + r) : 0.0;
            dm.windowSize = 4;
            dm.switches = 0;
            dm.suppressedSwitches = 0;
            dm.fallbackCount = 0;

            metricsList.add(dm);

            if ((i + 1) % 10 == 0 || i == allDays.size() - 1) {
                System.out.printf("  [Dynse] Day %2d/82 (%s) | Acc: %5.2f%% | F1: %5.2f%% | Time: %d ms\n",
                        dm.day, dm.lot, dm.accuracy, dm.f1, (dm.trainTimeMs + dm.inferenceTimeMs));
            }
        }
        return metricsList;
    }

    private static List<DayMetrics> runDartX(List<DayData> allDays, InstancesHeader header, long seed,
                                             boolean adaptiveWindow, boolean hysteresis, boolean reliability, boolean fallback) throws Exception {
        int trainBatchSize = 100;
        int numFeatures = 59;
        DartXClassifier dartx = new DartXClassifier(trainBatchSize, numFeatures);
        dartx.setModelContext(header);
        dartx.setAblationConfig(adaptiveWindow, hysteresis, reliability, fallback);

        Random rnd = new Random(seed);
        List<DayMetrics> metricsList = new ArrayList<>();

        for (int i = 0; i < allDays.size(); i++) {
            DayData day = allDays.get(i);
            DayMetrics dm = new DayMetrics();
            dm.day = day.order;
            dm.lot = day.lot;
            dm.date = day.dayDirName.substring(day.dayDirName.indexOf(")") + 1);
            dm.total = day.instances.size();
            dm.pos = day.posInstances.size();
            dm.neg = day.negInstances.size();

            // Training phase: 50 per class from previous day
            List<Instance> trainBatch = new ArrayList<>();
            List<Instance> sourcePos = (i == 0) ? day.posInstances : allDays.get(i - 1).posInstances;
            List<Instance> sourceNeg = (i == 0) ? day.negInstances : allDays.get(i - 1).negInstances;

            Collections.shuffle(sourcePos, rnd);
            Collections.shuffle(sourceNeg, rnd);
            for (int k = 0; k < 50; k++) {
                trainBatch.add(sourcePos.get(k));
                trainBatch.add(sourceNeg.get(k));
            }
            Collections.shuffle(trainBatch, rnd);

            long t0 = System.currentTimeMillis();
            for (Instance inst : trainBatch) {
                dartx.trainOnInstance(inst);
            }
            dm.trainTimeMs = System.currentTimeMillis() - t0;

            // Testing phase
            long prevSwitches = dartx.getHysteresis().getTotalSwitches();
            long prevSuppressed = dartx.getHysteresis().getSuppressedSwitches();
            long prevFallbacks = dartx.getFallback().getFallbackTriggers();

            long tTest0 = System.currentTimeMillis();
            int tp = 0, tn = 0, fp = 0, fn = 0;

            for (Instance inst : day.instances) {
                double[] votes = dartx.getVotesForInstance(inst);
                int pred = Utils.maxIndex(votes);
                int trueClass = (int) inst.classValue();

                if (pred == 1 && trueClass == 1) tp++;
                else if (pred == 0 && trueClass == 0) tn++;
                else if (pred == 1 && trueClass == 0) fp++;
                else if (pred == 0 && trueClass == 1) fn++;
            }
            dm.inferenceTimeMs = System.currentTimeMillis() - tTest0;

            dm.tp = tp;
            dm.tn = tn;
            dm.fp = fp;
            dm.fn = fn;
            dm.accuracy = (double) (tp + tn) / dm.total * 100.0;
            dm.precision = (tp + fp > 0) ? (double) tp / (tp + fp) * 100.0 : 0.0;
            dm.recall = (tp + fn > 0) ? (double) tp / (tp + fn) * 100.0 : 0.0;
            double p = dm.precision;
            double r = dm.recall;
            dm.f1 = (p + r > 0) ? 2.0 * p * r / (p + r) : 0.0;

            dm.windowSize = dartx.getWindowManager().getCurrentWindowBatches();
            dm.switches = dartx.getHysteresis().getTotalSwitches() - prevSwitches;
            dm.suppressedSwitches = dartx.getHysteresis().getSuppressedSwitches() - prevSuppressed;
            dm.fallbackCount = dartx.getFallback().getFallbackTriggers() - prevFallbacks;

            // Notify monitor of batch completion
            double errorRate = (double) (fp + fn) / dm.total;
            dartx.notifyBatchCompleted(dm.day, day.instances, errorRate);

            metricsList.add(dm);

            if ((i + 1) % 10 == 0 || i == allDays.size() - 1) {
                System.out.printf("  [DART-X] Day %2d/82 (%s) | Acc: %5.2f%% | F1: %5.2f%% | Win: %d | Fallbacks: %d | Time: %d ms\n",
                        dm.day, dm.lot, dm.accuracy, dm.f1, dm.windowSize, dm.fallbackCount, (dm.trainTimeMs + dm.inferenceTimeMs));
            }
        }
        return metricsList;
    }

    private static void writeDailyCsv(String filePath, List<DayMetrics> list) throws Exception {
        PrintWriter pw = new PrintWriter(new FileWriter(filePath));
        pw.println("day,lot,date,total,pos,neg,tp,tn,fp,fn,accuracy,precision,recall,f1,train_time_ms,inference_time_ms,window_batches,switches,suppressed_switches,fallbacks");
        for (DayMetrics m : list) {
            pw.printf("%d,%s,%s,%d,%d,%d,%d,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%d,%d,%d,%d,%d,%d\n",
                    m.day, m.lot, m.date, m.total, m.pos, m.neg, m.tp, m.tn, m.fp, m.fn,
                    m.accuracy, m.precision, m.recall, m.f1, m.trainTimeMs, m.inferenceTimeMs,
                    m.windowSize, m.switches, m.suppressedSwitches, m.fallbackCount);
        }
        pw.close();
    }

    private static void writeComparisonCsv(String filePath, List<DayMetrics> dynse, List<DayMetrics> dartx) throws Exception {
        PrintWriter pw = new PrintWriter(new FileWriter(filePath));
        pw.println("day,lot,date,dynse_accuracy,dartx_accuracy,accuracy_diff,dynse_f1,dartx_f1,f1_diff,dartx_window,dartx_switches,dartx_suppressed,dartx_fallbacks");
        for (int i = 0; i < dynse.size(); i++) {
            DayMetrics d = dynse.get(i);
            DayMetrics x = dartx.get(i);
            pw.printf("%d,%s,%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%d,%d,%d\n",
                    d.day, d.lot, d.date, d.accuracy, x.accuracy, (x.accuracy - d.accuracy),
                    d.f1, x.f1, (x.f1 - d.f1), x.windowSize, x.switches, x.suppressedSwitches, x.fallbackCount);
        }
        pw.close();
    }

    private static void writeAblationCsv(String filePath, List<DayMetrics> full, List<DayMetrics> noWin, List<DayMetrics> noHyst, List<DayMetrics> noRel) throws Exception {
        PrintWriter pw = new PrintWriter(new FileWriter(filePath));
        pw.println("day,lot,date,full_dartx,no_adaptive_window,no_hysteresis,no_reliability");
        for (int i = 0; i < full.size(); i++) {
            pw.printf("%d,%s,%s,%.4f,%.4f,%.4f,%.4f\n",
                    full.get(i).day, full.get(i).lot, full.get(i).date,
                    full.get(i).accuracy, noWin.get(i).accuracy, noHyst.get(i).accuracy, noRel.get(i).accuracy);
        }
        pw.close();
    }

    private static void writeSummaryCsv(String filePath, String modelName, List<DayMetrics> list) throws Exception {
        PrintWriter pw = new PrintWriter(new FileWriter(filePath));
        pw.println("metric,overall,ufpr04,ufpr05,pucpr");

        double overallAcc = calcMeanAcc(list, 0, list.size());
        double overallStd = calcStdAcc(list, 0, list.size(), overallAcc);
        double ufpr04Acc = calcMeanAcc(list, 0, 22);
        double ufpr05Acc = calcMeanAcc(list, 22, 50);
        double pucprAcc = calcMeanAcc(list, 50, 82);

        pw.printf("mean_accuracy,%.2f,%.2f,%.2f,%.2f\n", overallAcc, ufpr04Acc, ufpr05Acc, pucprAcc);
        pw.printf("std_accuracy,%.2f,%.2f,%.2f,%.2f\n", overallStd,
                calcStdAcc(list, 0, 22, ufpr04Acc),
                calcStdAcc(list, 22, 50, ufpr05Acc),
                calcStdAcc(list, 50, 82, pucprAcc));

        long totalInfTime = 0;
        long totalInstances = 0;
        for (DayMetrics m : list) {
            totalInfTime += m.inferenceTimeMs;
            totalInstances += m.total;
        }
        pw.printf("avg_daily_inference_ms,%.1f,-,-,-\n", (double) totalInfTime / list.size());
        pw.printf("avg_instance_inference_us,%.2f,-,-,-\n", (double) totalInfTime / totalInstances * 1000.0);
        pw.close();
    }

    private static double calcMeanAcc(List<DayMetrics> list, int start, int end) {
        double sum = 0.0;
        for (int i = start; i < end; i++) {
            sum += list.get(i).accuracy;
        }
        return sum / (end - start);
    }

    private static double calcStdAcc(List<DayMetrics> list, int start, int end, double mean) {
        double sumSq = 0.0;
        for (int i = start; i < end; i++) {
            double diff = list.get(i).accuracy - mean;
            sumSq += diff * diff;
        }
        return Math.sqrt(sumSq / (end - start));
    }

    private static void printSummary(String name, List<DayMetrics> list) {
        double overallAcc = calcMeanAcc(list, 0, list.size());
        double overallStd = calcStdAcc(list, 0, list.size(), overallAcc);
        double ufpr04Acc = calcMeanAcc(list, 0, 22);
        double ufpr05Acc = calcMeanAcc(list, 22, 50);
        double pucprAcc = calcMeanAcc(list, 50, 82);

        double meanF1 = 0.0;
        long totalSwitches = 0;
        long totalSuppressed = 0;
        long totalFallbacks = 0;
        long totalInfTime = 0;

        for (DayMetrics m : list) {
            meanF1 += m.f1;
            totalSwitches += m.switches;
            totalSuppressed += m.suppressedSwitches;
            totalFallbacks += m.fallbackCount;
            totalInfTime += m.inferenceTimeMs;
        }
        meanF1 /= list.size();

        System.out.println("-----------------------------------------------------------------");
        System.out.printf("Summary: %s\n", name);
        System.out.printf("  Overall Accuracy : %5.2f%% (+/- %4.2f%%)\n", overallAcc, overallStd);
        System.out.printf("  UFPR04 (Days 1-22): %5.2f%%\n", ufpr04Acc);
        System.out.printf("  UFPR05 (Days 23-50): %5.2f%%\n", ufpr05Acc);
        System.out.printf("  PUCPR  (Days 51-82): %5.2f%%\n", pucprAcc);
        System.out.printf("  Macro F1-Score   : %5.2f%%\n", meanF1);
        System.out.printf("  Total Switches   : %d (Suppressed by Hysteresis: %d)\n", totalSwitches, totalSuppressed);
        System.out.printf("  Total Fallbacks  : %d\n", totalFallbacks);
        System.out.printf("  Inference Time   : %d ms total (%.1f ms/day)\n", totalInfTime, (double) totalInfTime / list.size());
        System.out.println("-----------------------------------------------------------------");
    }
}

