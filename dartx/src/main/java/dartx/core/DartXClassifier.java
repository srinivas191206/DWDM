package dartx.core;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;

import br.ufpr.dynse.classificationengine.IClassificationEngine;
import br.ufpr.dynse.classificationengine.KnoraEliminateClassificationEngine;
import br.ufpr.dynse.classifier.DynseClassifier;
import br.ufpr.dynse.classifier.MajorityVoting;
import br.ufpr.dynse.classifier.competence.ClassifierCompetence;
import br.ufpr.dynse.classifier.competence.IClassifierCompetence;
import br.ufpr.dynse.classifier.competence.IMultipleClassifiersCompetence;
import br.ufpr.dynse.classifier.competence.MultipleClassifiersCompetence;
import br.ufpr.dynse.classifier.factory.AbstractClassifierFactory;
import br.ufpr.dynse.classifier.factory.HoeffdingTreeFactory;
import br.ufpr.dynse.instance.InstanceComparator;
import br.ufpr.dynse.pruningengine.AgeBasedPruningEngine;
import br.ufpr.dynse.pruningengine.DynseClassifierPruningMetrics;
import br.ufpr.dynse.pruningengine.IPruningEngine;
import br.ufpr.dynse.util.InstancesUtils;
import dartx.monitor.DataCharacteristicMonitor;
import dartx.monitor.DataCharacteristicMonitor.DriftState;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.MultiClassClassifier;
import moa.classifiers.lazy.neighboursearch.LinearNNSearch;
import moa.classifiers.lazy.neighboursearch.NearestNeighbourSearch;
import moa.core.Measurement;
import moa.core.Utils;

public class DartXClassifier extends AbstractClassifier implements MultiClassClassifier, Serializable {
    private static final long serialVersionUID = 1L;

    // Core Dynse parameters
    private int trainBatchSize;
    private int maxPoolSize = 25;
    private int kNeighbors = 5;
    private AbstractClassifierFactory classifierFactory;
    private IPruningEngine<DynseClassifierPruningMetrics> pruningEngine;
    private NearestNeighbourSearch nnSearch;

    // Sliding buffers
    private LinkedList<Instance> trainAccumulator;
    private LinkedList<Instance> validationWindow;
    private Map<Instance, IMultipleClassifiersCompetence> classifiersMapping;
    private List<DynseClassifier<DynseClassifierPruningMetrics>> classifiersPool;
    private boolean updateNNSearch = false;

    // DART-X Enhancements
    private DataCharacteristicMonitor monitor;
    private AdaptiveWindowManager windowManager;
    private ReliabilityEvaluator reliabilityEvaluator;
    private SwitchingHysteresis hysteresis;
    private ModelTransitionLogger transitionLogger;
    private ReliabilityFallback fallback;

    // Ablation toggle switches
    private boolean enableAdaptiveWindow = true;
    private boolean enableHysteresis = true;
    private boolean enableReliabilityScoring = true;
    private boolean enableFallback = true;

    private MajorityVoting<DynseClassifier<DynseClassifierPruningMetrics>> classifierCombiner;
    private final InstanceComparator instanceComparator = new InstanceComparator();

    public DartXClassifier(int trainBatchSize, int numFeatures) {
        this.trainBatchSize = trainBatchSize;
        this.classifierFactory = new HoeffdingTreeFactory();
        try {
            this.pruningEngine = new AgeBasedPruningEngine(maxPoolSize);
            this.nnSearch = new LinearNNSearch();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        this.trainAccumulator = new LinkedList<>();
        this.validationWindow = new LinkedList<>();
        this.classifiersPool = new ArrayList<>();
        this.classifiersMapping = new TreeMap<>(instanceComparator);
        this.classifierCombiner = new MajorityVoting<>();

        // Initialize DART-X modules
        this.monitor = new DataCharacteristicMonitor(numFeatures);
        this.windowManager = new AdaptiveWindowManager(trainBatchSize, 4);
        this.reliabilityEvaluator = new ReliabilityEvaluator();
        this.hysteresis = new SwitchingHysteresis(0.04);
        this.transitionLogger = new ModelTransitionLogger();
        this.fallback = new ReliabilityFallback(0.35);
    }

    public void notifyBatchCompleted(int batchId, List<Instance> batchInstances, double batchErrorRate) {
        // Update monitor
        monitor.updateBatchCharacteristics(batchInstances, batchErrorRate);
        DriftState state = monitor.getCurrentState();

        if (enableAdaptiveWindow) {
            windowManager.updateWindowSize(state, monitor.getDriftMagnitude());
            pruneValidationWindowToCapacity();
        }
    }

    private void pruneValidationWindowToCapacity() {
        int maxAllowed = windowManager.getCurrentWindowInstances();
        while (validationWindow.size() > maxAllowed && !validationWindow.isEmpty()) {
            Instance removed = validationWindow.removeFirst();
            IMultipleClassifiersCompetence comp = classifiersMapping.get(removed);
            if (comp != null) {
                if (comp.getInstanceCount() > 1) {
                    comp.setInstanceCount(comp.getInstanceCount() - 1);
                } else {
                    classifiersMapping.remove(removed);
                }
            }
            updateNNSearch = true;
        }
    }

    @Override
    public void trainOnInstanceImpl(Instance instance) {
        trainAccumulator.addLast(instance);
        validationWindow.addLast(instance);

        int maxAllowed = enableAdaptiveWindow ? windowManager.getCurrentWindowInstances() : (4 * trainBatchSize);
        Instance removed = null;
        if (validationWindow.size() > maxAllowed) {
            removed = validationWindow.removeFirst();
        }

        updateClassifiersMap(instance, removed);

        if (trainAccumulator.size() >= trainBatchSize) {
            try {
                trainNewClassifierFromAccumulator();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
    }

    private void updateClassifiersMap(Instance instance, Instance removed) {
        Set<IClassifierCompetence> competences = new HashSet<>();
        for (DynseClassifier<DynseClassifierPruningMetrics> dc : classifiersPool) {
            double[] dist = dc.getVotesForInstance(instance);
            if (Utils.maxIndex(dist) == (int) instance.classValue()) {
                competences.add(new ClassifierCompetence(dc, dist));
            }
        }

        IMultipleClassifiersCompetence mcc = classifiersMapping.get(instance);
        if (mcc != null) {
            mcc.setInstanceCount(mcc.getInstanceCount() + 1);
            mcc.getClassifiersCompetence().addAll(competences);
        } else {
            mcc = new MultipleClassifiersCompetence();
            mcc.setInstance(instance);
            mcc.setClassifiersCompetence(competences);
            classifiersMapping.put(instance, mcc);
        }

        if (removed != null) {
            IMultipleClassifiersCompetence remComp = classifiersMapping.get(removed);
            if (remComp != null) {
                if (remComp.getInstanceCount() > 1) {
                    remComp.setInstanceCount(remComp.getInstanceCount() - 1);
                } else {
                    classifiersMapping.remove(removed);
                }
            }
        }
        updateNNSearch = true;
    }

    private void trainNewClassifierFromAccumulator() throws Exception {
        Classifier rawClassifier = classifierFactory.buildClassifier(trainAccumulator);
        DynseClassifier<DynseClassifierPruningMetrics> newDC = new DynseClassifier<>(rawClassifier);
        newDC.setDynseClassifierMetrics(new DynseClassifierPruningMetrics(newDC));

        // Apply pruning engine
        List<DynseClassifier<DynseClassifierPruningMetrics>> toPrune =
                pruningEngine.pruneClassifiers(newDC, classifiersPool, validationWindow);

        for (DynseClassifier<DynseClassifierPruningMetrics> dc : toPrune) {
            if (dc != newDC) {
                pruneClassifierFromMapping(dc);
                classifiersPool.remove(dc);
            } else {
                newDC = null;
            }
        }

        if (newDC != null) {
            classifiersPool.add(newDC);
            mapClassifierCompetenceOnWindow(newDC);
        }

        trainAccumulator.clear();
    }

    private void pruneClassifierFromMapping(DynseClassifier<DynseClassifierPruningMetrics> dc) {
        for (IMultipleClassifiersCompetence mcc : classifiersMapping.values()) {
            Set<IClassifierCompetence> toRemove = new HashSet<>();
            for (IClassifierCompetence comp : mcc.getClassifiersCompetence()) {
                if (comp.getClassifier().equals(dc)) {
                    toRemove.add(comp);
                }
            }
            mcc.getClassifiersCompetence().removeAll(toRemove);
        }
    }

    private void mapClassifierCompetenceOnWindow(DynseClassifier<DynseClassifierPruningMetrics> dc) {
        for (IMultipleClassifiersCompetence mcc : classifiersMapping.values()) {
            double[] dist = dc.getVotesForInstance(mcc.getInstance());
            if (Utils.maxIndex(dist) == (int) mcc.getInstance().classValue()) {
                mcc.getClassifiersCompetence().add(new ClassifierCompetence(dc, dist));
            }
        }
    }

    private void updateNNSearchInstances() {
        try {
            nnSearch.setInstances(InstancesUtils.gerarDataset(validationWindow, "Validation Instances"));
            updateNNSearch = false;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public double[] getVotesForInstance(Instance instance) {
        if (classifiersPool.isEmpty()) {
            if (!trainAccumulator.isEmpty()) {
                try {
                    trainNewClassifierFromAccumulator();
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            } else {
                double[] p = new double[instance.numClasses()];
                p[0] = 1.0;
                return p;
            }
        }

        if (updateNNSearch) {
            updateNNSearchInstances();
        }

        try {
            Instances neighbors = nnSearch.kNearestNeighbours(instance, kNeighbors);
            double[] distances = nnSearch.getDistances();

            // 1. Reliability Evaluation
            Map<DynseClassifier<DynseClassifierPruningMetrics>, Double> reliabilityScores = null;
            if (enableReliabilityScoring) {
                reliabilityScores = reliabilityEvaluator.evaluateReliability(
                        instance, neighbors, distances, classifiersPool, classifiersMapping);
            }

            // Check if fallback needed
            if (enableFallback && reliabilityScores != null) {
                double maxR = 0.0;
                for (double r : reliabilityScores.values()) {
                    if (r > maxR) maxR = r;
                }
                if (fallback.shouldTrigger(maxR)) {
                    return fallback.executeFallback(instance, classifiersPool);
                }
            }

            // 2. Candidate Selection (KNORA-E base selection)
            List<DynseClassifier<DynseClassifierPruningMetrics>> candidates = selectKnoraECandidates(neighbors);

            // 3. Switching Hysteresis
            if (enableHysteresis && reliabilityScores != null) {
                candidates = hysteresis.applyHysteresis(candidates, reliabilityScores, classifiersPool);
            }

            if (candidates.isEmpty()) {
                candidates = classifiersPool;
            }

            return classifierCombiner.distributionForInstance(instance, candidates);

        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private List<DynseClassifier<DynseClassifierPruningMetrics>> selectKnoraECandidates(Instances neighbors) {
        Map<DynseClassifier<DynseClassifierPruningMetrics>, Integer> hitMap = new java.util.HashMap<>();
        int k = neighbors.numInstances();
        int reqHits = k; // slack l = 0

        for (int j = 0; j < k; j++) {
            IMultipleClassifiersCompetence comp = classifiersMapping.get(neighbors.instance(j));
            if (comp != null) {
                for (IClassifierCompetence cc : comp.getClassifiersCompetence()) {
                    hitMap.put(cc.getClassifier(), hitMap.getOrDefault(cc.getClassifier(), 0) + 1);
                }
            }
        }

        List<DynseClassifier<DynseClassifierPruningMetrics>> selected = new ArrayList<>();
        while (reqHits > 0) {
            for (Map.Entry<DynseClassifier<DynseClassifierPruningMetrics>, Integer> entry : hitMap.entrySet()) {
                if (entry.getValue() >= reqHits && !selected.contains(entry.getKey())) {
                    selected.add(entry.getKey());
                }
            }
            if (!selected.isEmpty()) break;
            reqHits--;
        }

        return selected;
    }

    // Ablation configuration methods
    public void setAblationConfig(boolean adaptiveWindow, boolean hysteresis, boolean reliability, boolean fallback) {
        this.enableAdaptiveWindow = adaptiveWindow;
        this.enableHysteresis = hysteresis;
        this.enableReliabilityScoring = reliability;
        this.enableFallback = fallback;
    }

    public SwitchingHysteresis getHysteresis() {
        return hysteresis;
    }

    public DataCharacteristicMonitor getMonitor() {
        return monitor;
    }

    public AdaptiveWindowManager getWindowManager() {
        return windowManager;
    }

    public ReliabilityFallback getFallback() {
        return fallback;
    }

    public int getPoolSize() {
        return classifiersPool.size();
    }

    @Override
    public boolean isRandomizable() {
        return false;
    }

    @Override
    public void resetLearningImpl() {
        trainAccumulator.clear();
        validationWindow.clear();
        classifiersPool.clear();
        classifiersMapping.clear();
        updateNNSearch = false;
        if (hysteresis != null) hysteresis.resetCounts();
        if (fallback != null) fallback.reset();
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return new Measurement[] {
            new Measurement("Pool Size", classifiersPool.size()),
            new Measurement("Switches", hysteresis.getTotalSwitches()),
            new Measurement("Suppressed", hysteresis.getSuppressedSwitches()),
            new Measurement("Fallbacks", fallback.getFallbackTriggers())
        };
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
        out.append("DART-X Framework\n");
    }
}

