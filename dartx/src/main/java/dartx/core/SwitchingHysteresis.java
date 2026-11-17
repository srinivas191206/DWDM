package dartx.core;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import br.ufpr.dynse.classifier.DynseClassifier;
import br.ufpr.dynse.pruningengine.DynseClassifierPruningMetrics;

public class SwitchingHysteresis implements Serializable {
    private static final long serialVersionUID = 1L;

    private double hysteresisThreshold = 0.04; // 4% reliability advantage required to switch
    private DynseClassifier<DynseClassifierPruningMetrics> incumbentClassifier = null;
    private long totalSwitches = 0;
    private long suppressedSwitches = 0;

    public SwitchingHysteresis(double threshold) {
        this.hysteresisThreshold = threshold;
    }

    public List<DynseClassifier<DynseClassifierPruningMetrics>> applyHysteresis(
            List<DynseClassifier<DynseClassifierPruningMetrics>> candidateEnsemble,
            Map<DynseClassifier<DynseClassifierPruningMetrics>, Double> reliabilityScores,
            List<DynseClassifier<DynseClassifierPruningMetrics>> pool) {

        if (candidateEnsemble == null || candidateEnsemble.isEmpty()) {
            return candidateEnsemble;
        }

        // Find the highest reliability candidate
        DynseClassifier<DynseClassifierPruningMetrics> topCandidate = candidateEnsemble.get(0);
        double topReliability = reliabilityScores.getOrDefault(topCandidate, 0.0);
        for (DynseClassifier<DynseClassifierPruningMetrics> dc : candidateEnsemble) {
            double r = reliabilityScores.getOrDefault(dc, 0.0);
            if (r > topReliability) {
                topReliability = r;
                topCandidate = dc;
            }
        }

        if (incumbentClassifier == null || !pool.contains(incumbentClassifier)) {
            // First run or incumbent was pruned
            incumbentClassifier = topCandidate;
            totalSwitches++;
            return candidateEnsemble;
        }

        double incumbentReliability = reliabilityScores.getOrDefault(incumbentClassifier, 0.0);

        if (topCandidate.equals(incumbentClassifier)) {
            // No switch needed
            return candidateEnsemble;
        }

        // Hysteresis condition: candidate must strictly beat incumbent by threshold
        if (topReliability > incumbentReliability + hysteresisThreshold) {
            // Legitimate switch
            incumbentClassifier = topCandidate;
            totalSwitches++;
            return candidateEnsemble;
        } else {
            // Suppress switch to avoid model thrashing
            suppressedSwitches++;
            List<DynseClassifier<DynseClassifierPruningMetrics>> retained = new ArrayList<>();
            retained.add(incumbentClassifier);
            return retained;
        }
    }

    public long getTotalSwitches() {
        return totalSwitches;
    }

    public long getSuppressedSwitches() {
        return suppressedSwitches;
    }

    public void resetCounts() {
        totalSwitches = 0;
        suppressedSwitches = 0;
        incumbentClassifier = null;
    }
}

