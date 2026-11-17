package dartx.core;

import java.io.Serializable;
import java.util.List;
import java.util.Map;

import com.yahoo.labs.samoa.instances.Instance;

import br.ufpr.dynse.classifier.DynseClassifier;
import br.ufpr.dynse.pruningengine.DynseClassifierPruningMetrics;
import moa.classifiers.Classifier;

public class ReliabilityFallback implements Serializable {
    private static final long serialVersionUID = 1L;

    private double safetyThreshold = 0.35;
    private long fallbackTriggers = 0;

    public ReliabilityFallback(double safetyThreshold) {
        this.safetyThreshold = safetyThreshold;
    }

    public boolean shouldTrigger(double maxReliability) {
        return maxReliability < safetyThreshold;
    }

    public double[] executeFallback(Instance instance, List<DynseClassifier<DynseClassifierPruningMetrics>> pool) {
        fallbackTriggers++;
        if (pool == null || pool.isEmpty()) {
            double[] probs = new double[instance.numClasses()];
            probs[0] = 1.0;
            return probs;
        }

        // Strategy: Combine recent models weighted by recency (age-weighted decay)
        double[] combinedVotes = new double[instance.numClasses()];
        double weightSum = 0.0;
        int n = pool.size();

        for (int i = 0; i < n; i++) {
            DynseClassifier<DynseClassifierPruningMetrics> dc = pool.get(i);
            // Exponential recency weight: newest model gets highest weight
            double w = Math.exp((double) i / n);
            double[] votes = dc.getVotesForInstance(instance);

            for (int c = 0; c < Math.min(votes.length, combinedVotes.length); c++) {
                combinedVotes[c] += w * votes[c];
            }
            weightSum += w;
        }

        if (weightSum > 0) {
            for (int c = 0; c < combinedVotes.length; c++) {
                combinedVotes[c] /= weightSum;
            }
        }
        return combinedVotes;
    }

    public long getFallbackTriggers() {
        return fallbackTriggers;
    }

    public void reset() {
        fallbackTriggers = 0;
    }
}

