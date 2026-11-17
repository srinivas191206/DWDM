package dartx.monitor;

import java.io.Serializable;
import java.util.List;
import com.yahoo.labs.samoa.instances.Instance;

public class DataCharacteristicMonitor implements Serializable {
    private static final long serialVersionUID = 1L;

    public enum DriftState {
        STABLE,
        SUSPECTED_DRIFT,
        DRIFT_CONFIRMED,
        TRANSITION_RECOVERY
    }

    private int numFeatures;
    private double[] prevCentroid;
    private double[] currentCentroid;
    private double driftMagnitude;
    private DriftState currentState;

    private double driftThreshold = 0.08; // Normalized threshold for LBP feature centroid shift
    private double recentErrorRate = 0.0;
    private double prevErrorRate = 0.0;
    private int recoveryCooldown = 0;

    public DataCharacteristicMonitor(int numFeatures) {
        this.numFeatures = numFeatures;
        this.prevCentroid = null;
        this.currentCentroid = new double[numFeatures];
        this.driftMagnitude = 0.0;
        this.currentState = DriftState.STABLE;
    }

    public void updateBatchCharacteristics(List<Instance> batch, double batchErrorRate) {
        if (batch == null || batch.isEmpty()) return;

        // Compute centroid
        double[] newCentroid = new double[numFeatures];
        for (Instance inst : batch) {
            for (int i = 0; i < numFeatures; i++) {
                newCentroid[i] += inst.value(i);
            }
        }
        for (int i = 0; i < numFeatures; i++) {
            newCentroid[i] /= batch.size();
        }

        // Compute distance from previous centroid
        if (currentCentroid != null && prevCentroid != null) {
            double sumSq = 0.0;
            for (int i = 0; i < numFeatures; i++) {
                double diff = newCentroid[i] - currentCentroid[i];
                sumSq += diff * diff;
            }
            this.driftMagnitude = Math.sqrt(sumSq);
        } else {
            this.driftMagnitude = 0.0;
        }

        this.prevCentroid = this.currentCentroid;
        this.currentCentroid = newCentroid;

        // Update error dynamics
        this.prevErrorRate = this.recentErrorRate;
        this.recentErrorRate = batchErrorRate;
        double errorJump = recentErrorRate - prevErrorRate;

        // Drift state transition logic
        if (recoveryCooldown > 0) {
            recoveryCooldown--;
            currentState = DriftState.TRANSITION_RECOVERY;
        } else if (driftMagnitude > driftThreshold * 1.5 || errorJump > 0.15) {
            currentState = DriftState.DRIFT_CONFIRMED;
            recoveryCooldown = 2; // 2 time-steps recovery
        } else if (driftMagnitude > driftThreshold || errorJump > 0.08) {
            currentState = DriftState.SUSPECTED_DRIFT;
        } else {
            currentState = DriftState.STABLE;
        }
    }

    public DriftState getCurrentState() {
        return currentState;
    }

    public double getDriftMagnitude() {
        return driftMagnitude;
    }

    public double getRecentErrorRate() {
        return recentErrorRate;
    }

    public void setDriftThreshold(double threshold) {
        this.driftThreshold = threshold;
    }
}

