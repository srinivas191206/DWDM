package dartx.core;

import java.io.Serializable;
import dartx.monitor.DataCharacteristicMonitor;
import dartx.monitor.DataCharacteristicMonitor.DriftState;

public class AdaptiveWindowManager implements Serializable {
    private static final long serialVersionUID = 1L;

    private int minWindowBatches = 1;
    private int maxWindowBatches = 6;
    private int defaultWindowBatches = 4;
    private int currentWindowBatches = 4;
    private int batchSize = 100;

    public AdaptiveWindowManager(int batchSize, int defaultBatches) {
        this.batchSize = batchSize;
        this.defaultWindowBatches = defaultBatches;
        this.currentWindowBatches = defaultBatches;
    }

    public int updateWindowSize(DriftState state, double driftMagnitude) {
        switch (state) {
            case DRIFT_CONFIRMED:
                // Fast eviction of pre-drift instances
                currentWindowBatches = minWindowBatches;
                break;
            case SUSPECTED_DRIFT:
                currentWindowBatches = Math.max(minWindowBatches, currentWindowBatches - 1);
                break;
            case TRANSITION_RECOVERY:
                currentWindowBatches = Math.min(3, currentWindowBatches + 1);
                break;
            case STABLE:
            default:
                // Expand window slowly up to maxWindowBatches
                currentWindowBatches = Math.min(maxWindowBatches, currentWindowBatches + 1);
                break;
        }
        return currentWindowBatches;
    }

    public int getCurrentWindowInstances() {
        return currentWindowBatches * batchSize;
    }

    public int getCurrentWindowBatches() {
        return currentWindowBatches;
    }

    public void setMinMax(int min, int max) {
        this.minWindowBatches = min;
        this.maxWindowBatches = max;
    }
}

