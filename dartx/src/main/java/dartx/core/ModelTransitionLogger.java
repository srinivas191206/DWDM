package dartx.core;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class ModelTransitionLogger implements Serializable {
    private static final long serialVersionUID = 1L;

    public static class TransitionRecord implements Serializable {
        private static final long serialVersionUID = 1L;
        public final int batchId;
        public final int fromModelId;
        public final int toModelId;
        public final String reason;
        public final double reliabilityDifference;

        public TransitionRecord(int batchId, int fromModelId, int toModelId, String reason, double diff) {
            this.batchId = batchId;
            this.fromModelId = fromModelId;
            this.toModelId = toModelId;
            this.reason = reason;
            this.reliabilityDifference = diff;
        }
    }

    private List<TransitionRecord> transitions = new ArrayList<>();

    public void logTransition(int batchId, int fromId, int toId, String reason, double diff) {
        transitions.add(new TransitionRecord(batchId, fromId, toId, reason, diff));
    }

    public List<TransitionRecord> getTransitions() {
        return transitions;
    }

    public int getTransitionCount() {
        return transitions.size();
    }

    public void clear() {
        transitions.clear();
    }
}

