package dartx.core;

import java.io.Serializable;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;

import br.ufpr.dynse.classifier.DynseClassifier;
import br.ufpr.dynse.classifier.competence.IClassifierCompetence;
import br.ufpr.dynse.classifier.competence.IMultipleClassifiersCompetence;
import br.ufpr.dynse.pruningengine.DynseClassifierPruningMetrics;
import moa.core.Utils;

public class ReliabilityEvaluator implements Serializable {
    private static final long serialVersionUID = 1L;

    private static final double EPSILON = 1e-5;

    public Map<DynseClassifier<DynseClassifierPruningMetrics>, Double> evaluateReliability(
            Instance instance,
            Instances neighbors,
            double[] neighborDistances,
            List<DynseClassifier<DynseClassifierPruningMetrics>> availableClassifiers,
            Map<Instance, IMultipleClassifiersCompetence> classifiersMapping) {

        Map<DynseClassifier<DynseClassifierPruningMetrics>, Double> reliabilityScores = new HashMap<>();
        if (neighbors == null || neighbors.numInstances() == 0 || availableClassifiers == null || availableClassifiers.isEmpty()) {
            return reliabilityScores;
        }

        int k = neighbors.numInstances();
        double[] weights = new double[k];
        double weightSum = 0.0;

        for (int j = 0; j < k; j++) {
            double d = (neighborDistances != null && j < neighborDistances.length) ? neighborDistances[j] : 1.0;
            weights[j] = 1.0 / (Math.max(0.0, d) + EPSILON);
            weightSum += weights[j];
        }
        for (int j = 0; j < k; j++) {
            weights[j] /= weightSum;
        }

        for (DynseClassifier<DynseClassifierPruningMetrics> classifier : availableClassifiers) {
            double weightedScore = 0.0;

            for (int j = 0; j < k; j++) {
                Instance neighborInst = neighbors.instance(j);
                IMultipleClassifiersCompetence comp = classifiersMapping.get(neighborInst);
                double localScore = 0.0;

                if (comp != null) {
                    for (IClassifierCompetence cc : comp.getClassifiersCompetence()) {
                        if (cc.getClassifier().equals(classifier)) {
                            double[] dist = cc.getCompetenceOnInstance();
                            int trueClass = (int) neighborInst.classValue();
                            double conf = (dist != null && trueClass < dist.length) ? dist[trueClass] : 1.0;
                            localScore = conf;
                            break;
                        }
                    }
                }
                weightedScore += weights[j] * localScore;
            }

            reliabilityScores.put(classifier, weightedScore);
        }

        return reliabilityScores;
    }
}

