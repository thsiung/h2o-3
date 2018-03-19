package hex.genmodel.easy.prediction;

/**
 * TODO
 */
public class DimReductionModelPrediction extends AbstractPrediction {
    public double[] dimensions; // contains the X factors
    /**
     * Reconstructed data, the array has same length as the original input. The user can use the original input
     * and reconstructed output to easily calculate eg. the reconstruction error.
     */
    public double[] reconstructed;

}
