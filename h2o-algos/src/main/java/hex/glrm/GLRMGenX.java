package hex.glrm;

import hex.genmodel.GenModel;
import hex.genmodel.MojoModel;
import hex.genmodel.easy.EasyPredictModelWrapper;
import hex.genmodel.easy.RowData;
import hex.genmodel.easy.exception.PredictException;
import hex.genmodel.easy.prediction.AbstractPrediction;
import hex.genmodel.easy.prediction.DimReductionModelPrediction;
import water.H2O;
import water.MRTask;
import water.MemoryManager;
import water.api.StreamingSchema;
import water.fvec.Chunk;
import water.fvec.NewChunk;
import water.util.JCodeGen;
import water.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

/**
 * GLRMGenX will generate the coefficients (X matrix) of a GLRM model given the archetype
 * for a dataframe.
 */
public class GLRMGenX  extends MRTask<GLRMGenX> {
  final GLRMModel _m;
  final int _k;   // store column size of X matrix
  final EasyPredictModelWrapper _epmw;
  final double[] _features;
  final GenModel _genmodel;

  public GLRMGenX(GLRMModel m, int k) {
    _m = m;
    _m._parms = m._parms;
    _k = k;

    String modelName = JCodeGen.toJavaId(_m._key.toString());
    final String filename = modelName + ".zip";
    StreamingSchema ss = new StreamingSchema(getMojo(), filename);

    try { // generate mojo model
      FileOutputStream os = new FileOutputStream(ss.getFilename());
      ss.getStreamWriter().writeTo(os);
      os.close();
      _genmodel = MojoModel.load(filename);
      _features = MemoryManager.malloc8d(_genmodel._names.length);
    } catch (IOException e1) {
      e1.printStackTrace();
      throw H2O.fail("Internal MOJO loading failed", e1);
    } finally {
      boolean deleted = new File(filename).delete();
      if (!deleted) Log.warn("Failed to delete the file");
    }
    // instantiate mojo model
    EasyPredictModelWrapper.Config config = new EasyPredictModelWrapper.Config();
    _epmw = new EasyPredictModelWrapper(
            config.setModel(_genmodel).setConvertUnknownCategoricalLevelsToNa(true));
  }

  public void map(Chunk[] chks, NewChunk[] preds) {
    RowData rowData = new RowData();  // massage each row of dataset into RowData format

    for (int rid = 0; rid < chks[0]._len; ++rid) {
      for (int col = 0; col < _features.length; col++) {
        double val = chks[col].atd(rid);
        rowData.put(
                _genmodel._names[col],
                _genmodel._domains[col] == null ? (Double) val
                        : Double.isNaN(val) ? val  // missing categorical values are kept as NaN, the score0 logic passes it on to bitSetContains()
                        : (int) val < _genmodel._domains[col].length ? _genmodel._domains[col][(int) val] : "UnknownLevel"); //unseen levels are treated as such
      }

      AbstractPrediction p;
      try {
        p = _epmw.predict(rowData);
        for (int c = 0; c < _k; c++)  // Output predictions; sized for train only (excludes extra test classes)
          preds[c].addNum(((DimReductionModelPrediction) p).dimensions[c]);
      } catch (PredictException e) {
          System.err.println("EasyPredict threw an exception when predicting row " + rowData);
          e.printStackTrace();
      }
    }
  }

  public GlrmMojoWriter getMojo() {
    return new GlrmMojoWriter(_m);
  }
}
