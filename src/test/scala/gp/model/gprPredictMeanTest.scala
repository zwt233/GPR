package gp.model

import breeze.linalg.{DenseMatrix, DenseVector}
import org.junit.Assert._
import org.junit._
import gp.cov.Cov_Matern
import gp.cov.Cov_SEiso
import breeze.numerics._


class gprPredictMeanTest {
  val covFunc = Cov_Matern()
  val fitted_covFuncParams = DenseVector(1.0, 1.0)
  val fitted_noiseStdDev =0.1

  @Test def test_1d_inputs = {

    val x = DenseMatrix((1.0,2.0, 3.0,4.0,5.0,6.0,7.0,8.0,9.0)).t
      val y = cos(DenseVector(1.0,2.0, 3.0,4.0,5.0,6.0,7.0,8.0,9.0))+1.0
      val z = DenseMatrix((2.5, 4.5,6.5,8.5,10.0,12.0)).t
    val true_pre_Z = cos(DenseVector(2.5, 4.5,6.5,8.5,10.0,12.0))+1.0

    println(y)
    val gpModel = GprModel(x, y, covFunc, fitted_covFuncParams, fitted_noiseStdDev)

    val prediction = gprPredict(z, gpModel)
//    val expected = new DenseMatrix(5, 2, Array(0.878, 4.407, 8.614, 10.975, 0.00001, 1.246, 1.123, 1.246, 6.063, 57.087))
    println(prediction)
    println(true_pre_Z)
//    assertEquals(expected.map(v => "%.3f".format(v)).toString, prediction.map(v => "%.3f".format(v)).toString())
  }
}