package gp.model

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.optimize.DiffFunction
import breeze.linalg.NotConvergedException
import breeze.linalg.MatrixNotSymmetricException

import scala.math.exp

case class GprDiffFunction(initialGpModel: GprModel) extends DiffFunction[DenseVector[Double]] {

  def calculate(params: DenseVector[Double]): (Double, DenseVector[Double]) = {

    try {
      val covFuncParams = DenseVector(params.toArray.dropRight(1))
      val noiseStdDev = params.toArray.last

      val gpModel = GprModel(initialGpModel.x, initialGpModel.y, initialGpModel.covFunc, covFuncParams, noiseStdDev, initialGpModel.meanFunc)

      val meanX = gpModel.meanFunc(gpModel.x)
      val kXX = gpModel.calcKXX()

      val kXXInv = gpModel.calcKXXInv(kXX)

      val f = -gprLoglik(meanX, kXX, kXXInv, gpModel.y)
//      println(f)

      //calculate partial derivatives
      val covFuncCovElemWiseD = gpModel.covFunc.covD(gpModel.x, gpModel.x, gpModel.covFuncParams)
      val noiseCovElemWiseD = 2 * noiseStdDev * DenseMatrix.eye[Double](gpModel.x.rows)
      val allParamsCovElemWiseD = covFuncCovElemWiseD :+ noiseCovElemWiseD

      val covFuncParamsD = gprLoglikD(meanX, kXXInv, gpModel.y, allParamsCovElemWiseD).map(d => -d)
//      println(covFuncParamsD)

      (f, covFuncParamsD)
    }catch {
      case e: NotConvergedException  => (Double.NaN, DenseVector.zeros[Double](params.size) * Double.NaN)
      case e: MatrixNotSymmetricException => (Double.NaN, DenseVector.zeros[Double](params.size) * Double.NaN)
    }
  }
}
