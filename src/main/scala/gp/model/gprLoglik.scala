package gp.model
import scala.math.Pi
import breeze.linalg.{DenseMatrix, DenseVector, cholesky}
import gp.math.logdetchol
import scala.math.log

object gprLoglik {

  def apply(xMean: DenseVector[Double], kXX: DenseMatrix[Double], kXXInv: DenseMatrix[Double], y: DenseVector[Double]): Double = {

    val m = xMean

    val logDet = logdetchol(cholesky(kXX))

    val loglikValue = (-0.5 * (y - m).t * kXXInv * (y - m) - 0.5 * logDet - 0.5 * xMean.size.toDouble * log(2 * Pi))

    loglikValue(0)
  }
}
