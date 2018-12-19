package gp.model
import gp.cov.CovFunc
import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.optimize.LBFGS

object gpr {

  def apply(x: DenseMatrix[Double], y: DenseVector[Double], covFunc: CovFunc, covFuncParams: DenseVector[Double], noiseStdDev: Double, mean: Double = 0d,gpMaxIter:Int=1000000): GprModel = {

    val initialGpModel = GprModel(x, y, covFunc, covFuncParams, noiseStdDev, mean)

    val diffFunc = GprDiffFunction(initialGpModel)
    val initialParams = DenseVector(covFuncParams.toArray :+ noiseStdDev)

    val optimizer = new LBFGS[DenseVector[Double]](maxIter = 100000, m = 3, tolerance = 1)
    val newParams = optimizer.minimize(diffFunc,initialParams)
//    println(optimizer)
//    println(result)
//    val optimizer = new LBFGS[DenseVector[Double]](maxIter = 10000, m = 3, tolerance = 1.0E-5)
//    val optIterations = optimizer.iterations(diffFunc, initialParams).toList
//    val newParams = optIterations.last.x

    val newCovFuncParams = DenseVector(newParams.toArray.dropRight(1))
    val newNoiseLogStdDev = newParams.toArray.last

    val trainedGpModel = GprModel(x,y,covFunc,newCovFuncParams,newNoiseLogStdDev,mean)
    trainedGpModel
  }
}
