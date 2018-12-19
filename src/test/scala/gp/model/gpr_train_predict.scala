package gp.model
import breeze.linalg._
import breeze.numerics._
import gp.cov.{Cov_Matern, Cov_SEiso}
import org.junit._
class gpr_train_predict {

//  1.Test linear(y=2x)
  val x = DenseMatrix((1.0,2.0, 3.0,4.0,5.0,6.0,7.0,8.0,9.0)).t
  val y = 2.0*DenseVector(1.0,2.0, 3.0,4.0,5.0,6.0,7.0,8.0,9.0)
  val z = DenseMatrix((2.5, 4.5,6.5,8.5,10.0,12.0)).t
  val true_pre_Z = 2.0*DenseVector(2.5, 4.5,6.5,8.5,10.0,12.0)

//  //2.Test no_linear(y=cos(x)+1)
//  val x = DenseMatrix((1.0,2.0, 3.0,4.0,5.0,6.0,7.0,8.0,9.0)).t
//  val y = cos(DenseVector(1.0,2.0, 3.0,4.0,5.0,6.0,7.0,8.0,9.0))+1.0
//  val z = DenseMatrix((2.5, 4.5,6.5,8.5,10.0,12.0)).t
//  val true_pre_Z = cos(DenseVector(2.5, 4.5,6.5,8.5,10.0,12.0))+1.0

//  //3.Test no_linear(y=x^2)
//  val x = DenseMatrix((1.0,2.0, 3.0,4.0,5.0,6.0,7.0,8.0,9.0)).t
//  val y = DenseVector(1.0,4.0, 9.0,16.0,25.0,36.0,49.0,64.0,81.0)
//  val z = DenseMatrix((2.5, 4.5,6.5,8.5,10.0,12.0)).t
//  val true_pre_Z = pow(z,2)

  val covFunc = Cov_SEiso()
  val initial_covFuncParams = DenseVector(1.0,1.0)
  val initial_noiseStdDev = 0.1

  @Test def test = {

    val gpModel = gpr(x, y, covFunc, initial_covFuncParams, initial_noiseStdDev)
    println("Fitted covFuncParams:")
    println(gpModel.covFuncParams)
    println("Fitted noiseStdDev:")
    println(gpModel.noiseStdDev)
    println("\n")

    val prediction = gprPredict(z, gpModel)
    println("Mean and Var:")
    println(prediction)
    println("True value:")
    println(true_pre_Z)

  }
}