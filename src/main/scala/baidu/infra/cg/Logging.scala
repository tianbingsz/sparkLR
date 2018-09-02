package baidu.infra.cg

class Logging extends Serializable {
  var loglik = Array[Double]()
  var rtime = Array[Double]()
  var alphas = Array[Double]()
  var betas = Array[Double]()
  var wNorm = Array[Double]()
  var wDiffNorm = Array[Double]()
  var w0 = Array[Double]()
  var gradNorm = Array[Double]()
  var gtd = Array[Double]()
  // g'mu
  var uHu = Array[Double]()
  // u' H u
  var gtold = Array[Double]() // g'oldmu
}
