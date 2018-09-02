package baidu.infra.cg

import org.apache.spark.rdd.RDD
import org.apache.spark.util.Vector

import scala.util.control.Breaks._

/**
 * LRCGT implements Functions for L2-regularized Logistic Regression w/ TreeAggregation.
 * f(w) = 1/n \sum_i {l(w, xi, yi)} + 0.5 * lambda * w'w
 * w <- argmin_w f(w), l(w, x, y) = log {1 + exp(- <w,x> y)}
 */
class LRCGT {
  // l(w, x, y) = log(1 + exp(-w'xy))
  // \sum_i {l(w, xi, yi)}, O(nd), treeAggregation
  def functionValue(examples: RDD[Example], w: Vector, lambda: Double): Double = {
    val ZERO = 0.0
    val fObj = examples.treeAggregate(ZERO)(
      seqOp = (c, e) => {
        // c : loss , e : example (x,y)
        var loss = 0.0
        var z = 0.0
        for (fIdx <- e.x) {
          // w'x
          z += w(fIdx)
        }
        z = z * e.y
        if (z >= 0) {
          loss = math.log(1 + math.exp(-z))
        } else {
          // z < 0, log(1 + exp(-z)) = log((1 + exp(z))/exp(z))
          // = -z + log(1 + exp(z))
          loss = -z + math.log(1 + math.exp(z))
        }
        c + loss
      },
      combOp = (c1, c2) => {
        // c : loss
        c1 + c2
      }
    )
    println("functionVal : " + fObj)
    fObj
  }

  // g(w) = lambda * w + \sum_i (sigmoid(w' xi yi) - 1) xi yi, g in R^d, O(nd)
  def gradient(examples: RDD[Example], w: Vector, lambda: Double): Vector = {
    val d = w.length
    val ZERO = Array.fill(d)(0.0)
    val grad = examples.treeAggregate(ZERO)(
      seqOp = (g, e) => {
        // g : grad , e : example (x,y)
        var z = 0.0
        for (fIdx <- e.x) {
          // w'x
          z += w(fIdx)
        }
        // (1.0 / (1 + exp(-y * w^T * x)) - 1.0) * y
        z = (Utils.sigmoid(e.y * z) - 1.0) * e.y
        for (fIdx <- e.x) {
          // (1.0 / (1 + exp(-y * w^T * x)) - 1.0) * y * x_i
          g(fIdx) += z
        }
        g
      },
      combOp = (g1, g2) => {
        // g : grad, todo sparse vector
        for (i <- 0 until g1.length) {
          g1(i) += g2(i)
        }
        g1
      }
    )
    new Vector(grad)
  }

  // H(w) = lambda I_d + X' D X = lambda I_d + \sum_i (sigmoid(1-sigmoid) xi xi'), xi in R^d
  // Dii = sigmoid(w'xiyi)(1 - sigmoid)
  // mu' H(w) mu = lambda * mu'mu + \sum_i{ sigmoid(w'xiyi)(1 -sigmoid) (xi'mu)^2}
  // H(w) mu in R^d, O(nd)
  def muHessianMu(examples: RDD[Example], w: Vector, lambda: Double, mu: Vector): Double = {
    val ZERO = 0.0
    val sHs = examples.treeAggregate(ZERO)(
      seqOp = (s, e) => {
        // s : u H u for x in some partion i, e : example (x,y)
        var z = 0.0
        var wa = 0.0
        for (fIdx <- e.x) {
          // w'x
          z += w(fIdx)
          //  x'mu
          wa += mu(fIdx)
        }
        // 1/(1 + exp(-z))
        val sigma = Utils.sigmoid(e.y * z)
        s + sigma * (1 - sigma) * wa * wa
      },
      combOp = (s1, s2) => {
        // s : u H u for x in some partions
        s1 + s2
      }
    )
    sHs
  }
}

/**
 * CG is used to solve an optimization problem
 */

class ConjugateGradient(val func: LRCGT) {
  def cgBatch(dataInfo: DataInfo, examples: RDD[Example], param: Parameter): Vector = {
    println("mini batch training with CG")
    println("num examples : " + dataInfo.n)
    println("feature dim: " + dataInfo.d)
    println("lambda: " + param.lambda)
    val n = dataInfo.n // num examples
    val d = dataInfo.d // feature dim
    val passes = param.nIter // how many passes of the whole data
    val lambda = param.lambda
    val log = new Logging()
    val exampleArray = shuffleExamples(examples, param, n)
    var w = Vector.zeros(d)
    // average w = 1/T \sum w_t, T total # mini batches
    var what = Vector.zeros(d)
    var b = 0
    var ip = 0
    while (ip < passes) {
      println("data passes: " + ip)
      for (mExamples <- exampleArray) {
        b += 1
        val bcWeight = examples.context.broadcast(what/(1.0*b))
        println("mini batch passes: " + b)
        val batchStart = System.currentTimeMillis()
        w = cgMiniBatchStick(dataInfo, mExamples, param, w, log)
        log.rtime :+= (System.currentTimeMillis() - batchStart) / 1000.0

        //loglik = \sum_i log(1 + exp(-y(i) * X(i,:) * w)) + 0.5 lambda w'w
        log.loglik :+= (0.5 * lambda * (bcWeight.value dot bcWeight.value) + func.functionValue(examples, bcWeight.value, lambda)) / (1.0 * n)
        println("loglik now : " + log.loglik.last)

        what += w
      }
      ip += 1
    }
    Utils.saveVec2File(log.loglik, param.dir + "/loglik")
    Utils.saveVec2File(log.rtime, param.dir + "/rtime")
    Utils.saveVec2File(log.alphas, param.dir + "/alpha")
    Utils.saveVec2File(log.betas, param.dir + "/beta")
    Utils.saveVec2File(log.wDiffNorm, param.dir + "/wdiffnorm")
    Utils.saveVec2File(log.wNorm, param.dir + "/wnorm")
    Utils.saveVec2File(log.w0, param.dir + "/w0")
    Utils.saveVec2File(log.gradNorm, param.dir + "/gradNorm")
    Utils.saveVec2File(log.gtd, param.dir + "/gtd")
    Utils.saveVec2File(log.gtold, param.dir + "/gtold")
    Utils.saveVec2File(log.uHu, param.dir + "/uHu")
    //w
    what / (1.0 * b)
  }

  /**
   * randomly shuffle examples and split into minibatches
   * @param n: num of examples
   */
  def shuffleExamples(examples: RDD[Example], param: Parameter, n: Long): Array[RDD[Example]] = {
    val batchSize = param.batchSize
    val numBatches = n.toInt / batchSize + 1
    println("batch size: " + batchSize + " , num batch: " + numBatches + " , ratio : " + 1.0 / numBatches)
    // ratio for each minibatch
    val ratio = Array.fill(numBatches)(1.0 / numBatches)
    examples.randomSplit(ratio)
  }

  // grad = grad + lamba * (w - w_{t-1}), f = loss + 0.5 * lambda * (w - w_{t-1})^2
  def cgMiniBatchStick(dataInfo: DataInfo, examples: RDD[Example], param: Parameter, weight: Vector, log: Logging): Vector = {
    val n = dataInfo.n // num examples
    val d = dataInfo.d // feature dim
    val lambda = param.lambda
    val iter = param.miter // iteration for each miniBatch
    var nScaling = param.nScaling 
    if (nScaling == 0 ) { 
        nScaling = n.toInt
    }
    println("nScaling: " + nScaling)
    println("num iterations for minibatch: " + iter)
    // start from last minibatch
    var w = weight
    var oldw = Vector.zeros(d)
    var oldGrad = Vector.zeros(d)
    var grad = Vector.zeros(d)
    var mu = Vector.zeros(d) // conjugate dir

    var sHs = 0.0
    var beta = 0.0
    var alpha = 0.0

    var i = 0
    breakable {
      while (i < iter) {
        val iterStart = System.currentTimeMillis()
        val bcWeight = examples.context.broadcast(w)
        println("iter " + i + "..................")
        grad = (func.gradient(examples, bcWeight.value, lambda) + (w - weight) * lambda) / (1.0 * nScaling)
        log.gradNorm :+= Utils.norm2(grad)
        println("gradNorm : " + log.gradNorm.last)
        log.gtold :+= oldGrad dot mu

        if (i == 0 || math.abs(log.gtold.last) < 1e-10) {
          if (i > 0) {
            println("restart to steepest descent..." + log.gtd.last)
          }
          mu = grad * -1.0 // mu_0 = -g_0
          log.betas :+= 0.0
        } else {
          val deltaG = grad - oldGrad
          val muDGrad = mu dot deltaG
          val oldGNorm = oldGrad dot oldGrad
          if (muDGrad == 0 || oldGNorm == 0) {
            // restart
            mu = grad * -1.0
            log.betas :+= 0.0
          } else {
            // Hestenes-Stiefel, beta = g'(g-oldG)/mu'(g - oldG)
            // beta = (grad dot deltaG)/muDGrad
            //beta = g'(g - oldg)/(oldg' oldg), polyak, auto restart
            //beta = (grad dot deltaG) / oldGNorm
            // beta = g'g/oldmu'(g - oldG), Dai and Yuan, faster convergence
            beta = (grad dot grad) / muDGrad
            // mu_k = -g_{k-1} + beta * mu_{k-1}
            if (beta < 0) {
              println("beta is < 0 .......")
            }
            println("beta: " + beta)
            log.betas :+= beta
            //beta = math.max(0, beta);
            mu = grad * -1.0 + mu * beta
          }
        }
        log.gtd :+= mu dot grad
        println("gtd: ........." + log.gtd.last)
        sHs = (func.muHessianMu(examples, bcWeight.value, lambda, mu) + lambda * (mu dot mu))/(1.0 * nScaling)
        println("sHs : ............." + sHs)
        log.uHu :+= sHs
        if (sHs == 0.0) {
          alpha = 0.0
        } else {
          alpha = -1.0 * (grad dot mu) / sHs
        }
        log.alphas :+= alpha
        println("alpha: " + alpha)
        // w = oldw + alpha * mu
        // alpha = -g'mu/mu'Hmu
        w += mu * alpha
        log.w0 :+= w(0)
        log.wNorm :+= Utils.norm2(w)
        println("w norm: " + log.wNorm.last)
        log.wDiffNorm :+= Utils.norm2(w - oldw)
        println("w diff norm: " + log.wDiffNorm.last)

        oldw = w * 1.0
        oldGrad = grad * 1.0

        if (log.wDiffNorm.last < param.eps) {
          println("|w - w*| < " + param.eps + " , stop with iterations: " + i)
          break
        }
        i += 1
      }
    }
    w
  }

  def cgMiniBatch(dataInfo: DataInfo, examples: RDD[Example], param: Parameter, weight: Vector, log: Logging): Vector = {
    val n = dataInfo.n // num examples
    val d = dataInfo.d // feature dim
    val lambda = param.lambda
    val iter = param.miter // iteration for each miniBatch
    var nScaling = param.nScaling 
    if (nScaling == 0 ) { 
        nScaling = n.toInt
    }
    println("nScaling: " + nScaling)
    println("num iterations for minibatch: " + iter)
    // start from last minibatch
    var w = weight
    var oldw = Vector.zeros(d)
    var oldGrad = Vector.zeros(d)
    var grad = Vector.zeros(d)
    var mu = Vector.zeros(d) // conjugate dir

    var sHs = 0.0
    var beta = 0.0
    var alpha = 0.0

    var i = 0
    breakable {
      while (i < iter) {
        val iterStart = System.currentTimeMillis()
        val bcWeight = examples.context.broadcast(w)
        println("iter " + i + "..................")
        grad = (func.gradient(examples, bcWeight.value, lambda) + w * lambda) / (1.0 * nScaling)
        log.gradNorm :+= Utils.norm2(grad)
        println("gradNorm : " + log.gradNorm.last)
        log.gtold :+= oldGrad dot mu

        if (i == 0 || math.abs(log.gtold.last) < 1e-10) {
          if (i > 0) {
            println("restart to steepest descent..." + log.gtd.last)
          }
          mu = grad * -1.0 // mu_0 = -g_0
          log.betas :+= 0.0
        } else {
          val deltaG = grad - oldGrad
          val muDGrad = mu dot deltaG
          val oldGNorm = oldGrad dot oldGrad
          if (muDGrad == 0 || oldGNorm == 0) {
            // restart
            mu = grad * -1.0
            log.betas :+= 0.0
          } else {
            // Hestenes-Stiefel, beta = g'(g-oldG)/mu'(g - oldG)
            // beta = (grad dot deltaG)/muDGrad
            //beta = g'(g - oldg)/(oldg' oldg), polyak, auto restart
            //beta = (grad dot deltaG) / oldGNorm
            // beta = g'g/oldmu'(g - oldG), Dai and Yuan, faster convergence
            beta = (grad dot grad) / muDGrad
            // mu_k = -g_{k-1} + beta * mu_{k-1}
            if (beta < 0) {
              println("beta is < 0 .......")
            }
            println("beta: " + beta)
            log.betas :+= beta
            //beta = math.max(0, beta);
            mu = grad * -1.0 + mu * beta
          }
        }
        log.gtd :+= mu dot grad
        println("gtd: ........." + log.gtd.last)
        sHs = (func.muHessianMu(examples, bcWeight.value, lambda, mu) + lambda * (mu dot mu)) / (1.0 * nScaling)
        println("sHs : ............." + sHs)
        log.uHu :+= sHs
        if (sHs == 0.0) {
          alpha = 0.0
        } else {
          alpha = -1.0 * (grad dot mu) / sHs
        }
        log.alphas :+= alpha
        println("alpha: " + alpha)
        // w = oldw + alpha * mu
        // alpha = -g'mu/mu'Hmu
        w += mu * alpha
        log.w0 :+= w(0)
        log.wNorm :+= Utils.norm2(w)
        println("w norm: " + log.wNorm.last)
        log.wDiffNorm :+= Utils.norm2(w - oldw)
        println("w diff norm: " + log.wDiffNorm.last)

        oldw = w * 1.0
        oldGrad = grad * 1.0

        if (log.wDiffNorm.last < param.eps) {
          println("|w - w*| < " + param.eps + " , stop with iterations: " + i)
          break
        }
        i += 1
      }
    }
    w
  }

  def cg(dataInfo: DataInfo, examples: RDD[Example], param: Parameter): Vector = {
    val n = dataInfo.n // num examples
    val d = dataInfo.d // feature dim
    val lambda = param.lambda
    val nIter = param.nIter
    // scaling factor for grad and sHs, usually n
    var nScaling = param.nScaling 
    if (nScaling == 0 ) { 
        nScaling = n.toInt
    }
    println("nScaling: " + nScaling)
    println("num examples : " + n)
    println("feature dim: " + d)
    println("lambda: " + lambda)
    println("num iterations: " + nIter)
    var w = Vector.zeros(d)
    var oldw = Vector.zeros(d)
    var oldGrad = Vector.zeros(d)
    var grad = Vector.zeros(d)
    var mu = Vector.zeros(d) // conjugate dir
    var loglik = Array.fill(nIter)(0.0)
    var rtime = Array.fill(nIter)(0.0)
    var alphas = Array.fill(nIter)(0.0)
    var betas = Array.fill(nIter)(0.0)
    var wNorm = Array.fill(nIter)(0.0)
    var wDiffNorm = Array.fill(nIter)(1.0)
    var w0 = Array.fill(nIter)(0.0)
    var gradNorm = Array.fill(nIter)(0.0)
    var gtd = Array.fill(nIter)(0.0) // g'mu
    var uHu = Array.fill(nIter)(0.0) // u' H u
    var gtold = Array.fill(nIter)(0.0) // g'oldmu

    var sHs = 0.0
    var beta = 0.0
    var alpha = 0.0

    var i = 0
    breakable {
      while (i < nIter) {
        val iterStart = System.currentTimeMillis()
        val bcWeight = examples.context.broadcast(w)
        println("iter " + i + "..................")
        grad = (func.gradient(examples, bcWeight.value, lambda) + w * lambda)/ (1.0 * nScaling)
        gtold(i) = oldGrad dot mu
        if (i == 0 || math.abs(gtold(i)) < 1e-10) {
          if (i > 0) {
            println("restart to steepest descent..." + gtd(i))
          }
          mu = grad * -1.0 // mu_0 = -g_0
          betas(i) = 0.0
        } else {
          val deltaG = grad - oldGrad
          val muDGrad = mu dot deltaG
          val oldGNorm = oldGrad dot oldGrad
          if (muDGrad == 0 || oldGNorm == 0) {
            // restart
            mu = grad * -1.0
            betas(i) = 0.0
          } else {
            // Hestenes-Stiefel, beta = g'(g-oldG)/mu'(g - oldG)
            // beta = (grad dot deltaG)/muDGrad
            //beta = g'(g - oldg)/(oldg' oldg), polyak, auto restart
            //beta = (grad dot deltaG) / oldGNorm
            // beta = g'g/oldmu'(g - oldG), Dai and Yuan, faster convergence
            beta = (grad dot grad) / muDGrad
            // mu_k = -g_{k-1} + beta * mu_{k-1}
            if (beta < 0) {
              println("beta is < 0 .......")
            }
            println("beta: " + beta)
            betas(i) = beta
            //beta = math.max(0, beta);
            mu = grad * -1.0 + mu * beta
          }
        }
        gtd(i) = mu dot grad
        println("gtd: ........." + gtd(i))
        sHs = (func.muHessianMu(examples, bcWeight.value, lambda, mu) + lambda * (mu dot mu))/(1.0 * nScaling)
        println("sHs : ............." + sHs)
        uHu(i) = sHs
        if (sHs == 0.0) {
          alpha = 0.0
        } else {
          alpha = -1.0 * (grad dot mu) / sHs
        }
        alphas(i) = alpha
        println("alpha: " + alpha)
        // w = oldw + alpha * mu
        // alpha = -g'mu/mu'Hmu
        w += mu * alpha
        wNorm(i) = Utils.norm2(w)
        println("w norm: " + wNorm(i))
        wDiffNorm(i) = Utils.norm2(w - oldw)
        println("w diff norm: " + wDiffNorm(i))
        //loglik = \sum_i log(1 + exp(-y(i) * X(i,:) * w)) + 0.5 lambda w'w
      //  loglik(i) = (0.5 * lambda * (w dot w) + func.functionValue(examples, bcWeight.value, lambda)) / (1.0 * n)
      //  println("loglik now : " + loglik(i))
        oldw = w * 1.0
        //oldGrad = grad
        oldGrad = grad * 1.0
        gradNorm(i) = Utils.norm2(grad)
        println("gradNorm : " + gradNorm(i))
        w0(i) = w(0)
        rtime(i) = (System.currentTimeMillis() - iterStart) / 1000.0

        if (wDiffNorm(i) < param.eps) {
          println("|w - w*| < " + param.eps + " , stop with iterations: " + i)
          break
        }
        i += 1
      }
    }
    Utils.saveVec2File(loglik.slice(0, i), param.dir + "/loglik")
    Utils.saveVec2File(rtime.slice(0, i), param.dir + "/rtime")
    Utils.saveVec2File(alphas.slice(0, i), param.dir + "/alpha")
    Utils.saveVec2File(betas.slice(0, i), param.dir + "/beta")
    Utils.saveVec2File(wDiffNorm.slice(0, i), param.dir + "/wdiffnorm")
    Utils.saveVec2File(wNorm.slice(0, i), param.dir + "/wnorm")
    Utils.saveVec2File(w0.slice(0, i), param.dir + "/w0")
    Utils.saveVec2File(gradNorm.slice(0, i), param.dir + "/gradNorm")
    Utils.saveVec2File(gtd.slice(0, i), param.dir + "/gtd")
    Utils.saveVec2File(gtold.slice(0, i), param.dir + "/gtold")
    Utils.saveVec2File(uHu.slice(0, i), param.dir + "/uHu")
    w
  }

}
