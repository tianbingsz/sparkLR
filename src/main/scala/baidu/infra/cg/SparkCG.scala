package baidu.infra.cg

import org.apache.spark.rdd.RDD
import org.apache.spark.util.Vector
import org.apache.spark.{SparkConf, SparkContext}

object SparkCG {

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("cg-lr")
    val sc = new SparkContext(conf)

    val param = new Parameter()
    var dataInfo = new DataInfo()
    for (i <- 0 to args.length - 1 by 2) {
      if (args(i)(0) == '-') {
        args(i)(1) match {
          case 'e' => {
            param.eps = args(i + 1).toDouble
            println("param.eps : " + param.eps)
          }
          case 'c' => {
            param.lambda = args(i + 1).toDouble
            println("param.lambda: " + param.lambda)
          }
          case 'n' => {
            param.nIter = args(i + 1).toInt
            println("param.nIter: " + param.nIter)
          }
          case 'd' => {
            param.dir = args(i + 1)
            println("param.dir: " + param.dir)
          }
          case 'i' => {
            param.train = args(i + 1)
            println("param.train: " + param.train)
          }
          case 't' => {
            param.test = args(i + 1)
            println("param.test: " + param.test)
          }
          case 'm' => {
            param.miter = args(i + 1).toInt
            println("param.miter: " + param.miter)
          }
          case 'b' => {
            param.batchSize = args(i + 1).toInt
            println("param.batchSize: " + param.batchSize)
          }
          case 'z' => {
            param.dict = args(i + 1)
            println("param.dict: " + param.dict)
          }
          case 's' => {
            param.nScaling = args(i + 1).toInt
            println("param.nScaling: " + param.nScaling)
          }
          case _ => {
            System.err.println("ERROR: unknown option")
            System.exit(1)
          }
        }
      }
    }

    val model = Train(sc, param)
    saveModel(model, param)
    Test(sc, param, model)
    sc.stop()
  }

  private def Train(sc: SparkContext, param: Parameter): Model = {
    println("loading training dataset ................")
    val startLoad = System.currentTimeMillis()
    //val examples = Utils.loadData(sc, param.train)
    val examples = Utils.LoadAdfeaFeatures(sc, param.train, param.dict)
    val dataInfo = new DataInfo()
    dataInfo.getStats(examples.cache())
    println("Loading time: " + (System.currentTimeMillis() - startLoad) / 1000.0 + " seconds")

    println("Training model ..........................")
    val startTrain = System.currentTimeMillis()
    val model: Model = trainModel(dataInfo, examples, param)
    println("Training time: " + (System.currentTimeMillis() - startTrain) / 1000.0 + " seconds")

    val accuracy = model.testAccuracy(examples)
    println("Training Accuracy: " + accuracy)
    val auc = model.AUCROC(examples)
    println("Training Area under ROC = " + auc)

    model
  }

  private def trainModel(dataInfo: DataInfo, examples: RDD[Example], param: Parameter): Model = {
    val model: Model = new Model(param)
    model.w = train(dataInfo, examples, param)
    model
  }

  private def train(dataInfo: DataInfo, examples: RDD[Example], param: Parameter): Vector = {
    var w: Vector = null
    val solver = new ConjugateGradient(new LRCGT())
  //  w = solver.cgBatch(dataInfo, examples, param)
    w = solver.cg(dataInfo, examples, param)
    w
  }

  private def Test(sc: SparkContext, param: Parameter, model: Model) = {
    println("loading test dataset ................")
    val startLoad = System.currentTimeMillis()
    //    val examples = Utils.loadData(sc, param.test).cache()
    val examples = Utils.LoadAdfeaFeatures(sc, param.test, param.dict).cache()
    examples.count()
    println("Loading time: " + (System.currentTimeMillis() - startLoad) / 1000.0 + " seconds")

    // Evaluate model on test examples and compute test error
    val accuracy = model.testAccuracy(examples)
    println("Test Accuracy: " + accuracy)
    val auc = model.AUCROC(examples)
    println("Test Area under ROC = " + auc)
  }

  private def saveModel(model: Model, param: Parameter) {
    model.saveModel(param.dir + "/model")
    val fMap = new FeatureMapping()
    fMap.loadFeatureMap(param.dict)
    model.saveModel(param.dir + "/modelMap", fMap.revFeatureMap)
  }

}
