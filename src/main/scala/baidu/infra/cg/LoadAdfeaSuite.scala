package baidu.infra.cg

import java.io.FileWriter

import org.apache.spark.{SparkConf, SparkContext}

object loadAdfeaSuite {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("load-adfea")
    val sc = new SparkContext(conf)
    val param = new Parameter()
    TestLoadAdfea(sc, param)
  }

  def TestLoadAdfea(sc: SparkContext, param: Parameter): Unit = {
    val rddResult = Utils.LoadAdfeaFeatures(sc, param.train, param.dict)
    val localResult = rddResult.collect()
    val sampleFile = new FileWriter("/tmp/samples")
    localResult.foreach(oneSample => {
      sampleFile.write(java.lang.Double.toString(oneSample.y))
      sampleFile.write(System.getProperty("line.separator"))
      sampleFile.write(oneSample.x.mkString(" "))
      sampleFile.write(System.getProperty("line.separator"))
      sampleFile.write(System.getProperty("line.separator"))
    })
    sampleFile.flush()
    sampleFile.close()
  }

}
