package baidu.infra.cg

/*
 * Program entrance point to generate the dimensionality reduction dictionary
 * Original adfea data version
 * Including show/click and tab handling
 */

import java.io.FileWriter
import org.apache.spark.{SparkConf, SparkContext}

object DimensionReduction {
  var filterThreshold = 1000
  var inputPath : String = "/Users/tianbingxu/work/cg_dense_1/data/tusou/part-00006"
  var outputPath : String = "/Users/tianbingxu/work/cg_dense_1/data/dict/DR_dict"
  def main(args: Array[String]) {
    for (i <- 0 to args.length - 1 by 2) {
      if (args(i)(0) == '-') {
        args(i)(1) match {
          case 'i' => {
            inputPath = args(i + 1)
          }
          case 'o' => {
            outputPath = args(i + 1)
          }
          case 'f' => {
            filterThreshold = args(i + 1).toInt
          }
          case _ => {
            System.err.println("ERROR: unknown option")
            System.exit(1)
          }
        }
      }
    }
    println("dict input path : " + inputPath)
    println("dict output path : " + outputPath)
    println("frequency cut-off threshold : " + filterThreshold)
    genDRDict()
  }

  def genDRDict(): Unit = {
    // Comment out these two lines if running in spark-shell since sc is already defined by the shell
    val sparkConf = new SparkConf().setAppName("DimensionReductionOperations")
    val sc = new SparkContext(sparkConf)

    val origInput = sc.textFile(inputPath)

    println(s"The total number of partitions to be processed : ${origInput.partitions.size}")

    // Compute unmerged histogram on each executor
    val initHis = origInput.flatMap(line => {
      // Segments of each record are separated by tabs
      val segs = line.split("\t")

      // Extract the show, click and feature list for the first segment
      val showClickFea = segs(0).split(" ")
      val show1 = java.lang.Integer.parseInt(showClickFea(0))
      val click1 = java.lang.Integer.parseInt(showClickFea(1))
      val fea1 = showClickFea.drop(2)

      // If there is only one segment, then itself will generate multiple samples, otherwise,
      // each of the remaining segments will combine with the first segment to produce multiple samples, while
      // the first segment itself won't be processed.
      val initHis = scala.collection.mutable.Map[Long, Int]()
      if (segs.size == 1) {
        fea1.foreach(fea => {
          val pair1 = fea.split(":")
          initHis += (java.lang.Long.parseUnsignedLong(pair1(0))-> show1)
        })
      } else {
        for (i <- 1 to (segs.size-1)) {
          val cur_showClickFea = segs(i).split(" ")
          val cur_show = java.lang.Integer.parseInt(cur_showClickFea(0))
          val cur_click = java.lang.Integer.parseInt(cur_showClickFea(1))
          val cur_fea = cur_showClickFea.drop(2)
          fea1.foreach(fea => {
            val cur_pair = fea.split(":")
            val commonFS = java.lang.Long.parseUnsignedLong(cur_pair(0))
            // In case a common feature signature has been updated before by the previsou segment of the same line
            var addon = 0
            if (initHis contains commonFS) {
              addon = initHis(commonFS)
            }
            initHis += (commonFS -> (cur_show + addon))
          })
          cur_fea.foreach(fea => {
            val cur_pair = fea.split(":")
            initHis += (java.lang.Long.parseUnsignedLong(cur_pair(0))-> cur_show)
          })
        }
      }
      initHis
    })

    // Aggregated histogram on the driver and apply the threshold to filter out uncommon fea sign
    //val finalHis = initHis.countByKey()
    val finalHis = initHis.reduceByKey((a, b) => a + b)
    val filteredHis = finalHis.filter{case (feaSign, num) => {num > filterThreshold}}.collect()

    // Assign a new dim for each remaining fea sign
    // Note: this could possibly be done on executor through accumulator?
    // so that we don't have two immutable copies on driver.
    var curDim = 0
    val dimDict = filteredHis.map{case (feaSign, num) => {
      curDim += 1
      (feaSign, curDim)
    }}

    // Save the filtered histogram to local file
    val file_his = new FileWriter(outputPath + "_his")
    filteredHis.foreach{case (feaSign, num) => {
      file_his.write(java.lang.Long.toUnsignedString(feaSign))
      file_his.write(" ")
      file_his.write(num.toString)
      file_his.write(System.getProperty("line.separator"))
    }}
    file_his.flush()
    file_his.close()

    // Save the dictionary to a local file
    val file = new FileWriter(outputPath)
    dimDict.foreach{case (feaSign, dim) => {
      file.write(java.lang.Long.toUnsignedString(feaSign))
      file.write(" ")
      file.write(dim.toString)
      file.write(System.getProperty("line.separator"))
    }}
    file.flush()
    file.close()

    println(s"All done. The generated dictionary is save to $outputPath .")

  }

}
