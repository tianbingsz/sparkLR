package baidu.infra.cg

import java.io.{BufferedReader, FileReader}


class FeatureMapping extends Serializable {
  var featureMap = scala.collection.mutable.Map[Long, Int]()
  // todo, only featureMap.toArray with Long, save map space
  var revFeatureMap = scala.collection.mutable.Map[Int, Long]()

  def loadFeatureMap(mapFileName: String) {
    val dictFile = new FileReader(mapFileName)
    val fileBuffer = new BufferedReader(dictFile)
    var line = fileBuffer.readLine()
    // todo, try-catch
    while (line != null) {
      val parsedLine = line.split(" ")
      if (parsedLine.size == 2) {
        //featureMap += (java.lang.Long.parseUnsignedLong(parsedLine(0))
          //-> java.lang.Integer.parseInt(parsedLine(1)))
        //revFeatureMap += (java.lang.Integer.parseInt(parsedLine(1))
         // -> java.lang.Long.parseUnsignedLong(parsedLine(0)))
        featureMap += (java.lang.Long.parseUnsignedLong(parsedLine(0))
          -> parsedLine(1).toInt)
        revFeatureMap += (parsedLine(1).toInt
          -> java.lang.Long.parseUnsignedLong(parsedLine(0)))
      }
      line = fileBuffer.readLine()
    }
    fileBuffer.close()
    dictFile.close()
  }

  def printFeatureMap() {
    println("print feature map....")
    featureMap foreach { m => println(m._1 + "-->" + m._2) }
    println("print reverse feature map....")
    revFeatureMap foreach { m => println(m._1 + "-->" + m._2) }
  }

}
