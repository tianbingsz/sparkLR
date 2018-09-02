package baidu.infra.cg

/**
 * x :  <idx, val>, but all val = 1.0, so we only store Array[idx]
 */
class Example(val x: Array[Int], val y: Double) extends Serializable {
  def getMaxIndex(): Int = {
    this.x.last
  }

  def printExample() {
    print(y + ", ")
    for (fIdx <- x) {
      print(fIdx + ":" + "1.0 ")
    }
    println
  }
}
