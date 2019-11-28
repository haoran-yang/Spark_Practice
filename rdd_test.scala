import org.apache.spark.{ SparkContext, SparkConf}
import org.apache.spark.SparkContext._

object rdd1{
    def main(args: Array[String]){
        val conf = new SparkConf().setAppName("rdd01")
        val sc = new SparkContext(conf)

        val data1 = sc.textFile("./data/lpsa.data")

        val rdd1 = sc.parallelize(1 to 20, 2) // 转换为rdd
        val rdd2 = rdd1.map(x=> x*2) // 一对一映射
        val rdd3 = rdd1.filter(x=> x>10) // 过滤
        val rdd4 = rdd1.flatMap(x=> x to 20) // func需满足一对多映射

    }
}