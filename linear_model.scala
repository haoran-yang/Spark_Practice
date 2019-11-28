import org.apache.spark.mllib.regression.{LinearRegressionModel,LinearRegressionWithSGD} //线性回归库
import org.apache.spark.{ SparkContext, SparkConf} //环境配置库
import org.apache.spark.mllib.linalg.Vectors //向量
import org.apache.spark.mllib.regression.LabeledPoint //标注点
import org.apache.spark.mllib.util.MLUtils //用于加载和解析数据

object LinearRegression{
    def main(args: Array[String]){
        // 构建spark对象
        val conf = new SparkConf().setAppName("LinearRegression")
        val sc = new SparkContext(conf)
        //读取数据
        val data = sc.textFile("./data/lpsa.data")
        // 格式转换
        val datas = data.map{line=>
                    val parts = line.split(',')
                    LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
        }.cache()

        // 模型训练,参数依次为样本，迭代次数，步长，迭代因子
        val model = LinearRegressionWithSGD.train(input=datas,numIterations=100,stepSize=1,miniBatchFraction=1.0)
        // 在训练数据上预测 
        val pred = model.predict(datas.map(_.features))
        // 预测值和真实标签合并
        val pred_label = pred.zip(datas.map(_.label))
        // 打印前10个
        val pred_label10 = pred_label.take(10)
        println("predict_label" + "\t" + "true_label")
        for(i <- 0 to pred_label10.length-1 ) {
            println(pred_label10(i)._1 + "\t" + pred_label10(i)._2) 
        }

        // 误差平方和
        val loss = pred_label.map{
            case(p,l) =>
            val diff = p-l
            diff*diff
        }.reduce(_+_)
        // 均方误差
        val rmse = math.sqrt(loss/datas.count)
        println(s"rmse is $rmse")

        // 模型存储和加载
        model.save(sc,"./LinerGression")
        val model2 = LinearRegressionModel.load(sc,"./LinerGression")
    }
}
