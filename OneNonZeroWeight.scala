import org.apache.spark.mllib.linalg.{VectorUDT, Vectors}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.{SparkContext, SparkConf}

object OneNonZeroWeight {

  var sc: SparkContext = _
  var sqlContext: SQLContext = _

  def fitLogisticRegression(rows: List[Row]): Unit = {
    println("ROWS: " + rows)
    val schema = StructType(Seq(
      StructField("label", DoubleType),
      StructField("features", new VectorUDT),
      StructField("weight", DoubleType)))
    val rdd = sc.parallelize(rows)
    val df = sqlContext.createDataFrame(rdd, schema)

    val logisticRegression = new LogisticRegression().setWeightCol("weight")
    val logisticRegressionTrainedModel = logisticRegression.fit(df)
  }

  def main(args: Array[String]): Unit = {
    println("Initializing Spark")
    val sparkConf: SparkConf = new SparkConf()
      .setMaster("local[4]")
      .setAppName("OneNonZeroWeight")
      .set("spark.ui.enabled", "false")
    sc = new SparkContext(sparkConf)
    sqlContext = new SQLContext(sc)

    val rowsSuccess = List(
      Row(1.0, Vectors.dense(16.0, 11.0, 9.0), 1.0),
      Row(0.0, Vectors.dense(32.0, 11.0, 9.0), 0.0)
    )
    println("            rowsSuccess???")
    fitLogisticRegression(rowsSuccess)
    println("            rowsSuccess!!!")

    val rowsFail = List(
      Row(1.0, Vectors.dense(16.0, 11.0, 9.0), 0.0),
      Row(0.0, Vectors.dense(32.0, 11.0, 9.0), 1.0)
    )
    println("            rowsFail???")
    // It will throw exception here: "java.lang.ArrayIndexOutOfBoundsException: 1"
    fitLogisticRegression(rowsFail) 
    println("            rowsFail!!!")

    sc.stop()
  }
}
