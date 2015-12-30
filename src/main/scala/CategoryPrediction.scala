
import au.com.bytecode.opencsv.CSVParser
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row.fromSeq
import org.apache.spark.sql.types.{StructType, StringType, StructField}
import org.apache.spark.sql.{Row, DataFrame, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}


object CategoryPrediction {

  val TRAIN_PATH = "data/in/train.csv"

  val TEST_PATH = "data/in/test.csv"

  val RESULT_PATH ="data/out"

  val conf = new SparkConf().setAppName("CategoryPrediction").setMaster("local[4]")

  val sc = new SparkContext(conf)

  def main(args: Array[String]) {



    val trainDF = loadAndParseData(TRAIN_PATH, isTrain = true)
    val indexedTrainDF = indexedCategory(trainDF, indexedFeatures(trainDF))
    val model = train(indexedTrainDF)

    val testDF = loadAndParseData(TEST_PATH, isTrain = false)
    val indexedTestDF = indexedFeatures(testDF)
    val result: RDD[Vector] = predict(model, indexedTestDF)

    val map: RDD[Row] = result.map(line => fromSeq(line.toArray.map(d => "%.3f".format(d))))
    val sqlContext = new SQLContext(sc)

    val cols : Seq[StructField] = (0 until map.first().size).map(i => StructField("c" + i, StringType, nullable = false))
    val dataFrame: DataFrame = sqlContext.createDataFrame(map, StructType(cols))


    dataFrame.repartition(1)
            .write.format("com.databricks.spark.csv")
            .option("header", "false")
            .save(RESULT_PATH)

    System.exit(0)
  }

  def loadAndParseData(path: String, isTrain: Boolean) : DataFrame ={

    val inputFile: RDD[String] = sc.textFile(path)
    val header: String = inputFile.first()
    val filtered = inputFile.filter(line=> line != header)


    if (isTrain) {

      val processedRDD = filtered.map(line => line.split(",")).map(value => Crime(value(0), value(1), value(2), value(3), value(4), value(5), value(6), value(7).toDouble, value(8).toDouble))
      val sqlContext = new SQLContext(sc)
      import sqlContext.implicits._
      processedRDD.toDF()
    }
    else{

      val processedRDD = filtered.map(line => line.split(",")).map(value => TestCrime(value(0), value(1), value(2), value(3), value(4), value(5).toDouble, value(6).toDouble))
      val sqlContext = new SQLContext(sc)
      import sqlContext.implicits._
      processedRDD.toDF()
    }


  }

  def indexedFeatures(dataFrame: DataFrame) : DataFrame = {

    val dayOfWeekIndexer = new StringIndexer()
      .setInputCol("dayOfWeek")
      .setOutputCol("dayOfWeekIndex")

    val indexedDayOfWeek = dayOfWeekIndexer.fit(dataFrame).transform(dataFrame)

    val districtIndexer = new StringIndexer()
      .setInputCol("pdDistrict")
      .setOutputCol("districtIndex")

    districtIndexer.fit(dataFrame).transform(indexedDayOfWeek)
  }

  def indexedCategory(initial: DataFrame, dataFrame: DataFrame) : DataFrame = {

    val categoryIndexer = new StringIndexer()
      .setInputCol("category")
      .setOutputCol("categoryIndex")

    categoryIndexer.fit(initial).transform(dataFrame)
  }

  def train(indexedTrainDF: DataFrame) : NaiveBayesModel = {

    val trainSet: RDD[LabeledPoint] = indexedTrainDF.map(line => LabeledPoint(line(11).asInstanceOf[Double], Vectors.dense(line(9).asInstanceOf[Double], line(10).asInstanceOf[Double])))
    NaiveBayes.train(trainSet, lambda = 1.0, modelType = "multinomial")
  }

  def predict(model: NaiveBayesModel, indexedTestData: DataFrame) : RDD[Vector] = {

    val testSet: RDD[Vector] = indexedTestData.map(line => Vectors.dense(line(7).asInstanceOf[Double], line(8).asInstanceOf[Double]))
    model.predictProbabilities(testSet)
  }

  case class Crime(date: String, category: String, descript: String, dayOfWeek: String, pdDistrict: String, resolution: String, address: String, x: Double, y: Double)

  case class TestCrime(id: String, date: String, dayOfWeek: String, pdDistrict: String, address: String, x: Double, y: Double)
}