import os

import zipfile
from numpy.random import choice
from math import exp
from itertools import product

from pyspark import Row
from pyspark import SparkContext

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from pyspark.sql.functions import monotonically_increasing_id, udf, col, desc, round as dfround
from pyspark.sql.types import DoubleType

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder


def get_count(df, col):
    return df.select([col]).groupBy(col).count().count()


def evaluate_ALS(train, test, kwargs):
    # coldStartStrategy parameter to “drop” in order
    # to drop any rows in the DataFrame of predictions that contain NaN values.
    als = ALS(**kwargs)
    param_grid = ParamGridBuilder() \
        .addGrid(als.rank, list(range(12, 15))) \
        .addGrid(als.regParam, [i * 0.01 for i in range(1, 17, 5)]) \
        .build()

    evaluator = RegressionEvaluator(metricName="rmse", labelCol=kwargs["ratingCol"],
                                    predictionCol="prediction")
    tvs = TrainValidationSplit(
        estimator=als,
        estimatorParamMaps=param_grid,
        evaluator=evaluator
    )
    model = tvs.fit(train).bestModel

    train_preds = model.transform(train)
    test_preds = model.transform(test)
    print("Best model: {0} rank, {1} reg param".format(model.rank,
                                                       model._java_obj.parent().getRegParam()))
    print("train RMSE = {0}".format(evaluator.evaluate(train_preds)))
    print("test RMSE = {0}".format(evaluator.evaluate(test_preds)))

    return model


def get_recommendations(model, movies):
    # Generate top 10 movie recommendations for each user
    userRecs = model.recommendForAllUsers(10).rdd.flatMapValues(lambda x: x) \
        .map(lambda x: Row(userId=x[0], **x[1].asDict())) \
        .toDF() \
        .join(movies, on="movieId") \
        .select(["userId", "movieId", "title", "rating"]) \
        .sort(["userId", desc("rating")])

    # Generate top 10 user recommendations for each movie
    movieRecs = model.recommendForAllItems(10).rdd.flatMapValues(lambda x: x) \
        .map(lambda x: Row(movieId=x[0], **x[1].asDict())) \
        .toDF() \
        .join(movies, on="movieId") \
        .select(["movieId", "userId", "title", "rating"]) \
        .sort(["movieId", desc("rating")])

    return userRecs, movieRecs


