from pyspark.sql import SparkSession
from pyspark.mllib.evaluation import  RankingMetrics
from pyspark.ml.recommendation import ALS
from time import time
from pyspark import SparkContext
from pyspark.ml.evaluation import RegressionEvaluator

def als_grid_search_ranking(train,val,maxIters,regParams, ranks):

    '''
    GRID SEARCH FOR ALS
    Params:
	train: train set
        val: validation set
        maxIters: list of maxIter
        regParams: list of regParams
        ranks: list ofranks

    return:
        models: dic of models where key is its parameter and value is the ALS obj
        precision_at_k_scores: dic of precision_at_k_scores
        maps: dic of mean avearage precision
        NDCGs: dict of NDCG scores
        times: dict of time to run diff models
    '''
    models = {}
    rmse_dic = {}
    #precision_at_k_scores = {}
    #maps ={}
    #NDCGs = {}
    times = {}

    # grid-search
    for r in ranks:
        for Iter in maxIters:
            for reg in regParams:
                st = time()
                # initialize and train model
                model = ALS(rank = r,maxIter=Iter, regParam=reg,userCol='user_id',\
                            itemCol='book_id',ratingCol='rating',coldStartStrategy='drop',nonnegative=True)
                model = model.fit(train)
                models[(r,Iter,reg)] = model

                # evaluate on validation
                evaluator=RegressionEvaluator(metricName="rmse",labelCol="rating",predictionCol="prediction")
                predictions=model.transform(val)
                rmse=evaluator.evaluate(predictions)

               # preds = model.recommendForAllUsers(500).collect()
               # dic = groud_truth_dict(val)
               # dic2 = predict_dict(preds)
               # preAndTrue = predictAndTruth(dic,dic2)

               # ranking_obj = RankingMetrics(preAndTrue)
               # precision_at_k_scores[(r,Iter,reg)] = ranking_obj.precisionAt(500)
               # maps[(r,Iter,reg)] = ranking_obj.meanAveragePrecision
               # NDCGs[(r,Iter,reg)] = ranking_obj.ndcgAt(500)
                time_t = round(time() - st,5)
                times[(r,Iter,reg)] = time_t
                rmse_dic[(r,Iter,reg)] = rmse
                print('Model with maxIter = {}, reg = {}, rank = {} complete'.format(Iter,reg,r))
                print('RMSE:', str(rmse))
                print('Time:', str(time_t))
    return models, rmse_dic, times



def groud_truth_dict(df):
    '''
    params:
    df: a spark dataframe with user_id, rating and book_id in it

    return:
    dic: a dictioanry where key is the user_id and \
        value is a list containing all the ground truth book_id sorted by rating

    '''
    spark = SparkSession.builder.appName("try").getOrCreate()
    df.createOrReplaceTempView('file')
    df = spark.sql('SELECT user_id, book_id FROM file SORT BY user_id, rating DESC').collect()
    df = [[i[0],i[1]] for i in df]
    dic = {}
    for ele in df:
        # if user_id not in dictioanry, append
        if ele[0] not in dic:
            dic[ele[0]] = [ele[1]]
        else:
            dic[ele[0]] += [ele[1]]

    return dic

def predict_dict(preds):
    '''
    params:
    df: model.recommendForAllUsers(n).collect()

    return:
    dic: a dictioanry where key is the user_id and \
        value is a list containing all the prediction book_id sorted by rating

    '''
    dic2 = {}
    for i in preds:
        dic2[i.user_id] = []
        for row in i.recommendations:
            dic2[i.user_id] += [row.book_id]

    return dic2

def predictAndTruth(dic,dic2):
    '''
    params:
	dic: dictionary of predictions
        dic2: dictionary of ground truth

    return:
	list: a list of tuple for Rankingmetrics

    '''
    sc = SparkContext.getOrCreate()
    predAndTruth = []
    for user in dic.keys():
        gt = dic[user]
        pred_v = dic2[user]
        predAndTruth += [(pred_v,gt)]
    predAndTruth = sc.parallelize(predAndTruth)
    return predAndTruth
