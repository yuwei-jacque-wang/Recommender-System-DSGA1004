from pyspark.sql import SparkSession
from pyspark.mllib.evaluation import  RankingMetrics
from pyspark.ml.recommendation import ALS
from time import time 
from pyspark import SparkContext
from pyspark.sql.functions  import collect_list

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
    precision_at_k_scores = {}
    maps ={}
    NDCGs = {}
    times = {}
    sc = SparkContext.getOrCreate()
    spark = SparkSession.builder.appName("try").getOrCreate()
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
                preds = model.recommendForAllUsers(500)
                preds.createOrReplaceTempView('preds')
                val = spark.sql('SELECT user_id, book_id FROM val SORT BY rating DESC')
                val = val.groupBy('user_id').agg(collect_list('book_id').alias('book_id_val'))
                val.createOrReplaceTempView('val')
                predAndTruth = spark.sql('SELECT preds.recommendations, val.book_id_val FROM val join preds on preds.user_id = val.user_id')
                predAndTruth = predAndTruth.collect()
                final_predAndTruth = []
                for item in predAndTruth:
                    truth = item[1]
                    pred = [i.book_id for i in item[0]]
                    final_predAndTruth += [(pred,truth)]
                    
                    
                final_predAndTruth =  sc.parallelize(final_predAndTruth)
                
                ranking_obj = RankingMetrics(final_predAndTruth)
                precision_at_k_scores[(r,Iter,reg)] = ranking_obj.precisionAt(500)
                maps[(r,Iter,reg)] = ranking_obj.meanAveragePrecision
                NDCGs[(r,Iter,reg)] = ranking_obj.ndcgAt(500)
                times[(r,Iter,reg)] = round(time() - st,5)
                
                print('Model with maxIter = {}, reg = {}, rank = {} complete'.format(Iter,reg,r))
    return models, precision_at_k_scores,maps, NDCGs,times



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


