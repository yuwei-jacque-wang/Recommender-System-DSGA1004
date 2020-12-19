def data_split_subsampling(spark,  partial):
    '''Perform data subsampling and splitting on interaction data for basic recommender system

    This function returns three dataframes corresponding to training, validation
    and test

    Parameters
    ----------
    spark : spark session object

    partial : float
        The subsampling rate


    '''
    # read data
    interaction = spark.read.parquet('hdfs:/user/yx2090/interaction.parquet')

    # subsampling
    interaction = interaction.sample(False, partial, seed = 2020)
    interaction.createOrReplaceTempView('interaction')

    # delete rating = 0
    rating_interaction = spark.sql('SELECT * FROM interaction WHERE rating != 0')
    rating_interaction.createOrReplaceTempView('rating_interaction')

    # select users with 10 or more valid interactions
    user = spark.sql('SELECT DISTINCT(user_id) FROM rating_interaction GROUP BY user_id HAVING count(*) >= 10')
    user.createOrReplaceTempView('user')

    # 60% training users, 20% val users, 20% test users
    train_user, val_user, test_user = user.randomSplit([0.6, 0.2, 0.2], seed = 2020)
    train_user.createOrReplaceTempView('train_user')
    val_user.createOrReplaceTempView('val_user')
    test_user.createOrReplaceTempView('test_user')

    training_from_train = spark.sql('SELECT * FROM rating_interaction WHERE user_id IN (SELECT user_id FROM train_user)')
    training_from_train.createOrReplaceTempView('training_from_train')
    all_val_interaction = spark.sql('SELECT * FROM rating_interaction WHERE user_id IN (SELECT user_id FROM val_user)')
    all_test_interaction = spark.sql('SELECT * FROM rating_interaction WHERE user_id IN (SELECT user_id FROM test_user)')

    # select half interaction from val and test interactions to training
    # val
    all_val_interaction_rdd = all_val_interaction.rdd.zipWithIndex()
    all_val_interaction_rdd_final = all_val_interaction_rdd.toDF()
    all_val_interaction_rdd_final = all_val_interaction_rdd_final.withColumn('user_id', all_val_interaction_rdd_final['_1'].getItem("user_id"))
    all_val_interaction_rdd_final = all_val_interaction_rdd_final.withColumn('book_id', all_val_interaction_rdd_final['_1'].getItem("book_id"))
    all_val_interaction_rdd_final = all_val_interaction_rdd_final.withColumn('is_read', all_val_interaction_rdd_final['_1'].getItem("is_read"))
    all_val_interaction_rdd_final = all_val_interaction_rdd_final.withColumn('rating', all_val_interaction_rdd_final['_1'].getItem("rating"))
    all_val_interaction_rdd_final = all_val_interaction_rdd_final.withColumn('is_reviewed', all_val_interaction_rdd_final['_1'].getItem("is_reviewed"))

    temp_val_interaction = all_val_interaction_rdd_final.select('_2', 'user_id', 'book_id', 'is_read', 'rating', 'is_reviewed')
    temp_val_interaction.createOrReplaceTempView('temp_val_interaction')
    temp_even_val_interaction = spark.sql('SELECT * FROM temp_val_interaction WHERE _2 %2 =0')
    temp_odd_val_interaction = spark.sql('SELECT * FROM temp_val_interaction WHERE _2 %2 =1')

    # test
    all_test_interaction_rdd = all_test_interaction.rdd.zipWithIndex()
    all_test_interaction_rdd_final = all_test_interaction_rdd.toDF()
    all_test_interaction_rdd_final = all_test_interaction_rdd_final.withColumn('user_id', all_test_interaction_rdd_final['_1'].getItem("user_id"))
    all_test_interaction_rdd_final = all_test_interaction_rdd_final.withColumn('book_id', all_test_interaction_rdd_final['_1'].getItem("book_id"))
    all_test_interaction_rdd_final = all_test_interaction_rdd_final.withColumn('is_read', all_test_interaction_rdd_final['_1'].getItem("is_read"))
    all_test_interaction_rdd_final = all_test_interaction_rdd_final.withColumn('rating', all_test_interaction_rdd_final['_1'].getItem("rating"))
    all_test_interaction_rdd_final = all_test_interaction_rdd_final.withColumn('is_reviewed', all_test_interaction_rdd_final['_1'].getItem("is_reviewed"))

    temp_test_interaction = all_test_interaction_rdd_final.select('_2', 'user_id', 'book_id', 'is_read', 'rating', 'is_reviewed')
    temp_test_interaction.createOrReplaceTempView('temp_test_interaction')
    temp_even_test_interaction = spark.sql('SELECT * FROM temp_test_interaction WHERE _2 %2 =0')
    temp_odd_test_interaction = spark.sql('SELECT * FROM temp_test_interaction WHERE _2 %2 =1')


    temp_even_test_interaction =  temp_even_test_interaction.drop('_2')
    temp_odd_test_interaction = temp_odd_test_interaction.drop('_2')
    temp_odd_val_interaction =  temp_odd_val_interaction.drop('_2')
    temp_even_val_interaction =  temp_even_val_interaction.drop('_2')
    temp_even_val_interaction.createOrReplaceTempView('temp_even_val_interaction')
    temp_odd_val_interaction.createOrReplaceTempView('temp_odd_val_interaction')
    temp_odd_test_interaction.createOrReplaceTempView('temp_odd_test_interaction')
    temp_even_test_interaction.createOrReplaceTempView('temp_even_test_interaction')


    training = spark.sql('SELECT * FROM training_from_train UNION ALL SELECT * FROM temp_even_val_interaction UNION ALL SELECT * FROM temp_even_test_interaction')
    validation = temp_odd_val_interaction
    testing = temp_odd_test_interaction

    return training, validation, testing
