
# coding: utf-8

# In[1]:

get_ipython().system(u'pip install --user --upgrade --no-cache-dir ibmos2spark')
from pyspark.sql.types import *
import ibmos2spark as oss
import time

credentials = {

}

schema = StructType([
        StructField("COMPANY", StringType(), True),
        StructField("CUSTOMER_NUM", StringType(), True),
        StructField("ORDER_NUM", StringType(), True),
        StructField("ORDER_DATE_TIME", StringType(), True),
        StructField("MSTR_COMP_CODE", StringType(), True),
        StructField("PLCC_OFR_CONV", IntegerType(), True),
        StructField("PLCC_OFR_APPROVED", IntegerType(), True),
        StructField("VIP_OFFER_CONV", IntegerType(), True),
        StructField("AC_OFFER_CONV", IntegerType(), True),
        StructField("CUST_CHOICE_ACCPT", IntegerType(), True),
        StructField("PROTECT_PLUS_ACCPT", IntegerType(), True),
        StructField("COOR_OFFER_ACCPT", IntegerType(), True),
        StructField("PLCC_OFR_MADE", StringType(), True),
        StructField("VIP_OFFER_MADE", IntegerType(), True),
        StructField("AC_OFFER_MADE", IntegerType(), True),
        StructField("CUST_CHOICE_OFFRD", IntegerType(), True),
        StructField("PROTECT_PLUS_OFFRD", IntegerType(), True),
        StructField("COOR_OFFER_OFFRD", IntegerType(), True),
        StructField("NEW_CUST_FLG", IntegerType(), True),
        StructField("APL_CC", IntegerType(), True),
        StructField("LNC_CC", IntegerType(), True),
        StructField("TOG_CC", IntegerType(), True),
        StructField("WTS_CC", IntegerType(), True),
        StructField("AMO_CC", IntegerType(), True),
        StructField("BFA_CC", IntegerType(), True),
        StructField("BLR_CC", IntegerType(), True),
        StructField("GDV_CC", IntegerType(), True),
        StructField("NTO_CC", IntegerType(), True),
        StructField("SAH_CC", IntegerType(), True),
        StructField("SOL_CC", IntegerType(), True),
        StructField("DND_CC", IntegerType(), True),
        StructField("EXISTING_VIP", IntegerType(), True),
        StructField("WAS_VIP", IntegerType(), True),
        StructField("PRIOR_DECL_VIP", IntegerType(), True),
        StructField("VIP_NEVER", IntegerType(), True),
        StructField("PRIOR_AC_PURCH", IntegerType(), True),
        StructField("PRIOR_AC_OFFRS", IntegerType(), True),
        StructField("CUST_CHOICE_PREV_ACC", IntegerType(), True),
        StructField("PROT_PLUS_PREV_ACC", IntegerType(), True),
        StructField("COOR_OFFCOOR_OFFER_PREV_ACC", IntegerType(), True),
        StructField("MERCH_AMOUNT", DoubleType(), True),
        StructField("FREIGHT_AMOUNT", DoubleType(), True),
        StructField("ADDIT_FRT_AMOUNT", DoubleType(), True),
        StructField("MEDIA", IntegerType(), True),
        StructField("PROT_PLUS_FEE", DoubleType(), True),
        StructField("STATE", StringType(), True)
])

bmos = oss.bluemix2d(sc, credentials) #sc is the SparkContext instance

train = sqlContext.read .format('com.databricks.spark.csv') .option('header', 'true') .option('inferSchema', 'false') .schema(schema) .load(bmos.url('notebooks', 'train_sample.csv')) .cache()

test = sqlContext.read .format('com.databricks.spark.csv') .option('header', 'true') .option('inferSchema', 'true') .load(bmos.url('notebooks', 'test_sample.csv')) .cache()


# In[2]:

train.registerTempTable('train')
test.registerTempTable('test')
train = sqlContext.sql('select * from train where NEW_CUST_FLG = 0')
test = sqlContext.sql('select * from test where NEW_CUST_FLG = 0')

targets = {}
targets['plcc'] = 'PLCC_OFR_APPROVED'
targets['vip'] = 'VIP_OFFER_CONV'
targets['ac'] = 'AC_OFFER_CONV'
targets['gt'] = 'COOR_OFFER_ACCPT'
targets['cc'] = 'CUST_CHOICE_ACCPT'
targets['pp'] = 'PROTECT_PLUS_ACCPT'

string_fields = ['ORDER_DATE_TIME', 'MSTR_COMP_CODE', 'STATE']
for col in train.columns:
    if col not in string_fields:
        train = train.withColumn(col, train[col].cast("double"))
from pyspark.sql.functions import date_format, unix_timestamp, from_unixtime
train = train.withColumn('ORDER_DATE', from_unixtime(unix_timestamp('ORDER_DATE_TIME', 'yyyy-MM-dd-HH.mm.ss.uuuuuu'), 'yyyy-MM-dd'))
train = train.withColumn('YEAR', date_format('ORDER_DATE', 'yyyy').cast('double')).withColumn('MONTH', date_format('ORDER_DATE', 'MM').cast('double'))
train = train.withColumnRenamed('COOR_OFFCOOR_OFFER_PREV_ACC', 'COOR_OFFER_PREV_ACC')

for col in test.columns:
    if col not in string_fields:
        test = test.withColumn(col, test[col].cast("double"))
from pyspark.sql.functions import date_format, unix_timestamp, from_unixtime
test = test.withColumn('ORDER_DATE', from_unixtime(unix_timestamp('ORDER_DATE_TIME', 'yyyy-MM-dd-HH.mm.ss.uuuuuu'), 'yyyy-MM-dd'))
test = test.withColumn('YEAR', date_format('ORDER_DATE', 'yyyy').cast('double')).withColumn('MONTH', date_format('ORDER_DATE', 'MM').cast('double'))
test = test.withColumnRenamed('COOR_OFFCOOR_OFFER_PREV_ACC', 'COOR_OFFER_PREV_ACC')

train.registerTempTable('train')
test.registerTempTable('test')


# In[3]:

plcc_train = """
select *
from train
where ORDER_DATE >= '2013-02-05'
and MSTR_COMP_CODE in ('APL','TOG','WTS','LNS')
and PLCC_OFR_MADE = 1
union
select *
from train
where ORDER_DATE >= '2013-07-09'
and MSTR_COMP_CODE in ('NTO','SAH','SOL','GDV')
and PLCC_OFR_MADE = 1
union
select *
from train
where ORDER_DATE >= '2013-09-19'
and MSTR_COMP_CODE like 'DND'
and PLCC_OFR_MADE = 1
"""


df_train = sqlContext.sql(plcc_train)
df_train = df_train.withColumnRenamed(targets['plcc'], 'label')


# In[4]:

from pyspark.ml.feature import OneHotEncoder
year_encoder = OneHotEncoder(dropLast=False, inputCol="YEAR", outputCol="YEAR_VEC")
month_encoder = OneHotEncoder(dropLast=False, inputCol="MONTH", outputCol="MONTH_VEC")
media_encoder = OneHotEncoder(dropLast=False, inputCol="MEDIA", outputCol="MEDIA_VEC")

from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(    inputCols=[
            #    'APL_CC', 'LNC_CC', 'TOG_CC', 'WTS_CC', 'AMO_CC', 'BFA_CC', 'BLR_CC', 'GDV_CC', 'NTO_CC', 'SAH_CC', 'SOL_CC', 'DND_CC',\
               'WAS_VIP', 'PRIOR_DECL_VIP', \
#                'EXISTING_VIP', 'VIP_NEVER', \
               'PRIOR_AC_PURCH', 'PRIOR_AC_OFFRS', 'CUST_CHOICE_PREV_ACC', 'COOR_OFFER_PREV_ACC', 'PROT_PLUS_PREV_ACC', \
               'YEAR_VEC', 'MONTH_VEC', 'MEDIA_VEC', 'NEW_CUST_FLG' \
#                'STATE_VEC', 'MSTR_COMP_CODE_VEC'\
              ],\
    outputCol='F_VEC')


# In[6]:

plcc_test = """
select *
from test
where ORDER_DATE >= '2013-02-05'
and MSTR_COMP_CODE in ('APL','TOG','WTS','LNS')
and PLCC_OFR_MADE = 1
union
select *
from test
where ORDER_DATE >= '2013-07-09'
and MSTR_COMP_CODE in ('NTO','SAH','SOL','GDV')
and PLCC_OFR_MADE = 1
union
select *
from test
where ORDER_DATE >= '2013-09-19'
and MSTR_COMP_CODE like 'DND'
and PLCC_OFR_MADE = 1
"""


df_test = sqlContext.sql(plcc_test)
df_test = df_test.withColumnRenamed(targets['plcc'], 'label')


# In[22]:

from pyspark.ml.feature import StringIndexer
labelIndexer = StringIndexer(inputCol="label", outputCol="labelIndex")
from pyspark.ml.classification import RandomForestClassifier
rf = RandomForestClassifier(labelCol="labelIndex", featuresCol="F_VEC", numTrees=400)

from pyspark.ml import Pipeline
pipeline = Pipeline(stages=[year_encoder, month_encoder, media_encoder, #                           state_stringIndexer, state_encoder, shop_encoder, \
                            assembler, labelIndexer, rf])


# In[23]:

model = pipeline.fit(df_train)
preds = model.transform(df_test)
from pyspark.ml.evaluation import BinaryClassificationEvaluator
# from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = BinaryClassificationEvaluator().setLabelCol("labelIndex").setMetricName("areaUnderROC")
# evaluator = MulticlassClassificationEvaluator().setLabelCol("labelIndex").setMetricName("recall")
auc = evaluator.evaluate(preds)
print("AUC = %g " % auc)


# In[26]:

preds.select('CUSTOMER_NUM', 'ORDER_NUM', 'probability', 'label', 'rawPrediction').repartition(1).write .format('com.databricks.spark.csv') .option('header', 'true') .save(bmos.url('notebooks', 'plcc_rf_preds_400tree_existing_allprob.csv'))


# In[25]:

preds.columns

