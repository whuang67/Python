## Cheat sheet of Apache Spark

Set up Python 3 and PySpark

```bash
sudo apt-get install python3-pip
sudo apt-get install -y python3-setuptools
pip3 install wheel
pip3 install jupyter
sudo apt-get update
sudo apt-get install default-jre
sudo apt-get install scala
pip3 install py4j
```

Then download Apache Spark from [here](http://spark.apache.org/downloads.html).

```bash
sudo tar -zxvf spark-2.1.0-bin-hadoop2.7.tgz
export SPARK_HOME='home/mint/spark-2.1.0-bin-hadoop2.7'
export PATH=$SPARK_HOME:$PATH
export PYTHONPATH=$SPARK_HOME/python:$PYTHONPATH
export PYSPARK_DRIVER_PYTHON="jupyter"
export PYSPARK_DRIVER_PYTHON_OPTS="notebook"
export PYSPARK_PYTHON=python3
chmod 777 spark-2.1.0-bin-hadoop2.7
sudo chmod 777 spark-2.1.0-bin-hadoop2.7
cd spark-2.1.0-bin-hadoop2.7/
sudo chmod 777 python
cd python
sudo chmod 777 pyspark
```
