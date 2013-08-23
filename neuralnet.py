from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised import BackpropTrainer
import csv
def make_dataset():
	fid = open('normalized_data','r')
	data = SupervisedDataSet(4,1)
	csvObj = csv.reader(fid,delimiter=',')
	i=1
	for row in csvObj:
		if i<=25:
			inputs = [row[0],row[1],row[2],row[3]]
			output = [1]
			data.addSample(inputs,output)
		if i>50 and i<=75:
			inputs = [row[0],row[1],row[2],row[3]]
			output = [50]
			data.addSample(inputs,output)
		if i>100 and i>=125:
			inputs = [row[0],row[1],row[2],row[3]]
			output = [100]
			data.addSample(inputs,output)
		i+=1	
	return data
def training(d):
    """
    Builds a network and trains it.
    """
    n = buildNetwork(d.indim, 4,4, d.outdim,recurrent=False)
    t = BackpropTrainer(n, d, learningrate = 0.40, momentum = 0.99, verbose = True)
    for epoch in range(0,10):
        t.train()
    return t,n	

def test(trained):
	fid = open('normalized_data','r')
	data = SupervisedDataSet(4,1)
	csvObj = csv.reader(fid,delimiter=',')
	i=1
	for row in csvObj:
		if i<=50:
			inputs = [row[0],row[1],row[2],row[3]]
			output = [1]
			data.addSample(inputs,output)
		if i>50 and i<=100:
			inputs = [row[0],row[1],row[2],row[3]]
			output = [50]
			data.addSample(inputs,output)
		if i>=100:
			inputs = [row[0],row[1],row[2],row[3]]
			output = [100]
			data.addSample(inputs,output)
		i+=1	
	trained.testOnData(data,verbose=True)	


def run():
    """
    Use this function to run build, train, and test your neural network.
    """
    trainingdata = make_dataset()
    trained,network = training(trainingdata)
    # test(trained)
    print network.activate(5,6,7,8)