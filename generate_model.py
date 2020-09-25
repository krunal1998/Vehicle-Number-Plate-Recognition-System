import data_loader
import network
import pickle



training_data,test_data=data_loader.load_data()
net = network.Network([400, 100, 36])
net.SGD(training_data,2000, 3, 0.3, test_data)

with open('C:/Users/Admin/Desktop/project/ml_code/ml_models/model10.pkl', 'wb') as output:
	pickle.dump(net, output, pickle.HIGHEST_PROTOCOL)
	print ("success")
#del net
#with open('model4.pkl', 'rb') as input:
#	net=pickle.load(input)
#	print(net.biases[1])
#	print(net.weights[1][0])
#	print(net.weights[1])
	
	

