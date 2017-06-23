It's only a simple test for me to learn abow DEEP LEARNING and tensorflow.
I've used four models which are DNN,CNN,RNN,FCN to do tests on MNIST based on tensorflow.
What I've done is based on  windows system and run by python 3.6.
Limited by the performance of my pc, I've only choose 1000 test data as the test dataset.

For DNN,I only used a hidden layer but with 1024 nodes.
Because activited,the accuracy of test turned out to be 98.2% as we can see in the picture 'MNIST_DNN'.

For CNN,I used LeNet-5 as my model which only has two cll-connected layers with 1024 nodes and 10 nodes respectively.
As result,it ran slowly but it converged fast and the test accuracy got 99% which is the highest among the four models.

For RNN, the structure of LSTM was used. Although imported tensorflow ,I still couldn't use rnn_cell without any reason,so
I wrote the fuction,LSTM_cell on my own and it worked. The reuslt showed RNN convege slowly  and got a test accuracy of 97.5%.

For FCN(all convolution layers), we often use it on semantic image segmentation more than image classification.But I still had
a try.Without knowing why,it turned out badly after replacing all-connected layers with convolution layers. The procedure ran 
slowler than CNN and the test accuracy was only 90% which was much smaller than the other three.