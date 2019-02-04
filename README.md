# CNN-Model
A Convolutional Neural Network Model Training On The MNIST Dataset And Comparison Of The Optimizers (Graduation Thesis)

I have created this Convolutional Neural Network Model to be trained on the MNIST Dataset. 

In this model, I employed two Convolutional Layer and one Fully Connected Layer. In Convolutional Layer, I applied eight 3x3 and sixteen 5x5 filters. In this layer, I used the RELU activation function and Dropout with 25% ratio. In Fully Connected Layer, I used two Dense with RELU and Softmax activation function. After that, I designed the optimization, compiling and training parts. I tried the Keras Optimizers on this model separately. Through the scoring and plotting parts, I got the results which are Accuracy and Loss values. Then, I compared all optimizers with respect to Test Loss and Train Loss values. Finally, I got the predictions from the model which recognizes the MNIST handwritten digits averagely 97%.
