# xnorNeuralNetwork
Neural Network for logical XNOR

## Gen Data
The genData bash script compiles and executes `xnorDataGenerator.c` which produces data:
- Integer x1 between 0 - 49
- Integer x2 between 50 - 99
- 1 if (x1 < 50 and x2 < 50) or (x1 >= 50 and x2 >= 50), 0 otherwise
```sh
$ ./genData
```

## Running the Neural Network
The octave program will walk you through how a neural network understands an XNOR pattern.
```sh
$ octave --no-gui
>> xnor
```
## Recommendations
- In `xnorDataGenerator.c` try modifying the number of examples generated to see how different data sizes affects the results and accuracy
- In `xnor.m` try modifying the following to see how they affect the results
	* hidden\_layer\_size (Units in first hidden layer)
	* hidden\_layer\_2\_size (Units in second hidden layer)
	* MaxIter option
