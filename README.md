# Federated Learning for WES/WGS working group

Code for federated learning of Machine Learning models over the consurtium. A description of the algoritms can be found here:

https://docs.google.com/document/d/1SETsJs77z-32fHzovNEPhMOapK959w-u8PVynYCh8Fg/edit?usp=sharing

The libraries *socket* and *threading* are used to manage communications between the server and the client

The *sklearn* library is used for the ML model


* server.py: server code. It creates the sockets for communicating with the clients and it manages the federated learning process

* prepare_inputs.sh: It creates input files for the client

* client.py: client code. When requested by the server it performs a step of minimization of the loss function, and it sends back the updated model

Required inputs:

* The mask files in plink format (the same ones that we used as inputs for the burden test with regenie)

* A space-separated text file with phenotype information. The first column is the id of the sample. Multiple phenotypes can be included in the same file, then the one to be used in training is selected into client.py. An example is included as example_pheno.txt.

* A space-separated text file with covariate information. Here again the first column is the id of the sample. This file needs to include two columns named *age* and *sex*. Sex should be coded as 1/0 for female/male. An example is included as example_covar.txt

For testing the code locally:

* run prepare_inputs.sh to convert the mask files in plink format into txt

* run the server with: python server.py

* run a client with: python client.py

In this way only one client is executed and the code reproduces a local (not-federated) learning of the model
