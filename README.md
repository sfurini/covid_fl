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

* A space-separated text file with phenotype information. The requisites of this file are (see example_pheno.txt):
  * An header line is present
  * The first column corresponds to the id of the sample (the name of this column is not important)

Multiple phenotypes can be included in the same file. The phenotype that is used for training is selected into client.py. 

* A space-separated text file with covariate information. The requisites of this file are (see example_pheno.txt):
  * An header line is present
  * The first column corresponds to the id of the sample (the name of this column is not important)
  * Two columns named *age* and *sex* are included
  * Gender is coded as 1/0 for female/male

For testing the code locally:

* run prepare_inputs.sh to convert the mask files in plink format into txt

* run the server with: python server.py

* run a client with: python client.py

In this way only one client is executed and the code reproduces a local (not-federated) learning of the model
