## FW2022 CS655 GENI mini project:
# Effect of Network Topology on the Performance of Distributed Deep Learning

This repository is for the CS655 class project. 

* Client
  - The client sends a single image to institution server at a time.
  - The client receives the inference result from the institution server of the sent image.
	- The waiting times between the inference requests may vary (assume Poisson distribution).
* Institution Server
	- The institution server receives a single image from each of the clients.
	- The server takes the input received and performs an inference. While performing the inference, the institution server also learns parameters using a gradient descent optimization. For simplicity, we assume that the institution server only handles a single request at a time (no batch training). 
	- Institution server sends the gradient of the model to parameter server.
* Parameter Server
	- The parameter server receives the gradients.
	- After a certain time, the parameter server takes the mean of the received gradient, and update the global parameter.
	- The updated global parameter is broadcasted to the institution servers that are participating in the consensus group. In particular, denoting sent gradients from institution i within the cumulation phase as g_i^((1)),⋯,g_i^((n_i)), the new parameter value w updated using received information from N institutions can be expressed as follows:
w(t+1)←w(t)+ E[g_i^t: t∈{1,⋯,n_i }; i∈{1,⋯,N}]
