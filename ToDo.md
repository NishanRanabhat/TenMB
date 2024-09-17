1) Reconfigure the code to go from a traditional functional architecture to object oriented architecture (julia doesn't have OOP in traditional sense), so we use multiple dispatch and structs like the one in ExactDiagonalization repo.

2) Write an AutoMPO function like the one in iTensor to build Matrix Product operators given the Hamiltonian parameters. This is a challenge but with ChatGPT everything is doable. :) 

3) Reconfigure the code to run in GPU, the tensor contrctions will be faster. The communication overheads and RAM might be an issue.