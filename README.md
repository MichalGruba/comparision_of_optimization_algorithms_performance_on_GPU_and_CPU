# Comparision_of_optimization_algorithms_performance_on_GPU_and_CPU

The program includes implementation of 2 optimization algorithms:
- Particle Swarm Optimization algorithm
- Spiral Optimization Algorithm

on both CPU and GPU.

At the start of the program user is asked to provide number of particles searching for the minimum and number
of iterations of each algorithm as well as choice of saving the positions of particles at every iteration to txt file
in order to visualize the optimization process in MATLAB. After that user has a choice to choose one of the following
functions on which he wishes to test the algorithms on:
- Sphere function
![image](https://user-images.githubusercontent.com/67229687/176221144-57acc6d0-b227-4139-b16b-aeb9ddddef5a.png)

- Booth function
![image](https://user-images.githubusercontent.com/67229687/176221252-305fb22c-4148-4183-b966-dd05a62d62aa.png)

- Rastrigin function
![image](https://user-images.githubusercontent.com/67229687/176221357-2a46fe91-5b3b-47de-89c4-6b8bf73b09eb.png)

- Beale function
![image](https://user-images.githubusercontent.com/67229687/176221473-a9fb823e-a703-4331-bc72-28ffa45452a0.png)

- Goldstein-Price function
![image](https://user-images.githubusercontent.com/67229687/176222015-0329113d-e06d-4a8f-8875-6ffb68fcab01.png)

The output is a result each algorithm finds and time it took to run each algorithm.

Results acquired on Intel Core i5 6500 & GeForce GTX 1050ti

![iterations](https://user-images.githubusercontent.com/67229687/176214450-2022e6b3-dc06-41c0-8420-df5b80ef5d16.jpeg)

![particles](https://user-images.githubusercontent.com/67229687/176214655-e8898a7a-2a76-4827-a99c-1868888e985a.jpeg)

![both](https://user-images.githubusercontent.com/67229687/176214683-0187c4b4-0178-40fa-8250-f127a6603336.jpeg)
