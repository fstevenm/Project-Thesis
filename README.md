## Project-Thesis

This project is a project about my thesis. My thesis discusses the mathematical modeling of the spread of COVID19 that involving decreased immunity and its numerical solution using the fourth order Runge-Kutta method. The mathematical model that was built is the

  ![image](https://user-images.githubusercontent.com/99526319/162615810-67e68860-c1f7-418a-869b-3cb218e1d990.png)

model, where 𝑆 is a group of individuals who have a risk of being infected, 𝐸 is a group of individuals who are exposed and in the incubation period, 𝐼𝐴 is a group of infected individuals without symptoms, 𝐼𝑆 is a group of infected individuals with symptoms, 𝑍 is the group of recovered individuals, and 𝑍0 is the group of recovered individuals who return to being individuals who have a risk of infection due to decreased immunity. The mathematical model is presented as a system of nonlinear ordinary differential equations. This thesis also discusses the analysis of the equilibrium point and the stability of the equilibrium point of the model that has been formed. Furthermore, the model was solved numerically by the fourth-order Runge-Kutta method. The numerical solution is solved and computed in Python. The solution is determined to observe the dynamic behavior change of each existing group of individuals. It can be observed that the solution will go to the equilibrium points obtained.

Moreover, analysis of the effect of decreased immunity and vaccination was also carried out. The results of the analysis show that the higher the probability of decreased immunity of recovered individuals, then the greater the number of infected individuals with symptoms, the fewer the number of recovered individuals, and the greater the number of recovered individuals who again have the risk of being infected due to decreased immunity. The higher the vaccination rate, then the fewer the number of infected individuals without symptoms, the fewer the number of infected individuals with symptoms, and the
greater the number of recovered individuals.

Besides Python, I also use maple software to determine the endemic equilibrium point and simplify some equations in determining the stability of the disease equilibrium point.

#### Basic reproduction number obtained
![image](https://user-images.githubusercontent.com/99526319/162615962-37b9f03d-e68f-44ae-b749-e0f73a552e00.png)

with

![image](https://user-images.githubusercontent.com/99526319/162615968-d13ec06d-1343-4c47-8c34-afefc90aad4d.png)

Note that the value of 0 obtained will not be a complex number, because it has been analyzed that

![image](https://user-images.githubusercontent.com/99526319/162615994-e55d98de-70b6-4c5c-8788-fa70f63b73a9.png)

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

### The Results

Here is some plot results about my thesis project. Note that parameter values used are sourced from the main reference journal and some assumptions.

#### Numerical solutions - 1
![Semua](https://user-images.githubusercontent.com/99526319/162615409-c90f2fe9-df81-4ba3-b256-695978380356.png)

#### One-group plot example: Infected Symptomatics Group
![IS](https://user-images.githubusercontent.com/99526319/162615450-e73a4b35-b95d-45cf-876e-34d134eeb4aa.png)

Note that the odeint result from the SciPy library is also used to compare with the numerical solution of the fourth-order Runge-Kutta method.

#### Effect of decreased immunity (xi parameter) to Infected Symptomatics Group 
![Xi_IS](https://user-images.githubusercontent.com/99526319/162615602-540c1543-b55c-4da3-8e7c-2fc45a9326e8.png)

#### Numerical solutions - 2
![Penyelesaian Numeris](https://user-images.githubusercontent.com/99526319/162615613-f08329aa-6360-4246-9c24-4e595b3a57fd.png)

#### Effect of vaccination (sigma parameter) to Infected Symptomatics Group
![IS](https://user-images.githubusercontent.com/99526319/162615638-ceb2e18a-2d98-44e3-8d63-11cd9ae5c829.png)

We can also see the effect of decreased immunity and vaccination to others group by change the code program to each group. You can see full my thesis code in the folder 'Thesis'

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

### Don't forget to provide the source (me) if you use this project!
