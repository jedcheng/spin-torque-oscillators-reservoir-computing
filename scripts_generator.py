import numpy as np

no_of_stos_lower_limit = 2
no_of_stos_upper_limit = 6

current_density = 6

mackey_glass = np.load('mackey_glass_t17.npy')
mackey_glass = mackey_glass[0:2500]*6 + 0.7



for no_of_STOs in range(no_of_stos_lower_limit, no_of_stos_upper_limit):
    
    script = open(f"scripts/{no_of_STOs}.mx3", "w")

    script.write(f"NislandsX := {no_of_STOs} \n")

    with open("template.txt", "r") as template:
        for line in template:
            script.write(line)

    for i in range(2500):
        script.write(f'J.setregion(1, vector(0, 0, (-{current_density} - {mackey_glass[i]})*1e10)) \n')
        script.write(f'run(5e-9) \n')