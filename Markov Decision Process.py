import datetime
import numpy as np

# List of assignment states
states_list = ['a,b,c', 'a,b', 'a,c', 'b,c', '(a,b),c', '(b,a),c', '(a,c),b', '(c,a),b', '(b,c),a', '(c,b),a', '(a,b)',
               '(b,a)', '(a,c)', '(c,a)', '(c,b)', '(b,c)', '(a,b,c)', '(a,c,b)', '(b,a,c)', '(c,b,a)', '(c,a,b)', '(b,c,a)']
# List of assignment actions
actions = ['ga', 'gb', 'gc', 'at', 'bt', 'ct', 'ab', 'ac', 'ba', 'bc', 'ca', 'cb']
goal_state = '(a,c,b)'

gamma = 0.9
eps = 0.0009

value_iteration = True
policy_iteration = True


# Initializing the dictionary (dict) with zero values for all states (s) and actions (a) (S x A x S)
def dict_propositions(own_dict, s, a):
    for s_from in states_list:
        own_dict[s_from] = {}
        for a in actions:
            own_dict[s_from][a] = {}
            for s_to in states_list:
                own_dict[s_from][a][s_to] = 0


# To get transition probability from s_from to s_to while executing action a (S x A  S --> R)
def t(s_from, a, s_to):
    s_from = s_from.replace(' ', '')
    a = a.replace(' ', '')
    s_to = s_to.replace(' ', '')
    return t_dict[s_from][a][s_to]


# Setting the transition probability from s_from to s_to while executing action a to value
def set_t(s_from, a, s_to, value):
    s_from = s_from.replace(' ', '')
    a = a.replace(' ', '')
    s_to = s_to.replace(' ', '')
    t_dict[s_from][a][s_to] = value


# Computation of the expected reward for each state-action pair
def exp_r(s, a):
    expected_r = 0
    for s_to, p in t_dict[s][a].iteritems():
        if s_to == goal_state:
            expected_r += p * 100
        elif p < 0.5:
            expected_r += p * -10
        else:
            expected_r += p * -1
    return expected_r


def set_data():
    set_t('a,b,c', 'ga', 'b,c', 0.8)
    set_t('a,b,c', 'ga', 'a,c', 0.1)
    set_t('a,b,c', 'ga', 'a,b', 0.1)
    set_t('a,b,c', 'gb', 'b,c', 0.1)
    set_t('a,b,c', 'gb', 'a,c', 0.8)
    set_t('a,b,c', 'gb', 'a,b', 0.1)
    set_t('a,b,c', 'gc', 'b,c', 0.1)
    set_t('a,b,c', 'gc', 'a,c', 0.1)
    set_t('a,b,c', 'gc', 'a,b', 0.8)

    set_t('a,b', 'ct', 'a,b,c', 1.0)
    set_t('a,b', 'cb', '(b,c),a', 0.9)
    set_t('a,b', 'cb', '(a,c),b', 0.1)
    set_t('a,b', 'ca', '(b,c),a', 0.1)
    set_t('a,b', 'ca', '(a,c),b', 0.9)

    set_t('a,c', 'bt', 'a,b,c', 1.0)
    set_t('a,c', 'ba', '(a,b),c', 0.9)
    set_t('a,c', 'ba', '(c,b),a', 0.1)
    set_t('a,c', 'bc', '(a,b),c', 0.1)
    set_t('a,c', 'bc', '(c,b),a', 0.9)

    set_t('b,c', 'at', 'a,b,c', 1.0)
    set_t('b,c', 'ab', '(c,a),b', 0.1)
    set_t('b,c', 'ab', '(b,a),c', 0.9)
    set_t('b,c', 'ac', '(c,a),b', 0.9)
    set_t('b,c', 'ac', '(b,a),c', 0.1)

    set_t('(a,b),c', 'gb', 'a,c', 0.9)
    set_t('(a,b),c', 'gb', '(a,b)', 0.1)
    set_t('(a,b),c', 'gc', 'a,c', 0.1)
    set_t('(a,b),c', 'gc', '(a,b)', 0.9)

    set_t('(b,a),c', 'ga', 'b,c', 0.9)
    set_t('(b,a),c', 'ga', '(b,a)', 0.1)
    set_t('(b,a),c', 'gc', 'b,c', 0.1)
    set_t('(b,a),c', 'gc', '(b,a)', 0.9)

    set_t('(a,c),b', 'gb', '(a,c)', 0.9)
    set_t('(a,c),b', 'gb', 'a,b', 0.1)
    set_t('(a,c),b', 'gc', '(a,c)', 0.1)
    set_t('(a,c),b', 'gc', 'a,b', 0.9)

    set_t('(c,a),b', 'gb', '(c,a)', 0.9)
    set_t('(c,a),b', 'gb', 'b,c', 0.1)
    set_t('(c,a),b', 'ga', 'b,c', 0.9)
    set_t('(c,a),b', 'ga', '(c,a)', 0.1)

    set_t('(b,c),a', 'ga', '(b,c)', 0.9)
    set_t('(b,c),a', 'ga', 'a,b', 0.1)
    set_t('(b,c),a', 'gc', '(b,c)', 0.1)
    set_t('(b,c),a', 'gc', 'a,b', 0.9)

    set_t('(c,b),a', 'ga', '(c,b)', 0.9)
    set_t('(c,b),a', 'ga', 'a,c', 0.1)
    set_t('(c,b),a', 'gb', 'a,c', 0.9)
    set_t('(c,b),a', 'gb', '(c,b)', 0.1)

    set_t('(a,b)', 'ct', '(a,b),c', 1)
    set_t('(a,b)', 'cb', '(a,b,c)', 1)

    set_t('(b,a)', 'ct', '(b,a),c', 1)
    set_t('(b,a)', 'ca', '(b,a,c)', 1)

    set_t('(a,c)', 'bt', '(a,c),b', 1)
    set_t('(a,c)', 'bc', '(a,c,b)', 1)

    set_t('(c,a)', 'bt', '(c,a),b', 1)
    set_t('(c,a)', 'ba', '(c,a,b)', 1)

    set_t('(c,b)', 'at', '(c,b),a', 1)
    set_t('(c,b)', 'ab', '(c,b,a)', 1)

    set_t('(b,c)', 'at', '(b,c),a', 1)
    set_t('(b,c)', 'ac', '(b,c,a)', 1)

    set_t('(a,b,c)', 'gc', '(a,b)', 1)
    set_t('(b,a,c)', 'gc', '(b,a)', 1)
    set_t('(c,b,a)', 'ga', '(c,b)', 1)
    set_t('(c,a,b)', 'gb', '(c,a)', 1)
    set_t('(b,c,a)', 'ga', '(b,c)', 1)

    return t_dict


# ------------------------------------------------------------------------------------------

t_dict = {}
U_opt = {}
Pi = {}

dict_propositions(t_dict, states_list, actions)
set_data()

# Initialize all utilities to 0 and create an empty policy dictionary
for s_from in states_list:
    U_opt[s_from] = 0
    Pi[s_from] = ''

iter_counter = 0
start_time = datetime.datetime.now()

# value iteration algorithm
while value_iteration:

    iter_counter += 1
    delta = 0

    # Copy of optimal utilities in the previous iteration
    U_i = U_opt.copy()
    for s in states_list:

        # For every state, initialize an empty action and set the max_value to -inf
        max_action = ''
        max_value = -float('inf')

        # Find the action returning the highest expected utility when leaving state s
        for a in actions:

            # Compute the expected utility of executing some action a
            sum = 0
            for s1 in states_list:
                sum += t(s, a, s1) * U_i[s1]

            # If a higher utility is found than the one stored for state s, update the optimal utility and corresponding action
            if (exp_r(s, a) + gamma * sum) > max_value:
                max_value = exp_r(s, a) + gamma * sum
                max_action = a

        # If the maximum utility found in the current iteration exceeds the stored value, update the utility and policy
        if max_value > U_opt[s]:
            U_opt[s] = max_value
            Pi[s] = max_action

        # Update delta to the maximum distance between the new and old utility over all states
        if abs(U_opt[s] - U_i[s]) > delta:
            delta = abs(U_opt[s] - U_i[s])

    # Stop the algorithm if the maximum utility distance between two successive iterations is small enough
    # Hence, delta is least as small as epsilon, or simply zero
    if delta < eps or delta == 0:
        break

end_time = datetime.datetime.now()

print '\nVALUE ITERATION'
print '\n' + str(iter_counter) + ' steps where required in ' + str((end_time - start_time).total_seconds()) + ' seconds'
print "\nOptimal utility: ", U_opt
print "Optimal policy: ", Pi

# policy iteration
pi = {}
# Initializing all optimal utilities to 0 and all actions to the first action in the actions_list (at)
for s in states_list:
    U_opt[s] = 0
    Pi[s] = actions[0]


# Computing the optimal utilities for some policy Pi and updating U_opt
def opt_u():
    a = []
    b = []
    for s in states_list:
        b.append(exp_r(s, Pi[s]))
        row = []
        for s1 in states_list:
            if s == s1:
                row.append(1 - (gamma * t(s, Pi[s], s1)))
            else:
                row.append(-gamma * t(s, Pi[s], s1))

        a.append(row)

    #Solve and create dictionary
    solution = list(np.linalg.solve(np.array(a), np.array(b)))
    for s in states_list:
        U_opt[s] = solution[states_list.index(s)]


iter_counter = 0
start_time = datetime.datetime.now()


# Policy iteration algorithm
while policy_iteration:

    iter_counter += 1
    opt_u()                 # Update current optimal utilities
    new_policy = dict(Pi)   # Make copy of current policy

    for s in states_list:
        for a in actions:

            if a != Pi[s]:  # no need to compare to the action in your current policy
                sum = 0
                for s1 in states_list:
                    sum += t(s, a, s1) * U_opt[s1]
                other_action = exp_r(s, a) + (gamma * sum)

                if other_action > U_opt[s]:
                    # If the utility is higher for some action, replace it in the copy
                    # Don't break here because we want to find all the changes in pi before comparing
                    # if you set pi[s] = a, you aren't able to compare if the current policy (Pi) has changed
                    new_policy[s] = a

    if Pi == new_policy:
        Pi[goal_state] = ''
        break
    else:
        Pi = dict(new_policy)


end_time = datetime.datetime.now()

print '\nPOLICY ITERATION'
print '\n' + str(iter_counter) + ' steps where required in ' + str((end_time - start_time).total_seconds()) + ' seconds'
print "\nOptimal utility: ", U_opt
print "Optimal policy: ", Pi
