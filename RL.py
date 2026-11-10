import pandas as pd
import numpy as np
import itertools
import random
import joblib

df = pd.read_csv("jopacc_system_month_november_2022_to_june_2026.csv")

print(df["system"].unique())

df["Transactions_Level"], trans_bins = pd.qcut(
    df["transactions_total"],
    3,
    labels=[0, 1, 2],
    retbins=True,
    duplicates="drop"
)
df["Transactions_Level"] = df["Transactions_Level"].astype(int)

df["Value_Level"], value_bins = pd.qcut(
    df["transaction_value_jod"],
    3,
    labels=[0, 1, 2],
    retbins=True,
    duplicates="drop"
)
df["Value_Level"] = df["Value_Level"].astype(int)

# STATE

df["State"] = list(zip(
    df["system"],
    df["Transactions_Level"],
    df["Value_Level"]
))

# ACTIONS

actions = {

    "CliQ": {
        0: "Do Nothing",
        1: "Run Awareness Campaign",
        2: "Increase Legal Entity Participation",
        3: "Promote Merchant Payments"
    },

    "JoMoPay": {
        0: "Do Nothing",
        1: "Increase Wallet Usage",
        2: "Reduce Cash-Out Dependency",
        3: "Promote Wallet Transfers"
    },

    "ACH": {
        0: "Do Nothing",
        1: "Review High Value Settlements",
        2: "Promote Foreign Currency Transfers",
        3: "Improve Settlement Efficiency"
    },

    "ECCU": {
        0: "Do Nothing",
        1: "Reduce Returned Cheques",
        2: "Strengthen Risk Monitoring",
        3: "Improve Cheque Verification"
    },

    "eFAWATEERcom": {
        0: "Do Nothing",
        1: "Promote Digital Bill Payments",
        2: "Reduce Cash Payments",
        3: "Increase Biller Participation"
    }

}

n_actions = 4

# ACTION 
action_effect = {
    0: (0, 0),
    1: (1, 0),
    2: (0, 1),
    3: (1, 1),
}

action_cost = {
    0: 0,
    1: 5,
    2: 10,
    3: 15
}

# STATE SPACE

systems = sorted(df["system"].unique())

all_states = list(itertools.product(
    systems,
    [0, 1, 2],
    [0, 1, 2]
))

state_to_index = {s: i for i, s in enumerate(all_states)}


# Q TABLES

Q_reward1 = np.zeros((len(all_states), n_actions))
Q_reward2 = np.zeros((len(all_states), n_actions))

# TRAINING PARAMETERS

episodes = 3000
alpha = 0.05
gamma = 0.90
epsilon_decay = 0.97
epsilon_min = 0.05
initial_budget = 50
max_steps = 10

# =========================
# TRAINING - REWARD DESIGN 1


epsilon = 1.0    
for episode in range(episodes):

    state = random.choice(all_states)
    budget = initial_budget
    steps = 0

    while True:

        state_idx = state_to_index[state]

        # epsilon-greedy
        if random.random() < epsilon:
            action = random.randint(0, n_actions - 1)
        else:
            action = np.argmax(Q_reward1[state_idx])

        system, trans_level, value_level = state
        d_trans, d_value = action_effect[action]

        cost = action_cost[action]
        budget -= cost

         
        ineffective = (action != 0) and (trans_level == 2 and value_level == 2)

         
        reward = trans_level + value_level
        reward -= 1 if action != 0 else 0           
        if ineffective:
            reward -= 5                              

         
        next_state = (
            system,
            min(trans_level + d_trans, 2),
            min(value_level + d_value, 2)
        )
        next_idx = state_to_index[next_state]

         
        Q_reward1[state_idx, action] += alpha * (
            reward + gamma * np.max(Q_reward1[next_idx]) - Q_reward1[state_idx, action]
        )

        state = next_state
        steps += 1

         
        if budget <= 0:
            break
        if steps >= max_steps:
            break
        if state[1] == 2 and state[2] == 2:
            break

    epsilon = max(epsilon * epsilon_decay, epsilon_min)


 
# TRAINING - REWARD DESIGN 2
 

epsilon = 1.0

for episode in range(episodes):

    state = random.choice(all_states)
    budget = initial_budget
    steps = 0

    while True:

        state_idx = state_to_index[state]

        if random.random() < epsilon:
            action = random.randint(0, n_actions - 1)
        else:
            action = np.argmax(Q_reward2[state_idx])

        system, trans_level, value_level = state
        d_trans, d_value = action_effect[action]

        cost = action_cost[action]
        budget -= cost

        ineffective = (action != 0) and (trans_level == 2 and value_level == 2)

        reward = trans_level + value_level - cost
        reward -= 1 if action != 0 else 0
        if ineffective:
            reward -= 5

        next_state = (
            system,
            min(trans_level + d_trans, 2),
            min(value_level + d_value, 2)
        )
        next_idx = state_to_index[next_state]

        Q_reward2[state_idx, action] += alpha * (
            reward + gamma * np.max(Q_reward2[next_idx]) - Q_reward2[state_idx, action]
        )

        state = next_state
        steps += 1

        if budget <= 0:
            break
        if steps >= max_steps:
            break
        if state[1] == 2 and state[2] == 2:
            break

    epsilon = max(epsilon * epsilon_decay, epsilon_min)

 
# POLICY COMPARISON

score1 = np.mean(np.max(Q_reward1, axis=1))
score2 = np.mean(np.max(Q_reward2, axis=1))

print("Reward Design 1 Score (improvement - steps):", score1)
print("Reward Design 2 Score (improvement - cost - steps):", score2)

# =========================
# SELECT BEST POLICY

if score2 >= score1:
    best_Q = Q_reward2
    best_reward = "Reward Design 2 (improvement - cost - steps)"
else:
    best_Q = Q_reward1
    best_reward = "Reward Design 1 (improvement - steps)"

print("Selected Policy:", best_reward)

joblib.dump(best_Q, "q_learning_policy.pkl")
joblib.dump(trans_bins, "trans_bins.pkl")
joblib.dump(value_bins, "value_bins.pkl")
joblib.dump(state_to_index, "state_to_index.pkl")
joblib.dump(actions, "actions.pkl")
joblib.dump(action_cost, "action_cost.pkl")
joblib.dump(action_effect, "action_effect.pkl")

print("Policy and supporting artifacts saved successfully!")


# =========================
# RECOMMENDATION FUNCTION
# =========================
def recommend_action(system, transactions_total, feature_value, budget):

     
    transactions_total = np.clip(transactions_total, trans_bins[0], trans_bins[-1])
    feature_value = np.clip(feature_value, value_bins[0], value_bins[-1])

     
    trans_level = int(
        pd.cut([transactions_total], bins=trans_bins, labels=[0, 1, 2], include_lowest=True)[0]
    )
    value_level = int(
        pd.cut([feature_value], bins=value_bins, labels=[0, 1, 2], include_lowest=True)[0]
    )

    state = (system, trans_level, value_level)
    state_idx = state_to_index[state]
    q_values = best_Q[state_idx]
    top_actions = np.argsort(q_values)[::-1][:2]   

    recommendations = []

    for act in top_actions:
        act = int(act)
        cost = action_cost[act]

        ineffective = (act != 0) and (trans_level == 2 and value_level == 2)
        reward = trans_level + value_level - cost
        reward -= 1 if act != 0 else 0
        if ineffective:
            reward -= 5

        d_trans, d_value = action_effect[act]
        next_state = (
            system,
            min(trans_level + d_trans, 2),
            min(value_level + d_value, 2)
        )

        recommendations.append({
            "action": actions[system][act],
            "reward": reward,
            "cost": cost,
            "next_state": next_state,
            "remaining_budget": budget - cost
        })

    return recommendations