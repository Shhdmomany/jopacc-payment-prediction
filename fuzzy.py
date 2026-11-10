import numpy as np
import skfuzzy as fuzz
'''
features :  
Transactions -> all systems
Wallet users -> jomopay
value_cash_in_jod -> jomopay
Cash-out count -> jomopay


universe: 
np.arange(start, stop, step)
'''

# Universes
# Inputs
transactions_universe = np.arange(0, 1000001, 1000)
wallet_universe = np.arange(0, 500001, 500)
value_cash_in_jod_universe = np.arange(0, 100000001, 100000)
cashout_universe = np.arange(0, 500001, 500)
# Output 0–100 scale
growth_universe = np.arange(0,101,1)

# transactions
tx_low = fuzz.trimf(transactions_universe,[0,0,200000])
tx_medium = fuzz.trimf(transactions_universe,[100000,400000,700000])
tx_high = fuzz.trimf(transactions_universe,[500000,1000000,1000000])
# wallet users 
wallet_low = fuzz.trimf(wallet_universe,[0,0,100000])
wallet_medium = fuzz.trimf(wallet_universe,[50000,200000,350000])
wallet_high = fuzz.trimf(wallet_universe,[250000,500000,500000])
#Digital Payment Value
value_cash_in_jod_universe_low = fuzz.trimf(value_cash_in_jod_universe,[0,0,20000000])
value_cash_in_jod_universe_medium = fuzz.trimf(value_cash_in_jod_universe,[10000000,50000000,80000000])
value_cash_in_jod_universe_high = fuzz.trimf(value_cash_in_jod_universe,[60000000,100000000,100000000])
#Cash Out Count
cash_low = fuzz.trimf(cashout_universe,[0,0,100000])
cash_medium = fuzz.trimf(cashout_universe,[50000,200000,350000])
cash_high = fuzz.trimf(cashout_universe,[250000,500000,500000])
#Output Memberships
fragile = fuzz.trimf(growth_universe, [0,0,30])
unbalanced = fuzz.trimf(growth_universe,[20,40,60])
healthy = fuzz.trimf(growth_universe,[50,70,85])
sustainable = fuzz.trimf(growth_universe,[80,100,100])


def fuzzify(value, universe, mf):
    return fuzz.interp_membership(universe, mf, value)


def apply_rules(transactions,
                wallets,
                value_cash,
                cashout):

    # Transactions
    tx_l = fuzzify(transactions, transactions_universe, tx_low)
    tx_m = fuzzify(transactions, transactions_universe, tx_medium)
    tx_h = fuzzify(transactions, transactions_universe, tx_high)

    # Wallets
    w_l = fuzzify(wallets, wallet_universe, wallet_low)
    w_m = fuzzify(wallets, wallet_universe, wallet_medium)
    w_h = fuzzify(wallets, wallet_universe, wallet_high)

    # Digital
    d_l = fuzzify(value_cash, value_cash_in_jod_universe, value_cash_in_jod_universe_low)
    d_m = fuzzify(value_cash, value_cash_in_jod_universe, value_cash_in_jod_universe_medium)
    d_h = fuzzify(value_cash, value_cash_in_jod_universe, value_cash_in_jod_universe_high)

    # Cash Out
    c_l = fuzzify(cashout, cashout_universe, cash_low)
    c_m = fuzzify(cashout, cashout_universe, cash_medium)
    c_h = fuzzify(cashout, cashout_universe, cash_high)

    # ---------------- Rules ----------------
    
    rule1 = np.fmin(tx_h,
             np.fmin(w_h,
             np.fmin(d_h,c_l)))

 
    rule2 = np.fmin(tx_h,c_h)

 
    rule3 = np.fmin(tx_m,
         np.fmin(w_m, d_m))

 
    rule4 = np.fmin(d_h, c_l)

   
    rule5 = np.fmin(tx_h,
         np.fmin(w_h, c_m))

 
    rule6 = np.fmin(w_h, d_l)
    
 
    rule7 = np.fmin(tx_l,c_h)

    # Combine outputs
    fragile_activation = rule7
    unbalanced_activation = np.fmax(rule2,np.fmax(rule6, rule7))
    healthy_activation = np.fmax(rule3,np.fmax(rule4, rule5))
    sustainable_activation = rule1
    activated_rules = []

    if rule1 > 0:
        activated_rules.append("Rule 1")

    if rule2 > 0:
        activated_rules.append("Rule 2")

    if rule3 > 0:
        activated_rules.append("Rule 3")

    if rule4 > 0:
        activated_rules.append("Rule 4")

    if rule5 > 0:
        activated_rules.append("Rule 5")

    if rule6 > 0:
        activated_rules.append("Rule 6")

    if rule7 > 0:
        activated_rules.append("Rule 7")
    return (fragile_activation,
            unbalanced_activation,
            healthy_activation,
            sustainable_activation,
            activated_rules)


def aggregate(fragile_a,unbalanced_a,healthy_a,sustainable_a):

    fragile_cut = np.fmin(fragile_a, fragile)
    unbalanced_cut = np.fmin(unbalanced_a, unbalanced)
    healthy_cut = np.fmin(healthy_a, healthy)
    sustainable_cut = np.fmin(sustainable_a, sustainable)

    aggregated = np.fmax(
        fragile_cut,
        np.fmax(
            unbalanced_cut,
            np.fmax(
                healthy_cut,
                sustainable_cut
            )
        )
    )

    return aggregated

def growth_level(score):
    if score < 30:
        return "Fragile Growth"
    elif score < 55:
        return "Unbalanced Growth"
    elif score < 80:
        return "Healthy Growth"
    else:
        return "Sustainable Growth"


 
def centroid_defuzz(aggregated):
     if np.max(aggregated) == 0:
        return 0.0
     return fuzz.defuzz(growth_universe, aggregated, "centroid")


def mom_defuzz(aggregated):
    if np.max(aggregated) == 0:
        return 0.0
    return fuzz.defuzz(growth_universe, aggregated, "mom")

def evaluate_growth(transactions,wallets,digital,cashout):
    transactions = float(np.clip(transactions,transactions_universe.min(),transactions_universe.max()))
    wallets = float(np.clip(wallets,wallet_universe.min(),wallet_universe.max()))
    digital = float(np.clip(digital,value_cash_in_jod_universe.min(),value_cash_in_jod_universe.max()))
    cashout = float(np.clip(cashout,cashout_universe.min(),cashout_universe.max()))
    f, u, h, s, rules = apply_rules(transactions,wallets,digital,cashout)
    aggregated = aggregate(f, u, h, s)
    centroid_score = round(centroid_defuzz(aggregated), 2)
    mom_score = round(mom_defuzz(aggregated), 2)

    level = growth_level(centroid_score)

    return centroid_score, mom_score, level, rules
