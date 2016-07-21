from numpy import array
def indicator(domain_to_be_indicated, barrier):
    """the indicator influences the function argument, not value. So here it iterates through x-domain and cuts any
    values of function with an argument less than H"""
    indicated = []
    for elem in domain_to_be_indicated:
        if elem > barrier:
            indicated.append(1.0)
        else:
            indicated.append(0.0)
    return array(indicated)

def G(S, K):
    """the payoff function of put option. Nothing to do with barrier"""
    return max(K-S, 0)
