# RAM-UR-IRA definition

RAM-UR-IRA is a random attention model with unobservable references and independent random attention. It combines a strict total preference order with an independent random attention function gamma. Gamma maps each alternative x in X to a number in (0, 1]. This number is the independent probability that x receives attention. Some alternatives may have gamma equal to one; those alternatives are attention-privileged and can represent reference alternatives.

# Regularity and axioms

The characterization of RAM-UR-IRA uses R-Asymmetry (R-ASYM), R-Independence (R-IND), Non-Triviality (NT), Existence of a Dominant Alternative (EDA*), and Regularity (REG). Regularity states that if menu A is a subset of menu B, then p(a,A) is at least p(a,B) for each alternative a in A. Thus adding alternatives cannot increase the choice probability of an alternative that remains available.

# Choice probability formula

For alternative x in menu A, RAM-UR-IRA assigns choice probability p(x,A) = gamma(x) multiplied by the product of (1 - gamma(y)) for every y in A that is preferred to x. The attention parameter gives each alternative's attention probability, while the strict preference order determines which alternatives can block selection of x.
