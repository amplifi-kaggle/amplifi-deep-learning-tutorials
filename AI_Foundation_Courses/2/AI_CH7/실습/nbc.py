Prior_Cavity = {
    True:  0.2, # p(Cavity=True)
    False: 0.8  # p(Cavity=False)
}

CPT_Toothache = {
    (True,  True):  0.6, # p(Toothache=True |Cavity=True)
    (False, True):  0.4, # p(Toothache=False|Cavity=True)
    (True,  False): 0.1, # p(Toothache=True |Cavity=False)
    (False, False): 0.9, # p(Toothache=False|Cavity=False)
}

CPT_Catch = {
    (True,  True):  0.9, # p(Catch=True |Cavity=True)
    (False, True):  0.1, # p(Catch=False|Cavity=True)
    (True,  False): 0.2, # p(Catch=True |Cavity=False)
    (False, False): 0.8, # p(Catch=False|Cavity=False)
}

print CPT_Toothache[(True, True)]

def compute_cavity_prob(Toothache=None, Catch=None):
    # BEGIN_YOUR_CODE
    raise Exception('Not implemented yet')
    # END_YOUR_CODE
    
print 'p(cavity|Toothache=False, Catch=False)', compute_cavity_prob(Toothache=False, Catch=False)
print 'p(cavity|Toothache=False, Catch=True)',  compute_cavity_prob(Toothache=False, Catch=True)
print 'p(cavity|Toothache=True,  Catch=False)', compute_cavity_prob(Toothache=True,  Catch=False)
print 'p(cavity|Toothache=True,  Catch=True)',  compute_cavity_prob(Toothache=True,  Catch=True)
print 'p(cavity|Toothache=True)', compute_cavity_prob(Toothache=True)
print 'p(cavity|Catch=True)', compute_cavity_prob(Catch=True)
print 'p(cavity)', compute_cavity_prob()