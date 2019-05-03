import numpy as np
import sys
sys.path.append('/home/yluan/Documents/SOWFA PostProcessing/SOWFA-Postprocess')
from PostProcess_FieldData import FieldData
sys.path.append('/home/yluan/Documents/ML/TBNN/examples/turbulence')
from turbulencekepspreprocessor import TurbulenceKEpsDataProcessor
from turbulence_example_driver import plot_results
# For Python 2.7, use cpickle
try:
    import cpickle as pickle
except ModuleNotFoundError:
    import pickle

from tbdt_v8 import TBDT  # Tensor Basis Decision Tree algorithm
from tbrf_v4 import TBRF


"""
User Inputs
"""
caseName = 'ALM_N_H_OneTurb'
caseDir = '/media/yluan'
times = 'latest'
fields = ('kResolved', 'kSGSmean', 'epsilonSGSmean', 'nuSGSmean', 'gradUAvg', 'uuPrime2')
# Whether use existing pickle raw field data and/or ML x, y pickle data
useRawPickle, useXYpickle = True, True
read_treefile = False

# Whether confine the domain of interest, useful if mesh is too large
confineBox, plotConfinedBox = True, True
# Only when confineBox and saveFields are True
# Subscript of the confined field file name
confinedFieldNameSub = 'Confined'
# Whether auto generate confine box for each case,
# the confinement is bordered by 1st/2nd refinement zone
boxAutoDim = 'second'  # 'first', 'second', None
# Only when boxAutoDim is False
# Confine box counter-clockwise rotation in x-y plane
boxRot = np.pi/6
# Confine box origin, width, length, height
boxOrig = (0, 0, 0)  # (x, y, z)
boxL, boxW, boxH = 0, 0, 0  # float
# Absolute cap value for Sji and Rij and scalar basis x
capSijRij, capScalarBasis = 1e9, 1e9


"""
TBRF Hyper-parameters
"""
regularization = True
regularization_lambda = 1e-12
write_g = False #write tensor basis coefficient fields
splitting_features='all' # 'all', 'div3', 'sqrt', or an integer value
min_samples_leaf = 9
n_trees = 3
optim_split = True
sampling = True
# Only if sampling is True
# Number of samples and whether to use with replacement, i.e. same sample can be re-picked
fraction, replace = 0.003, False
seed = 12345 # use for reproducibility, set equal to None for no seeding
split_fraction = 0.8  # Fraction of data to use for training



"""
Process User Inputs
"""
# Save pickle fields automatically
saveFields = True if any((useRawPickle, useXYpickle)) else False
inputsEnsembleName = 'Inputs_' + caseName
if times == 'latest':
    if caseName == 'ALM_N_H_ParTurb':
        times = '22000.0918025'
    elif caseName == 'ALM_N_H_OneTurb':
        times = '24995.0788025'

if confineBox and boxAutoDim is not None:
    if caseName == 'ALM_N_H_ParTurb':
        # 1st refinement zone as confinement box
        if boxAutoDim == 'first':
            boxOrig = (1074.225, 599.464, 0)
            boxL, boxW, boxH = 1134, 1134, 405
        # 2nd refinement zone as confinement box
        elif boxAutoDim == 'second':
            boxOrig = (1120.344, 771.583, 0)
            boxL, boxW, boxH = 882, 378, 216
    elif caseName == 'ALM_N_H_OneTurb':
        if boxAutoDim == 'first':
            boxOrig = (948.225, 817.702, 0)
            boxL, boxW, boxH = 1134, 630, 405
        # 2nd refinement zone as confinement box
        elif boxAutoDim == 'second':
            boxOrig = (994.344, 989.583, 0)
            boxL, boxW, boxH = 882, 378, 216


"""
Read Pickle Data
"""
# Set file names for pickle
fileNames = ('Sij', 'Rij', 'x', 'tb', 'y') if not confineBox \
    else \
    ('Sij_' + confinedFieldNameSub, 'Rij_' + confinedFieldNameSub, 'x_' + confinedFieldNameSub, 'tb_' + confinedFieldNameSub, 'y_' + confinedFieldNameSub)

case = FieldData(caseName = caseName, caseDir = caseDir, times = times, fields = fields, save = saveFields)

dataDict = case.readPickleData(fileNames = fileNames, resultPath = case.resultPath[times])
Sij, Rij = dataDict[fileNames[0]], dataDict[fileNames[1]]
x, tb, y = dataDict[fileNames[2]], dataDict[fileNames[3]], dataDict[fileNames[4]]



# ----------- set up the tree filename which can be read later on if necessary
# The tree filenames contain the training data parameters, as well as the random
# forest parameters such as min. features per node etc.
tree_filename_ReCase = 'test'
tree_filename_var = 'test'

# ------------ path where the tree files need to be saved ------------------
tree_filename = './TreeFiles/TBRF_TREE%i_' + tree_filename_ReCase + tree_filename_var



tbrf = TBRF(min_samples_leaf=min_samples_leaf,tree_filename=tree_filename,n_trees=n_trees,regularization=regularization,
            regularization_lambda=regularization_lambda, splitting_features=splitting_features,
            optim_split=optim_split,optim_threshold=100,read_from_file=read_treefile)


# If use sampling
if sampling:
    x, y, tb = tbrf.randomSampling(x.T, y.T, tb.T, fraction = fraction, replace = replace)

x_train, tb_train, y_train, x_test, tb_test, y_test = \
    TurbulenceKEpsDataProcessor.train_test_split(x.T, tb.T, y.T, fraction=split_fraction, seed=seed)
print('\nTrain and test data split')

# Convert every array to nFeatures x nPoints
x_train, x_test, y_train, y_test, tb_train, tb_test = x_train.T, x_test.T, y_train.T, y_test.T, tb_train.T, tb_test.T

forest = tbrf.fit(x_train, y_train, tb_train)

bij_hat, bij_forest, g_forest = tbrf.predict(x_test, tb_test, forest)

# # filter predictions using the median filter (bij_MF), and spatial filter (bij_FMF):
# bij_MF = medianFilteredField(bij_forest, 3, bounds = False)
# bij_FMF = filterField(np.reshape(bij_MF, [3, 3, testParam['Nx'], testParam['Ny']]), [3, 3])
#
# # get predicted tensor basis coefficients for further analysis
# g_MF = medianFilteredField(g_forest, 3, bounds = False)
# g_FMF = filterField(np.reshape(g_MF, [1, 10, testParam['Nx'], testParam['Ny']]), [3, 3])
# g_FMF = np.reshape(g_FMF, [10, testParam['Nx']*testParam['Ny']])

# # No median filter for now
# bij_MF, g_MF = bij_hat, g_forest
#
# RMSE_MF = np.sqrt(np.mean(np.square(bij_MF - y_test)))
# print('RMSE b_ij, median filter: %f' % RMSE_MF)
# RMSE_FMF = np.sqrt(np.mean(np.square(bij_FMF-np.reshape(Y_test,[3,3,testParam['Nx'],testParam['Ny']]))))
# print('RMSE b_ij, median filter + gaussian filter: %f' % RMSE_FMF)


# Plot the results
plot_results(y_test.T, bij_hat.T)
