from tdc.single_pred import Tox
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, MACCSkeys
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pandas as pd

# -- Obtaining the data --

data  = Tox(name='LD50_Zhu')
split = data.get_split()

train_df = split['train']
valid_df = split['valid']
test_df  = split['test']

# --  Featurization --
'''
Translating the molecules identifiers in SMILES (Simplified Molecular Input Line Entry System)
nomenclature into numeric forms legible to the model
'''

def featurize(smiles, radius=2, nBits=512):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(10 + nBits + 167)

    total_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())

   # Molecular descriptors of drugs
    '''
    Characteristics such as number of rings that serve
    '''
    desc = [
        Descriptors.TPSA(mol), # Topilogical Polar Surface are
        Descriptors.MolLogP(mol), # Wildman-Crippen LogP (how hydrophobic it is)
        Descriptors.NumRotatableBonds(mol), # number of rotatable bonds
        Descriptors.NumAromaticRings(mol), # number of aromatic rings
        Descriptors.FractionCSP3(mol), # Fraction of sp3 Carbons
        Descriptors.RingCount(mol), # number of rings
        Descriptors.HeavyAtomCount(mol), # number of non-hydrogen carbons
        Descriptors.NumHAcceptors(mol), # number of Hydrogen acceptors (O,N,F)
        Descriptors.NumHDonors(mol), # number of hydrogen donors (Lone pairs),
        total_charge # Net charge of molecule
    ]

    # Morgan Finger Print
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=512)
    fp_array = np.array(fp)

    # MACCS keys
    maccs = MACCSkeys.GenMACCSKeys(mol)
    maccs_array = np.array(maccs)

    return np.concatenate([desc, fp_array, maccs_array])

X_train = np.stack(train_df['Drug'].apply(featurize))
X_valid = np.stack(valid_df['Drug'].apply(featurize))
X_test  = np.stack(test_df['Drug'].apply(featurize))

y_train = train_df['Y'].values
y_valid = valid_df['Y'].values
y_test  = test_df['Y'].values

# ---- Feature names ---
'''
Turns features into names to later allow the top
features displayed to be visualized in a legible form
'''
descriptor_names = [
    'TPSA', 'LogP', 'RotatableBonds', 'AromaticRings',
    'FractionCSP3', 'RingCount', 'HeavyAtomCount',
    'HAcceptors', 'HDonors', 'TotalCharge'
]
morgan_names = [f'Morgan_{i}' for i in range(512)]
maccs_names  = [f'MACCS_{i}'  for i in range(167)]
all_feature_names = descriptor_names + morgan_names + maccs_names

# Fit selector on train only, apply to all
selector = VarianceThreshold(threshold=0.02)
X_train = selector.fit_transform(X_train)
X_valid = selector.transform(X_valid)
X_test  = selector.transform(X_test)

# Keeps only the names that survived the selector
selected_mask  = selector.get_support()
selected_names = [all_feature_names[i] for i, kept in enumerate(selected_mask) if kept]

# --- XGBoost regression model ----

xgb_model = xgb.XGBRegressor(
    n_estimators=1200,
    max_depth=5,
    learning_rate=0.04,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=0.4,
    reg_lambda=2,
    random_state=42
)

xgb_model.fit(X_train, y_train)

# -- Running Validation Set --
y_valid_pred = xgb_model.predict(X_valid)

print("=== Validation Set ===")
print("R²:  ", r2_score(y_valid, y_valid_pred))
print("MAE: ", mean_absolute_error(y_valid, y_valid_pred))
print("RMSE:", mean_squared_error(y_valid, y_valid_pred, squared=False))

# --- Running Test Set ---
y_test_pred = xgb_model.predict(X_test)

print("\n=== Final Test Set ===")
print("R²:  ", r2_score(y_test, y_test_pred))
print("MAE: ", mean_absolute_error(y_test, y_test_pred))
print("RMSE:", mean_squared_error(y_test, y_test_pred, squared=False))

# Scatter plot of Test: Predicted LD50 (y-axis) vs actual LD50 (x-axis)

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_test_pred, alpha=0.7)

# Perfect prediction line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--'
)
plt.xlabel("Actual log(LD50)")
plt.ylabel("Predicted log(LD50)")
plt.title("XGBoost: Predicted vs Actual")
plt.grid(True)
plt.show()

# Importance dataframe directly using selected_names
'''
Ranks importance by which features gave the least errors throughout
the XGBoost trees
'''
importances = xgb_model.feature_importances_  # one score per feature
importance_df = pd.DataFrame({
    'Feature': selected_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

# Plot top 20 features

top20 = importance_df.head(20)
plt.figure(figsize=(10, 6))
plt.barh(top20['Feature'][::-1], top20['Importance'][::-1])
plt.xlabel("Importance (Gain)")
plt.title("Top 20 Most Important Features")
plt.tight_layout()
plt.show()

# Residual distrubution
plt.hist(y_test_pred - y_test, bins=40)
plt.xlabel("Prediction Error")
plt.title("Residual Distribution")
plt.show()