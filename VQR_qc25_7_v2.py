# VQR (Variational Quantum Regressor)

# v2: VQR deo koristi tačno 5 ulaznih brojeva -> 5 qubita (usklađeno sa qc25: 5×5=25 u demo kolu, bez 7×5=35).
# v2: predikcija i dalje za svih 7 pozicija (6. i 7. kao u QCBM obrascu, bez dodatnog bloka od 5 qubita).


"""
https://github.com/forsing
https://github.com/forsing?tab=repositories
"""

"""
Loto Skraceni Sistemi
https://www.lotoss.info
ABBREVIATED LOTTO SYSTEMS
"""


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from IPython.display import display
from IPython.display import clear_output


from qiskit_machine_learning.utils import algorithm_globals
import random

# =========================
# Seed za reproduktivnost
# =========================
SEED = 39
np.random.seed(SEED)
random.seed(SEED)
algorithm_globals.random_seed = SEED


"""
svih 4586 izvlacenja Loto 7/39 u Srbiji
30.07.1985.- 24.03.2026.
"""

# 1. Učitaj loto podatke
df = pd.read_csv("/Users/4c/Desktop/GHQ/data/loto7_4586_k24.csv", header=None)


###################################
print()
print("Prvih 5 ucitanih kombinacija iz CSV fajla:")
print()
print(df.head())
print()
"""

Prvih 5 ucitanih kombinacija iz CSV fajla:

    0   1   2   3   4   5   6
0   5  14  15  17  28  30  34
1   2   3  13  18  19  23  37
2  13  17  18  20  21  26  39
3  17  20  23  26  35  36  38
4   3   4   8  11  29  32  37

"""

print()
print("Zadnjih 5 ucitanih kombinacija iz CSV fajla:")
print()
print(df.tail())
print()
"""

Zadnjih 5 ucitanih kombinacija iz CSV fajla:

       0   1   2   3   4   5   6
4581   1   5  11  14  15  25  39
4582   7  22  23  30  31  34  38
4583   1   8  11  12  29  36  39
4584  17  20  27  30  31  36  37
4585   1  11  25  27  31  32  39

"""
####################################


# 2. Minimalni i maksimalni dozvoljeni brojevi po poziciji
min_val = [1, 2, 3, 4, 5, 6, 7]
max_val = [33, 34, 35, 36, 37, 38, 39]

# 3. Funkcija za mapiranje brojeva u indeksirani opseg [0..range_size-1]
def map_to_indexed_range(df, min_val, max_val):
    df_indexed = df.copy()
    for i in range(df.shape[1]):
        df_indexed[i] = df[i] - min_val[i]
        # Provera da li su svi brojevi u validnom opsegu
        if not df_indexed[i].between(0, max_val[i] - min_val[i]).all():
            raise ValueError(f"Vrednosti u koloni {i} nisu u opsegu 0 do {max_val[i] - min_val[i]}")
    return df_indexed

# 4. Primeni mapiranje
df_indexed = map_to_indexed_range(df, min_val, max_val)

# 5. Provera rezultata
print()
print(f"Učitano kombinacija: {df.shape[0]}, Broj pozicija: {df.shape[1]}")
print()
"""
Učitano kombinacija: 4586, Broj pozicija: 7
"""


print()
print("Prvih 5 mapiranih kombinacija:")
print()
print(df_indexed.head())
print()
"""

Prvih 5 mapiranih kombinacija:

    0   1   2   3   4   5   6
0   4  12  12  13  23  24  27
1   1   1  10  14  14  17  30
2  12  15  15  16  16  20  32
3  16  18  20  22  30  30  31
4   2   2   5   7  24  26  30

"""

print()
print("Zadnjih 5 mapiranih kombinacija:")
print()
print(df_indexed.tail())
print()
"""

Zadnjih 5 mapiranih kombinacija:

       0   1   2   3   4   5   6
4581   0   3   8  10  10  19  32
4582   6  20  20  26  26  28  31
4583   0   6   8   8  24  30  32
4584  16  18  24  26  26  30  30
4585   0   9  22  23  26  26  32

"""




# Parametri
num_qubits = 5          # 5 qubita po poziciji
num_layers = 2          # Dubina varijacionog sloja
num_positions = 5       # 6 pozicija (brojeva) u loto kombinaciji

def encode_position(value):
    """
    Sigurno enkoduje 'value' u QuantumCircuit sa tacno num_qubits qubita.
    Ako value zahteva vise bitova od num_qubits, koristi se LSB (zadnjih num_qubits bitova),
    i ispisuje se upozorenje.
    """
    # osiguraj int
    v = int(value)
    bin_full = format(v, 'b')  # pravi binarni bez vodećih nula
    if len(bin_full) > num_qubits:
        # upozorenje: vrednost ne staje u broj qubita; koristimo zadnjih num_qubits bita (LSB)
        print(f"Upozorenje: value={v} zahteva {len(bin_full)} bitova, a num_qubits={num_qubits}. Koristim zadnjih {num_qubits} bita.")
        bin_repr = bin_full[-num_qubits:]
    else:
        bin_repr = bin_full.zfill(num_qubits)

    qc = QuantumCircuit(num_qubits)
    # reversed da bi LSB išao na qubit 0 (ako želiš suprotno, ukloni reversed)
    for i, bit in enumerate(reversed(bin_repr)):
        if bit == '1':
            qc.x(i)
    return qc





# Varijacioni sloj: Ry rotacije + CNOT lanac
def variational_layer(params):
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.ry(params[i], i)
    for i in range(num_qubits - 1):
        qc.cx(i, i+1)
    return qc

# QCBM ansambl: slojevi varijacionih blokova
def qcbm_ansatz(params):
    qc = QuantumCircuit(num_qubits)
    for layer in range(num_layers):
        start = layer * num_qubits
        end = (layer + 1) * num_qubits
        qc.compose(variational_layer(params[start:end]), inplace=True)
    return qc

# Kompletan QCBM za svih 7 pozicija
def full_qcbm(params_list, values):
    total_qubits = num_qubits * num_positions
    qc = QuantumCircuit(total_qubits)

    for pos in range(num_positions):
        start_q = pos * num_qubits
        end_q = start_q + num_qubits

        # Enkoduj vrednost za poziciju
        qc_enc = encode_position(values[pos])
        qc.compose(qc_enc, qubits=range(start_q, end_q), inplace=True)

        # Dodaj varijacioni ansambl
        qc_var = qcbm_ansatz(params_list[pos])
        qc.compose(qc_var, qubits=range(start_q, end_q), inplace=True)

    # Dodaj merenja za svih 30 qubita
    qc.measure_all()

    return qc


test_values = [5, 10, 15, 20, 25, 30, 35]
np.random.seed(39)
params_list = [np.random.uniform(0, 2*np.pi, num_layers * num_qubits) for _ in range(num_positions)]

# Generiši QCBM za svih 6 pozicija
full_circuit = full_qcbm(params_list, test_values)



# Prikaz celog kruga u 'mpl' formatu
full_circuit.draw('mpl')
# plt.show()

# fold=40 prelama linije tako da veliki krug stane na ekran.
full_circuit.draw('mpl', fold=40)
# plt.show()



# Kompaktni prikaz kola
print("\nKompaktni prikaz kvantnog kola (text):\n")
# print(full_circuit.draw('text'))
"""
Kompaktni prikaz kvantnog kola (text):


Upozorenje: value=38 zahteva 6 bitova, a num_qubits=5. Koristim zadnjih 5 bita.
Upozorenje: value=35 zahteva 6 bitova, a num_qubits=5. Koristim zadnjih 5 bita.
"""


# display(full_circuit.draw())     
display(full_circuit.draw("mpl"))
# plt.show()


# v2: uklonjeno circuit_drawer(..., output='latex') i sav LaTeX izlaz.


"""
# v2: Sačuvaj kao PDF / latex jpg / latex png — uklonjeno (draw('latex')).

# Sačuvaj kao sliku u matplotlib formatu jpg
img4 = full_circuit.draw('mpl', fold=40)
img4.savefig("/data/qc25_5_4.jpg")

# Sačuvaj kao sliku u matplotlib formatu png
img5 = full_circuit.draw('mpl', fold=40)
img5.savefig("/data/qc25_5_5.png")





# Sačuvaj kao sliku u matplotlib formatu jpg
img4 = full_circuit.draw('mpl', fold=40)
img4.savefig("/KvantniRegresor/1VQR/VQR_qc25_7_4.jpg")


"""


###############################################







import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from qiskit_aer import AerSimulator
from qiskit_machine_learning.algorithms import NeuralNetworkRegressor
from qiskit_machine_learning.utils import algorithm_globals

from qiskit_algorithms.optimizers import COBYLA, SPSA
from qiskit_algorithms.optimizers import ADAM

from tqdm import tqdm
import random

from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN

from qiskit_machine_learning.optimizers import GradientDescent

from qiskit_aer.primitives import Sampler as AerSampler

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

import matplotlib.pyplot as plt

from qiskit_machine_learning.algorithms import VQR


# =========================
# 2. Koristimo svih N=4586 izvlacenja
# =========================
# v2: za kraće čekanje — samo poslednjih 100 izvlačenja u VQR delu (povećaj N po potrebi).
N = 100
df = df.tail(N).reset_index(drop=True)



X = df.iloc[:, :5].values  # prvih 5 brojeva (5 qubita u feature map-u; ne 6 iz :-1)
y_full = df.values          # svi 7 brojeva

# Skaliranje
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X).astype(np.float64)

# =========================
# Treniranje i predikcija po brojevima
# =========================

predicted_combination = []
print()
for i in range(7):  # 5 brojeva + dodatni broj
    print(f"\n--- Treniranje QNN regresora za broj {i+1} ---")
    y = y_full[:, i].astype(np.float64)
    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1,1)).ravel()

    # SamplerQNN sa lokalnim AerSimulator-om
    backend = AerSimulator()

    num_qubits = X_scaled.shape[1]
    print()
    print("\nnum_qubits")
    print(num_qubits, "\n")
    print()
    



    sampler = Sampler()

    
    gradient = GradientDescent()  # param-shift rule




    # -------------------------
    # Feature map sa num_qubits parametara
    # -------------------------
    param_x = ParameterVector("x", num_qubits)
    feature_map = QuantumCircuit(num_qubits, name="fm")

    for j in range(num_qubits):
        feature_map.ry(param_x[j], j)

    feature_map.barrier()
    for j in range(num_qubits - 1):
        feature_map.cz(j, j+1)
    feature_map.cz(num_qubits-1, 0)

    feature_map.draw("mpl", style="clifford")
    # plt.show()

    feature_map.decompose().draw(output="mpl", style="clifford", fold=20)
    # plt.show()

    # -------------------------
    # Ansatz sa num_qubits parametara
    # -------------------------
    param_y = ParameterVector("y", num_qubits)
    ansatz = QuantumCircuit(num_qubits, name="vf")

    for j in range(num_qubits):
        ansatz.ry(param_y[j], j)

    ansatz.barrier()
    for j in range(num_qubits - 1):
        ansatz.cz(j, j+1)
    ansatz.cz(num_qubits-1, 0)

    ansatz.draw("mpl", style="clifford")
    # plt.show()

    ansatz.decompose().draw(output="mpl", style="clifford", fold=20)
    # plt.show()




    # 3. Spoji ih u jedan parametarski krug
    full_circuit_map = feature_map.compose(ansatz)
    # full_circuit = feature_map.compose(feature_map)


    full_circuit_map.draw("mpl", style="clifford", fold=20)
    # plt.show()

    
    """
    # VQR is a special variant of the NeuralNetworkRegressor with a EstimatorQNN
    
    # -------------------------
    # QNN (sada eksplicitno prosleđujemo parametre)
    # -------------------------
    regression_estimator_qnn = EstimatorQNN(
        circuit=full_circuit_map,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        gradient=gradient
    )
    """

    
    optimizer = COBYLA(maxiter=500, tol=1e-7)
    # optimizer = SPSA(maxiter=300)
    # optimizer = ADAM(maxiter=150, lr=0.1)


    total_iters = len(X_scaled)
    pbar = tqdm(total=total_iters, desc=f"Broj {i+1}")

    def progress_callback(weights, loss):
        pbar.update(1)



    vqr = VQR(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        callback=progress_callback
    )

    # VQR built ✅  | ansatz params: 6

    
    # create empty array for callback to store evaluations of the objective function
    objective_func_vals = []
    plt.rcParams["figure.figsize"] = (12, 6)

    # Fit sa progres bar
    vqr.fit(X_scaled, y_scaled)
    pbar.close()

    # return to default figsize
    plt.rcParams["figure.figsize"] = (6, 4)

    

    # score result
    y_hat_scaled = vqr.predict(X_scaled).reshape(-1, 1)
    y_hat = scaler_y.inverse_transform(y_hat_scaled).ravel()
    print(f"R2 train (skalirano ne, na originalnoj skali): {r2_score(y, y_hat):.6f}")
    print("vqr.score(X_scaled, y_scaled)")
    print(vqr.score(X_scaled, y_scaled), "\n")
    
    
    print("vqr.weights")
    print(vqr.weights, "\n")
    




    # plot data
    plt.plot(X_scaled, y_scaled, "bo")
    plt.title(f"Broj {i+1} - Podaci")
    plt.xlabel("Ulazni podaci (prvih 5 brojeva)")
    plt.ylabel(f"Izlazni podaci (broj {i+1})")
    plt.grid()
    # plt.show()




    # Predikcija sledećeg broja
    last_scaled = scaler_X.transform([X[-1]]).astype(np.float64)
    pred_scaled = vqr.predict(last_scaled)
    pred = scaler_y.inverse_transform(pred_scaled.reshape(-1,1))[0][0]
    pred = max(1, min(39, int(round(pred))))  # Loto 7/39: 1–39

    predicted_combination.append(pred)
    print(f"Predikcija za broj {i+1}: {pred}")

print()
print("\n=== Predviđena sledeća loto kombinacija (5+2) ===")
print(" ".join(str(num) for num in predicted_combination))
print()
"""

100 zadnjih kombinacija iz CSV fajla za trening VQR modela

--- Treniranje QNN regresora za broj 1 ---


num_qubits
5 


Broj 1:   0%|                         | 0/100 [00:00<?, ?it/s]No gradient function provided, creating a gradient function. If your Estimator requires transpilation, please provide a pass manager.
Broj 1:   0%|                         | 0/100 [00:04<?, ?it/s]
R2 train (skalirano ne, na originalnoj skali): -0.552140
vqr.score(X_scaled, y_scaled)
-0.552139629888347 

vqr.weights
[1.57081743 1.57078063 1.57078702 1.5708043  1.57077948] 

Predikcija za broj 1: 1

--- Treniranje QNN regresora za broj 2 ---


num_qubits
5 


Broj 2:   0%|                         | 0/100 [00:00<?, ?it/s]No gradient function provided, creating a gradient function. If your Estimator requires transpilation, please provide a pass manager.
Broj 2:   0%|                         | 0/100 [00:22<?, ?it/s]
R2 train (skalirano ne, na originalnoj skali): 0.629389
vqr.score(X_scaled, y_scaled)
0.629389435152274 

vqr.weights
[ 2.45210611 -0.10330801  1.60156109  0.00987067 -0.60751679] 

Predikcija za broj 2: 14

--- Treniranje QNN regresora za broj 3 ---


num_qubits
5 


Broj 3:   0%|                         | 0/100 [00:00<?, ?it/s]No gradient function provided, creating a gradient function. If your Estimator requires transpilation, please provide a pass manager.
Broj 3:   0%|                         | 0/100 [00:16<?, ?it/s]
R2 train (skalirano ne, na originalnoj skali): 0.822169
vqr.score(X_scaled, y_scaled)
0.8221686577817968 

vqr.weights
[-0.34880931 -0.05764464 -1.47978826 -0.06160044 -0.68736035] 

Predikcija za broj 3: x

--- Treniranje QNN regresora za broj 4 ---


num_qubits
5 


/Users/4c/qiskit_env/lib/python3.11/site-packages/qiskit/visualization/circuit/matplotlib.py:287: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`). Consider using `matplotlib.pyplot.close()`.
  mpl_figure = plt.figure()
Broj 4:   0%|                         | 0/100 [00:00<?, ?it/s]No gradient function provided, creating a gradient function. If your Estimator requires transpilation, please provide a pass manager.
Broj 4:   0%|                         | 0/100 [00:21<?, ?it/s]
R2 train (skalirano ne, na originalnoj skali): 0.569435
vqr.score(X_scaled, y_scaled)
0.5694351725176028 

vqr.weights
[ 2.90738105 -0.37834306  1.99242286 -0.29318158 -1.20413257] 

Predikcija za broj 4: y

--- Treniranje QNN regresora za broj 5 ---


num_qubits
5 


Broj 5:   0%|                         | 0/100 [00:00<?, ?it/s]No gradient function provided, creating a gradient function. If your Estimator requires transpilation, please provide a pass manager.
Broj 5:   0%|                         | 0/100 [00:21<?, ?it/s]
R2 train (skalirano ne, na originalnoj skali): 0.580152
vqr.score(X_scaled, y_scaled)
0.5801518731638741 

vqr.weights
[ 2.86326629 -0.63840625 -0.34322865  2.21006137 -1.05039392] 

Predikcija za broj 5: z

--- Treniranje QNN regresora za broj 6 ---


num_qubits
5 


Broj 6:   0%|                         | 0/100 [00:00<?, ?it/s]No gradient function provided, creating a gradient function. If your Estimator requires transpilation, please provide a pass manager.
Broj 6:   0%|                         | 0/100 [00:20<?, ?it/s]
R2 train (skalirano ne, na originalnoj skali): 0.396694
vqr.score(X_scaled, y_scaled)
0.3966939113827871 

vqr.weights
[-0.10642198 -0.43615932 -0.62291633 -0.23718084 -1.31031314] 

Predikcija za broj 6: 33

--- Treniranje QNN regresora za broj 7 ---


num_qubits
5 


Broj 7:   0%|                         | 0/100 [00:00<?, ?it/s]No gradient function provided, creating a gradient function. If your Estimator requires transpilation, please provide a pass manager.
Broj 7:   0%|                         | 0/100 [00:22<?, ?it/s]
R2 train (skalirano ne, na originalnoj skali): -0.139619
vqr.score(X_scaled, y_scaled)
-0.1396190783262954 

vqr.weights
[-0.38130837 -0.14178675 -0.64893084 -0.48115589 -0.84471522] 

Predikcija za broj 7: 36


=== Predviđena sledeća loto kombinacija (5+2) ===
1 14 x y z 33 36

"""





"""
U VQR delu num_qubits sada prati tačno 5 ulaznih kolona: X = df.iloc[:, :5] → 5 qubita u feature map / ansatz.
Petlja i dalje trenira 7 izlaza (6. i 7. broj bez širenja kola na 7×5).
COBYLA(maxiter=500, tol=1e-7), ispis R² na originalnoj skali.
"""
